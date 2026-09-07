import os
import random
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import TimeLimit

from cleanrl_utils.atari_wrappers import TransposeMinAtarObs
from cleanrl_utils.buffers import ProbReplayBuffer


# NumPy 2.x compatibility for older replay-buffer code that may still reference np.float_.
if not hasattr(np, "float_"):
    np.float_ = np.float64


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, torch.backends.cudnn.deterministic is enabled"""
    cuda: bool = True
    """if toggled, CUDA will be enabled when available"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the W&B project name"""
    wandb_entity: Optional[str] = None
    """the W&B entity/team"""
    wandb_path: Optional[str] = None
    """optional W&B local directory"""
    capture_video: bool = False
    """whether to capture videos"""
    save_model: bool = False
    """whether to save the final model"""

    # Environment / DQN arguments. Defaults intentionally stay close to the
    # attached RDQ MinAtar implementation so comparisons are apples-to-apples.
    env_id: str = "MinAtar/Asterix-v1"
    """the Gymnasium environment id"""
    total_timesteps: int = 10_000_000
    """total environment timesteps"""
    learning_rate: float = 1e-4
    """Adam learning rate"""
    num_envs: int = 1
    """number of vectorized environments; this script currently supports 1"""
    buffer_size: int = 1_000_000
    """replay memory size"""
    gamma: float = 0.99
    """discount factor"""
    tau: float = 1.0
    """target-network Polyak coefficient; 1.0 is a hard copy"""
    use_target_network: bool = False
    """use a target network for bootstrapping; kept False to match the attached RDQ file"""
    target_network_frequency: int = 1000
    """environment-step frequency of target-network updates"""
    batch_size: int = 32
    """replay minibatch size"""
    start_e: float = 1.0
    """initial epsilon for epsilon-greedy exploration"""
    end_e: float = 0.01
    """final epsilon"""
    exploration_fraction: float = 0.10
    """fraction of total timesteps over which epsilon is annealed"""
    learning_starts: int = 80_000
    """number of transitions collected before learning begins"""
    random_steps: int = 0
    """initial steps that are uniformly random regardless of epsilon"""
    train_frequency: int = 4
    """perform one gradient update every this many environment steps"""

    # Mean-expansion / IB-DQN argument.
    mean_scaling_coefficient: float = -1.0
    """ME-layer coefficient k; negative means use k=n_actions (the paper's default)"""

    # Periodic evaluation / logging arguments carried over from the RDQ file.
    eval_frequency: int = 200_000
    """evaluate every eval_frequency environment steps; 0 disables periodic evaluation"""
    eval_seeds: str = "0,1,2,3,4"
    """comma-separated evaluation seeds"""
    eval_results_path: str = ""
    """directory for runs/eval JSONL, or an explicit .jsonl file for evaluation output"""
    progress_file: str = ""
    """path to write lightweight progress events as JSON lines"""

class MeanExpansionLayer(nn.Module):
    """Parameter-free mean-expansion layer from Nagarajan et al. (2026).

    Given residual vector z and coefficient k:
        q = z + k * mean(z) * 1

    Equivalently, this scales z's mean component by (k + 1) while leaving
    all mean-orthogonal action differences unchanged.
    """

    def __init__(self, mean_scaling_coefficient: float):
        super().__init__()
        if mean_scaling_coefficient < 0:
            raise ValueError("mean_scaling_coefficient k must be >= 0")
        self.register_buffer(
            "scale",
            torch.tensor(1.0 + float(mean_scaling_coefficient), dtype=torch.float32),
        )

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        mean = vec.mean(dim=-1, keepdim=True)
        residual = vec - mean
        return self.scale * mean + residual

class QNetwork(nn.Module):
    """MinAtar Q-network whose final action vector is passed through the ME layer.

    The learned linear output is z(s, .), the residual representation.  The
    externally visible output is q(s, .) = M_k z(s, .), so all action selection,
    bootstrapping, and TD losses operate on actual Q-values.
    """

    def __init__(self, env: gym.vector.VectorEnv, mean_scaling_coefficient: float):
        super().__init__()
        in_channels = env.single_observation_space.shape[0]
        n_actions = env.single_action_space.n

        # Same MinAtar feature extractor/head shape as the attached RDQ networks.
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=n_actions),
        )
        self.mean_expansion = MeanExpansionLayer(mean_scaling_coefficient)

    def residuals(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pre-ME residual vector z(s, .)."""
        return self.network(x)

    def q_from_residuals(self, z: torch.Tensor) -> torch.Tensor:
        return self.mean_expansion(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_from_residuals(self.residuals(x))

    def greedy_actions(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.forward(x), dim=1)

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = TransposeMinAtarObs(env)
        env = TimeLimit(env, max_episode_steps=5000)

        env.action_space.seed(seed)
        return env

    return thunk


def linear_schedule(start_e: float, end_e: float, duration: float, t: int) -> float:
    if duration <= 0:
        return end_e
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)



def parse_eval_seeds(eval_seeds: str) -> list[int]:
    if not eval_seeds.strip():
        return []
    return [int(seed.strip()) for seed in eval_seeds.split(",") if seed.strip()]

def evaluate_q_network(
    q_network: QNetwork,
    env_id: str,
    eval_seeds: list[int],
    device: torch.device,
    gamma: float,
):
    """Greedy evaluation plus the overestimation diagnostics used by the RDQ file.

    In a single Q-network there is no separate V stream, so the state estimate
    used for this diagnostic is max_a Q(s,a), which is also the greedy action's
    predicted action-value.
    """

    q_network.eval()
    episodic_returns: list[float] = []
    episodic_lengths: list[int] = []
    average_overestimations: list[float] = []
    start_overestimations: list[float] = []

    with torch.no_grad():
        for eval_seed in eval_seeds:
            env = gym.make(env_id)
            env = TransposeMinAtarObs(env)
            env = TimeLimit(env, max_episode_steps=5000)
            env.action_space.seed(eval_seed)
            obs, _ = env.reset(seed=eval_seed)

            done = False
            episodic_return = 0.0
            rewards: list[float] = []
            greedy_q_estimates: list[float] = []
            truncated = False

            while not done:
                obs_tensor = torch.as_tensor(
                    np.asarray([obs]), dtype=torch.float32, device=device
                )
                q_values = q_network(obs_tensor)
                greedy_q, greedy_action = q_values.max(dim=1)
                greedy_q_estimates.append(float(greedy_q.item()))

                obs, reward, terminated, truncated, _ = env.step(int(greedy_action.item()))
                episodic_return += float(reward)
                rewards.append(float(reward))
                done = bool(terminated or truncated)

            # If TimeLimit truncated the episode, use the learned value of the
            # final state as the empirical-return bootstrap, matching the spirit
            # of the diagnostic in the attached file.
            bootstrap_value = 0.0
            if truncated:
                final_obs_tensor = torch.as_tensor(
                    np.asarray([obs]), dtype=torch.float32, device=device
                )
                bootstrap_value = float(q_network(final_obs_tensor).max(dim=1).values.item())

            returns: list[float] = []
            discounted_return = bootstrap_value
            for reward in reversed(rewards):
                discounted_return = reward + gamma * discounted_return
                returns.append(discounted_return)
            returns.reverse()

            overestimations = [
                estimate - empirical_return
                for estimate, empirical_return in zip(greedy_q_estimates, returns)
            ]

            env.close()
            episodic_returns.append(episodic_return)
            episodic_lengths.append(len(rewards))
            average_overestimations.append(float(np.mean(overestimations)))
            start_overestimations.append(float(overestimations[0]))

    q_network.train()
    return (
        episodic_returns,
        episodic_lengths,
        average_overestimations,
        start_overestimations,
    )

def write_eval_result(path: str, result: dict):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(result) + "\n")



def write_progress_event(path: str, event: dict):
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(event) + "\n")

if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    time_name = int(time.time())
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time_name}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            dir=args.wandb_path
        )
    writer = SummaryWriter(args.eval_results_path + f"/runs/{args.exp_name}/{args.env_id}__{args.seed}__{time_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    run_dir = args.eval_results_path + f"/runs/{args.exp_name}/{args.env_id}__{args.seed}__{time_name}"
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    n_actions = envs.single_action_space.n
    me_k = (
        float(n_actions)
        if args.mean_scaling_coefficient < 0
        else float(args.mean_scaling_coefficient)
    )
    if me_k < 0:
        raise ValueError("mean_scaling_coefficient must be >= 0, or negative to request k=n")

    writer.add_scalar("config/mean_scaling_coefficient", me_k, 0)
    writer.add_scalar("config/mean_expansion_scale", 1.0 + me_k, 0)
    print(f"IB-DQN mean-expansion coefficient: k={me_k:g} (n_actions={n_actions})")

    q_network = QNetwork(envs, me_k).to(device)
    target_network = QNetwork(envs, me_k).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(
        q_network.parameters(),
        lr=args.learning_rate)

    rb = ProbReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )

    start_time = time.time()
    eval_seeds = parse_eval_seeds(args.eval_seeds)

    if args.eval_results_path:
        if os.path.isdir(args.eval_results_path):
            # Keeps runs organized inside your custom directory: /scratch/work/.../run_name/eval_results.jsonl
            eval_results_path = args.eval_results_path + f"/runs/{args.exp_name}/{args.env_id}__{args.seed}__{time_name}/eval_results.jsonl"
        else:
            # If a full file path was explicitly provided (e.g., .../custom_filename.jsonl)
            eval_results_path = args.eval_results_path
    else:
        # Default fall-back path
        eval_results_path = f"runs/{run_name}/eval_results.jsonl"

    # Write a config that includes the resolved k (important when the CLI value
    # is the -1 sentinel meaning k=n_actions).
    # config_to_save = asdict(args)
    # config_to_save["resolved_mean_scaling_coefficient"] = me_k
    # config_to_save["n_actions"] = n_actions
    # with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
    #     json.dump(config_to_save, f, indent=2)

    write_progress_event(
        args.progress_file,
        {
            "event": "started",
            "global_step": 0,
            "total_timesteps": args.total_timesteps,
            "algorithm": args.exp_name,
            "train_seed": args.seed,
        },
    )
    def run_periodic_eval(global_step):
        if args.eval_frequency <= 0 or not eval_seeds:
            return
        episodic_returns, episodic_lengths, average_overestimations, start_overestimations = evaluate_q_network(
            q_network,
            args.env_id,
            eval_seeds,
            device,
            args.gamma,
        )
        result = {
            "global_step": global_step,
            "env_id": args.env_id,
            "algorithm": args.exp_name,
            "train_seed": args.seed,
            "eval_seeds": eval_seeds,
            "episodic_returns": episodic_returns,
            "episodic_lengths": episodic_lengths,
            "mean_return": float(np.mean(episodic_returns)),
            "std_return": float(np.std(episodic_returns)),
            "mean_length": float(np.mean(episodic_lengths)),
            "std_length": float(np.std(episodic_lengths)),
            "average_overestimations": average_overestimations,
            "start_overestimations": start_overestimations,
            "mean_average_overestimation": float(np.mean(average_overestimations)),
            "mean_start_overestimation": float(np.mean(start_overestimations)),
            "num_eval_episodes": len(episodic_returns),
        }
        write_eval_result(eval_results_path, result)
        write_progress_event(args.progress_file, {"event": "eval", "total_timesteps": args.total_timesteps, **result})
        writer.add_scalar("eval/mean_return", result["mean_return"], global_step)
        writer.add_scalar("eval/mean_average_overestimation", result["mean_average_overestimation"], global_step)
        writer.add_scalar("eval/mean_start_overestimation", result["mean_start_overestimation"], global_step)

    # Start interaction.
    obs, _ = envs.reset(seed=args.seed)
    run_periodic_eval(0)

    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )

        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            q_values = q_network(obs_tensor)
            greedy_actions = torch.argmax(q_values, dim=1).cpu().numpy()

        if global_step < args.random_steps or random.random() < epsilon:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions = greedy_actions

        # ProbReplayBuffer in the attached setup records behavior-policy action
        # probabilities. IB-DQN itself does not use them, but we preserve the
        # buffer API so this file is drop-in compatible with that setup.
        if global_step < args.random_steps:
            action_probs = np.full(
                envs.num_envs, 1.0 / n_actions, dtype=np.float32
            )
        else:
            action_probs = np.where(
                actions == greedy_actions,
                (1.0 - epsilon) + (epsilon / n_actions),
                epsilon / n_actions,
            ).astype(np.float32)

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(
                        f"global_step={global_step}, "
                        f"episodic_return={info['episode']['r']}"
                    )
                    writer.add_scalar(
                        "charts/episodic_return", info["episode"]["r"], global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", info["episode"]["l"], global_step
                    )

        # Preserve the terminal observation on TimeLimit truncation.  Only true
        # terminations are stored as done, matching the attached buffer usage.
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        rb.add(
            obs,
            real_next_obs,
            actions,
            rewards,
            terminations,
            infos,
            action_probs,
        )
        obs = next_obs

        # DQN / IB-DQN update.  The ONLY algorithmic change from an ordinary
        # Q-network is that q_network(...) includes the parameter-free ME layer.
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)

            with torch.no_grad():
                bootstrap_network = target_network if args.use_target_network else q_network
                max_next_q = bootstrap_network(data.next_observations).max(dim=1).values
                q_target = (
                    data.rewards.flatten()
                    + args.gamma * max_next_q * (1.0 - data.dones.flatten())
                )

            # Compute z once so we can both construct q and log the implicit
            # baseline / lower-norm representation without a second forward pass.
            z = q_network.residuals(data.observations)
            q_all = q_network.q_from_residuals(z)
            current_q = q_all.gather(1, data.actions.long()).squeeze(1)

            td_loss = F.mse_loss(current_q, q_target)

            optimizer.zero_grad()
            td_loss.backward()
            optimizer.step()

            if global_step % 100 == 0:
                with torch.no_grad():
                    z_mean = z.mean(dim=-1)
                    implicit_baseline = me_k * z_mean
                    action_gaps = (
                        torch.topk(q_all, k=min(2, n_actions), dim=1).values
                    )
                    if n_actions >= 2:
                        mean_action_gap = (action_gaps[:, 0] - action_gaps[:, 1]).mean()
                    else:
                        mean_action_gap = torch.zeros((), device=device)

                writer.add_scalar("losses/td_loss", td_loss.item(), global_step)
                writer.add_scalar("values/q_mean", q_all.mean().item(), global_step)
                writer.add_scalar("values/q_max_mean", q_all.max(dim=1).values.mean().item(), global_step)
                writer.add_scalar("values/residual_mean", z.mean().item(), global_step)
                writer.add_scalar(
                    "values/residual_l2_mean",
                    torch.linalg.vector_norm(z, dim=1).mean().item(),
                    global_step,
                )
                writer.add_scalar(
                    "values/implicit_baseline_mean",
                    implicit_baseline.mean().item(),
                    global_step,
                )
                writer.add_scalar("values/action_gap_mean", mean_action_gap.item(), global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
                sps = int(global_step / max(time.time() - start_time, 1e-8))
                print("SPS:", sps)
                writer.add_scalar("charts/SPS", sps, global_step)

        # Target-network update.  This stays outside the gradient-update block so
        # its cadence is expressed in environment steps exactly as in the RDQ file.
        if (
            args.use_target_network
            and global_step > args.learning_starts
            and global_step % args.target_network_frequency == 0
        ):
            with torch.no_grad():
                for target_param, online_param in zip(
                    target_network.parameters(), q_network.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * online_param.data
                        + (1.0 - args.tau) * target_param.data
                    )

        completed_step = global_step + 1
        if args.eval_frequency > 0 and completed_step % args.eval_frequency == 0:
            run_periodic_eval(completed_step)

    if args.eval_frequency > 0 and args.total_timesteps % args.eval_frequency != 0:
        run_periodic_eval(args.total_timesteps)

    write_progress_event(
        args.progress_file,
        {
            "event": "finished",
            "global_step": args.total_timesteps,
            "total_timesteps": args.total_timesteps,
            "algorithm": args.exp_name,
            "train_seed": args.seed,
            "mean_scaling_coefficient": me_k,
        },
    )

    if args.save_model:
        model_path = os.path.join(run_dir, f"{args.exp_name}.cleanrl_model")
        torch.save(
            {
                "q_network": q_network.state_dict(),
                "args": asdict(args),
                "resolved_mean_scaling_coefficient": me_k,
                "n_actions": n_actions,
            },
            model_path,
        )
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()