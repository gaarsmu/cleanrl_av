"""Separate-network RPQ on Atari (Gymnasium 0.29).

Timesteps count agent decisions, each repeating an action for four ALE frames.
Training uses clipped rewards and episodic lives; evaluation reports raw full-game
scores. Value-error diagnostics use clipped rewards and stop returns at life loss.
The 108000-frame time limit is terminal in both training and evaluation. Defaults and
RPQ losses follow rpq_separate_network_minatar.py; target networks remain optional.
"""
import os
import random
import json
import time
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from cleanrl_utils.buffers import ProbReplayBuffer



if not hasattr(np, "float_"):
    np.float_ = np.float64


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    wandb_path: str = None
    """the path to the wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """unsupported for this implementation; use save_model for a local checkpoint"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    remove_one: bool = False
    """whether to remove one in advantage networks update"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """agent decisions (four ALE frames per decision, excluding reset actions)"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    beta: float = 1.0
    """beta factor in  Residual-Preconditioned RDQ algorithm"""
    l2_coef: float = 5e-3
    """l2 regularization coefficient"""
    tau: float = 1.0
    """the target network update rate"""
    value_lr_multiplier: float = 1.0
    """the learning rate multiplier for the value network"""
    adv_lr_multiplier: float = 1.0
    """the learning rate multiplier for the advantage network"""
    beta_scheduling: bool = False
    """whether to use beta scheduling for the Residual-Preconditioned RDQ algorithm"""
    beta_final: float = 1.0
    """the final beta value for the Residual-Preconditioned RDQ algorithm"""
    beta_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-beta to go end-beta"""
    max_rarity: float = 5.0
    """maximum rarity value to prevent extreme importance weights"""
    use_target_network: bool = False
    """whether to use a separate target network for bootstrapping"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    random_steps: int = 0
    """number of initial environment steps with uniformly random actions"""
    train_frequency: int = 4
    """the frequency of training"""
    eval_frequency: int = 200000
    """evaluate every eval_frequency environment steps; 0 disables periodic evaluation"""
    eval_seeds: str = "0,1,2,3,4"
    """comma-separated evaluation seeds used at every evaluation point"""
    eval_epsilon: float = 0.001
    """epsilon for full-game evaluation; 0 selects fully greedy evaluation"""
    eval_results_path: str = ""
    """output directory, or an explicit .jsonl file for evaluation results"""
    progress_file: str = ""
    """path to write lightweight progress events as JSON lines"""

def make_env(env_id, seed, idx, capture_video, run_name, evaluation=False):
    def thunk():
        # Explicit frameskip prevents double action repetition for ALE/*-v5 IDs.
        kwargs = {"frameskip": 1, "repeat_action_probability": 0.0, "max_episode_steps": 108000}
        if capture_video and idx == 0:
            kwargs["render_mode"] = "rgb_array"
        env = gym.make(env_id, **kwargs)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        if not evaluation:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        if not evaluation:
            env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)
        return env

    return thunk


class AdvNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        in_channels = env.single_observation_space.shape[0]
        self.advnetwork = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=3136, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=env.single_action_space.n)
        )

    def forward(self, x):
        return self.advnetwork(x.float() / 255.0)

    def greedy_actions(self, x):
        return torch.argmax(self.advnetwork(x.float() / 255.0), dim=1)

class ValueNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        in_channels = env.single_observation_space.shape[0]
        self.valuenetwork = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=3136, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1)
        )

    def forward(self, x):
        return self.valuenetwork(x.float() / 255.0)


    def state_values(self, x):
        return self.valuenetwork(x.float() / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    if duration <= 0:
        return end_e
    fraction = min(max(t / duration, 0.0), 1.0)
    return start_e + fraction * (end_e - start_e)

def parse_eval_seeds(eval_seeds: str) -> list[int]:
    if not eval_seeds.strip():
        return []
    return [int(seed.strip()) for seed in eval_seeds.split(",") if seed.strip()]

def evaluate_q_network(adv_network,value_network, env_id, eval_seeds, device, gamma, epsilon=0.001):
    adv_network.eval()
    value_network.eval()
    rng = np.random.default_rng(0)
    episodic_returns = []
    episodic_lengths = []
    average_overestimations = []
    start_overestimations = []
    with torch.no_grad():
        for eval_seed in eval_seeds:
            env = make_env(env_id, eval_seed, 0, False, "eval", evaluation=True)()
            env.action_space.seed(eval_seed)
            obs, _ = env.reset(seed=eval_seed)
            done = False
            episodic_return = 0.0
            rewards = []
            state_value_estimates = []
            life_terminals = []
            lives = env.unwrapped.ale.lives()
            truncated = False
            while not done:
                obs_tensor = torch.Tensor(np.array([obs])).to(device)
                state_value_estimates.append(value_network(obs_tensor).item())
                action = int(adv_network.greedy_actions(obs_tensor).cpu().numpy()[0])
                if rng.random() < epsilon:
                    action = int(env.action_space.sample())
                obs, reward, terminated, truncated, _ = env.step(action)
                episodic_return += float(reward)
                rewards.append(float(np.sign(reward)))
                next_lives = env.unwrapped.ale.lives()
                life_terminals.append(terminated or (next_lives < lives and next_lives > 0))
                lives = next_lives
                done = terminated or truncated
            returns = []
            discounted_return = 0.0
            for reward, life_terminal in zip(reversed(rewards), reversed(life_terminals)):
                discounted_return = reward + gamma * discounted_return * (not life_terminal)
                returns.append(discounted_return)
            returns.reverse()
            overestimations = [
                estimate - actual_return
                for estimate, actual_return in zip(state_value_estimates, returns)
            ]
            env.close()
            episodic_returns.append(episodic_return)
            episodic_lengths.append(len(rewards))
            average_overestimations.append(float(np.mean(overestimations)))
            start_overestimations.append(overestimations[0])
    adv_network.train()
    value_network.train()
    return episodic_returns, episodic_lengths, average_overestimations, start_overestimations

def write_eval_result(path, result):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as file:
        file.write(json.dumps(result) + "\n")

def write_progress_event(path, event):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as file:
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
    if args.upload_model:
        raise ValueError("Model upload is not supported; use --save-model for a local checkpoint")
    if args.eval_results_path.endswith(".jsonl"):
        eval_results_path = args.eval_results_path
        output_root = os.path.dirname(os.path.abspath(eval_results_path))
    else:
        output_root = args.eval_results_path or "."
        eval_results_path = None
    run_dir = os.path.join(output_root, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    eval_results_path = eval_results_path or os.path.join(run_dir, "eval_results.jsonl")
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )
    with open(os.path.join(run_dir, "config.json"), "w") as file:
        json.dump(vars(args), file, indent=2)
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

    adv_network = AdvNetwork(envs).to(device)
    value_network = ValueNetwork(envs).to(device)

    value_lr = args.learning_rate * args.value_lr_multiplier
    adv_lr = args.learning_rate * args.adv_lr_multiplier

    adv_optimizer = optim.Adam(
        adv_network.parameters(),
        lr=adv_lr,)

    value_optimizer = optim.Adam(
        value_network.parameters(),
        lr=value_lr,)
        
    adv_target_network = AdvNetwork(envs).to(device)
    adv_target_network.load_state_dict(adv_network.state_dict())

    value_target_network = ValueNetwork(envs).to(device)
    value_target_network.load_state_dict(value_network.state_dict())


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
            adv_network,
            value_network,
            args.env_id,
            eval_seeds,
            device,
            args.gamma,
            args.eval_epsilon,
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

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    run_periodic_eval(0)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        beta = linear_schedule(args.beta, args.beta_final, args.beta_fraction * args.total_timesteps, global_step) if args.beta_scheduling else args.beta

        with torch.no_grad():
            adv_values = adv_network(torch.as_tensor(np.asarray(obs), device=device))
        greedy_actions = torch.argmax(adv_values, dim=1).cpu().numpy()

        if global_step < args.random_steps or random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions = greedy_actions

        n_actions = envs.single_action_space.n
        if global_step < args.random_steps:
            # Pure random initial steps: P(a) = 1 / N
            action_probs = np.full(envs.num_envs, 1.0 / n_actions, dtype=np.float32)
        else:
            # Epsilon-greedy steps:
            # Non-greedy action: epsilon / N
            # Greedy action:     (1 - epsilon) + (epsilon / N)
            action_probs = np.where(
                actions == greedy_actions,
                (1.0 - epsilon) + (epsilon / n_actions),
                epsilon / n_actions
            ).astype(np.float32)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        # Treat the 108000-frame evaluation horizon as terminal. The compact
        # replay buffer shares adjacent observations across episode boundaries.
        dones = np.logical_or(terminations, truncations)
        rb.add(obs, real_next_obs, actions, rewards, dones, infos, action_probs)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        
        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    adv_bootstrap_network = adv_target_network if args.use_target_network else adv_network
                    value_bootstrap_network = value_target_network if args.use_target_network else value_network

                    next_value = value_bootstrap_network(data.next_observations).flatten()
                    next_advantage = adv_bootstrap_network(data.next_observations).max(dim=1).values
                    max_next_q = next_value + next_advantage
                    
                    q_target = (
                        data.rewards.flatten()
                        + args.gamma * max_next_q * (1 - data.dones.flatten())
                    )
                    

                values = value_network(data.observations).flatten()
                
                advantages = adv_network(data.observations)
                
                selected_advantages = advantages.gather(1, data.actions.long()).squeeze(1)

                value_reg = torch.square(values)  
                adv_reg = torch.sum(torch.square(advantages), dim=-1) 

                current_q = (values + selected_advantages.detach() if args.remove_one else values + selected_advantages)

                td_loss = F.mse_loss(current_q, q_target)
                with torch.no_grad():
                    delta = q_target - current_q
                    rarity = (1.0 / data.action_probs.flatten().clamp(min=1e-3))
                    rarity = torch.clamp(rarity, max=args.max_rarity)
                    importance = (beta * rarity)


                advantage_target = selected_advantages.detach() + importance.detach() * delta.detach()
                extra_advantage_loss = 0.5 * F.mse_loss(selected_advantages, advantage_target)

                l2_loss = 0.5 * args.l2_coef * (value_reg + adv_reg).mean()
                loss = td_loss  + l2_loss + extra_advantage_loss

                if global_step % 100 == 0:
                    writer.add_scalar("losses/tdloss", td_loss, global_step)
                    writer.add_scalar("losses/extra_advantage_loss", extra_advantage_loss, global_step)
                    writer.add_scalar("losses/total_loss", loss, global_step)
                    writer.add_scalar("losses/values", values.mean().item(), global_step)
                    writer.add_scalar("losses/advantages", selected_advantages.mean().item(), global_step)
                    writer.add_scalar("losses/l2_loss", l2_loss, global_step)
                    writer.add_scalar("losses/importance", importance.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                adv_optimizer.zero_grad()
                value_optimizer.zero_grad()
                
                loss.backward()
                
                adv_optimizer.step()
                value_optimizer.step()
            # update target network
            if args.use_target_network and global_step % args.target_network_frequency == 0:
                for adv_target_network_param, adv_network_param in zip(adv_target_network.parameters(), adv_network.parameters()):
                    adv_target_network_param.data.copy_(
                        args.tau * adv_network_param.data + (1.0 - args.tau) * adv_target_network_param.data
                    )

                for value_target_network_param, value_network_param in zip(value_target_network.parameters(), value_network.parameters()):
                    value_target_network_param.data.copy_(
                        args.tau * value_network_param.data + (1.0 - args.tau) * value_target_network_param.data
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
        },
    )

    if args.save_model:
        model_path = os.path.join(run_dir, f"{args.exp_name}.cleanrl_model")
        torch.save({
            "adv_network": adv_network.state_dict(),
            "value_network": value_network.state_dict(),
            "args": vars(args),
        }, model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
