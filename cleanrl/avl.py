# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
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

from cleanrl_utils.buffers import ReplayBuffer


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
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    use_target_network: bool = False
    """whether to use a separate target network for bootstrapping"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    random_steps: int = 0
    """number of initial environment steps with uniformly random actions"""
    train_frequency: int = 10
    """the frequency of training"""
    eval_frequency: int = 0
    """evaluate every eval_frequency environment steps; 0 disables periodic evaluation"""
    eval_seeds: str = "0,1,2,3,4"
    """comma-separated evaluation seeds used at every evaluation point"""
    eval_epsilon: float = 0.0
    """deprecated; periodic evaluation is fully greedy"""
    eval_results_path: str = ""
    """path to write periodic evaluation results as JSON lines"""
    progress_file: str = ""
    """path to write lightweight progress events as JSON lines"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.n_actions = env.single_action_space.n
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.LayerNorm(120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.LayerNorm(84),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(84, 1)
        self.advantage_head = nn.Linear(84, self.n_actions)

    def forward(self, x):
        return self.value(x) + self.advantage(x)

    def value(self, x):
        hidden = self.network(x)
        return self.value_head(hidden)

    def advantage(self, x):
        hidden = self.network(x)
        return self.advantage_head(hidden)

    def greedy_actions(self, x):
        return torch.argmax(self.advantage(x), dim=1)

    def state_values(self, x):
        return self.value(x).flatten()


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def parse_eval_seeds(eval_seeds: str) -> list[int]:
    if not eval_seeds.strip():
        return []
    return [int(seed.strip()) for seed in eval_seeds.split(",") if seed.strip()]


def evaluate_q_network(q_network, env_id, eval_seeds, device, gamma):
    q_network.eval()
    episodic_returns = []
    episodic_lengths = []
    average_overestimations = []
    start_overestimations = []
    with torch.no_grad():
        for eval_seed in eval_seeds:
            env = gym.make(env_id)
            env.action_space.seed(eval_seed)
            obs, _ = env.reset(seed=eval_seed)
            done = False
            episodic_return = 0.0
            rewards = []
            state_value_estimates = []
            truncated = False
            while not done:
                obs_tensor = torch.Tensor(np.array([obs])).to(device)
                state_value_estimates.append(float(q_network.state_values(obs_tensor).cpu().numpy()[0]))
                action = int(q_network.greedy_actions(obs_tensor).cpu().numpy()[0])
                obs, reward, terminated, truncated, _ = env.step(action)
                episodic_return += float(reward)
                rewards.append(float(reward))
                done = terminated or truncated
            bootstrap_value = 0.0
            if truncated:
                bootstrap_value = float(q_network.state_values(torch.Tensor(np.array([obs])).to(device)).cpu().numpy()[0])
            returns = []
            discounted_return = bootstrap_value
            for reward in reversed(rewards):
                discounted_return = reward + gamma * discounted_return
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
    q_network.train()
    return episodic_returns, episodic_lengths, average_overestimations, start_overestimations


def write_eval_result(path, result):
    with open(path, "a") as file:
        file.write(json.dumps(result) + "\n")


def write_progress_event(path, event):
    if not path:
        return
    with open(path, "a") as file:
        file.write(json.dumps(event) + "\n")


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

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

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    eval_seeds = parse_eval_seeds(args.eval_seeds)
    eval_results_path = args.eval_results_path or f"runs/{run_name}/eval_results.jsonl"
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

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if global_step < args.random_steps or random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

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
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    bootstrap_network = target_network if args.use_target_network else q_network
                    next_value = bootstrap_network.value(data.next_observations).flatten()
                    current_value_target = q_network.value(data.observations).flatten()
                    advantages_target = q_network.advantage(data.observations)
                    selected_advantage_target = advantages_target.gather(1, data.actions).squeeze()
                    max_advantage_target = advantages_target.max(dim=1).values
                    td_error_target = (
                        data.rewards.flatten()
                        + args.gamma * next_value * (1 - data.dones.flatten())
                        - current_value_target
                    )
                    value_target = (
                        data.rewards.flatten()
                        + args.gamma * next_value * (1 - data.dones.flatten())
                        - (selected_advantage_target - max_advantage_target)
                    )

                values = q_network.value(data.observations).flatten()
                advantages = q_network.advantage(data.observations)
                selected_advantages = advantages.gather(1, data.actions).squeeze()
                advantage_loss = F.mse_loss(td_error_target, selected_advantages)
                value_loss = F.mse_loss(value_target, values)
                loss = advantage_loss + value_loss

                if global_step % 100 == 0:
                    writer.add_scalar("losses/advantage_loss", advantage_loss, global_step)
                    writer.add_scalar("losses/value_loss", value_loss, global_step)
                    writer.add_scalar("losses/total_loss", loss, global_step)
                    writer.add_scalar("losses/values", values.mean().item(), global_step)
                    writer.add_scalar("losses/advantages", selected_advantages.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if args.use_target_network and global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
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
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=args.end_e,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "AVL", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
