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
from wandb import env
from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
    TransposeMinAtarObs
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
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    decouple_learning: bool = False

    # Algorithm specific arguments
    env_id: str = "MinAtar/Asterix-v1"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
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
    two_time_scale: bool = False
    """whether to use two-time-scale learning for the value and advantage networks"""
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
    eval_frequency: int = 1000
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
        env = TransposeMinAtarObs(env)

        # env = NoopResetEnv(env, noop_max=30)
        # env = MaxAndSkipEnv(env, skip=4)
        # env = EpisodicLifeEnv(env)
        # if "FIRE" in env.unwrapped.get_action_meanings():
        #     env = FireResetEnv(env)
        # env = ClipRewardEnv(env)
        # env = gym.wrappers.ResizeObservation(env, (84, 84))
        # env = gym.wrappers.GrayScaleObservation(env)
        # env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk

class AdvNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        in_channels = env.single_observation_space.shape[0]
        self.advnetwork = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=env.single_action_space.n)
        )

    def forward(self, x):
        return self.advnetwork(x)

    def greedy_actions(self, x):
        return torch.argmax(self.advnetwork(x), dim=1)

class ValueNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        in_channels = env.single_observation_space.shape[0]
        self.valuenetwork = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1)
        )

    def forward(self, x):
        return self.valuenetwork(x)


    def state_values(self, x):
        return self.valuenetwork(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def parse_eval_seeds(eval_seeds: str) -> list[int]:
    if not eval_seeds.strip():
        return []
    return [int(seed.strip()) for seed in eval_seeds.split(",") if seed.strip()]

def evaluate_q_network(adv_network,value_network, env_id, eval_seeds, device, gamma):
    adv_network.eval()
    value_network.eval()
    episodic_returns = []
    episodic_lengths = []
    average_overestimations = []
    start_overestimations = []
    with torch.no_grad():
        for eval_seed in eval_seeds:
            env = gym.make(env_id)
            env = TransposeMinAtarObs(env)
            env.action_space.seed(eval_seed)
            obs, _ = env.reset(seed=eval_seed)
            done = False
            episodic_return = 0.0
            rewards = []
            state_value_estimates = []
            truncated = False
            while not done:
                obs_tensor = torch.Tensor(np.array([obs])).to(device)
                state_value_estimates.append(float(value_network(obs_tensor).cpu().numpy()[0]))
                action = int(adv_network.greedy_actions(obs_tensor).cpu().numpy()[0])
                obs, reward, terminated, truncated, _ = env.step(action)
                episodic_return += float(reward)
                rewards.append(float(reward))
                done = terminated or truncated
            bootstrap_value = 0.0
            if truncated:
                bootstrap_value = float(value_network(torch.Tensor(np.array([obs])).to(device)).cpu().numpy()[0])
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
    adv_network.train()
    value_network.train()
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
            dir=args.wandb_path
        )
    writer = SummaryWriter(args.eval_results_path + f"/runs/{run_name}")
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

    adv_network = AdvNetwork(envs).to(device)
    value_network = ValueNetwork(envs).to(device)

    value_lr = args.learning_rate * args.value_lr_multiplier

    adv_optimizer = optim.Adam(
        adv_network.parameters(),
        lr=args.learning_rate,)

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
    if args.eval_results_path:
        if os.path.isdir(args.eval_results_path):
            # Keeps runs organized inside your custom directory: /scratch/work/.../run_name/eval_results.jsonl
            eval_results_path = args.eval_results_path + f"/runs/{run_name}/eval_results.jsonl"
        else:
            # If a full file path was explicitly provided (e.g., .../custom_filename.jsonl)
            eval_results_path = args.eval_results_path
    else:
        # Default fall-back path
        eval_results_path = f"runs/{run_name}/eval_results.jsonl"
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
    # obs = obs.astype(np.float32) 
    
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)

        adv_values = adv_network(torch.Tensor(obs).to(device))
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
        # next_obs = next_obs.astype(np.float32)
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
        # print(f"Adding to replay buffer: obs shape {obs.shape}, next_obs shape {real_next_obs.shape}")
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos, action_probs)

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
                
                selected_advantages = advantages.gather(1, data.actions).squeeze()

                value_reg = torch.square(values)  
                adv_reg = torch.sum(torch.square(advantages), dim=-1) 

                current_q = values + selected_advantages

                td_loss = F.mse_loss(current_q, q_target)
                with torch.no_grad():
                    delta = q_target - current_q
                    # mean_abs_delta = delta.abs().mean() + 1e-8
                    # td_gate = (delta.abs()/ (delta.abs() + mean_abs_delta))
                    rarity = (1.0 / data.action_probs.flatten().clamp(min=1e-3))
                    rarity = torch.clamp(rarity, max=args.max_rarity)
                    importance = (args.beta * rarity)


                advantage_target = selected_advantages.detach() + importance.detach() * delta.detach()
                extra_advantage_loss = F.mse_loss(selected_advantages, advantage_target)

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
                
                adv_optimizer.step()
                value_optimizer.step()

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

    # if args.save_model:
    #     model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
    #     torch.save(q_network.state_dict(), model_path)
    #     print(f"model saved to {model_path}")
    #     from cleanrl_utils.evals.dqn_eval import evaluate

    #     episodic_returns = evaluate(
    #         model_path,
    #         make_env,
    #         args.env_id,
    #         eval_episodes=10,
    #         run_name=f"{run_name}-eval",
    #         Model=QNetwork,
    #         device=device,
    #         epsilon=args.end_e,
    #     )
    #     for idx, episodic_return in enumerate(episodic_returns):
    #         writer.add_scalar("eval/episodic_return", episodic_return, idx)

    #     if args.upload_model:
    #         from cleanrl_utils.huggingface import push_to_hub

    #         repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
    #         repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
    #         push_to_hub(args, episodic_returns, repo_id, "AVL", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
