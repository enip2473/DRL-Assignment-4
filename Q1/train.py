import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import random
from collections import deque
import time
import matplotlib.pyplot as plt

# --- Hyperparameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

ENV_NAME = "Pendulum-v1"
SEED = 0 # For reproducibility

BUFFER_SIZE = 1_000_0
BATCH_SIZE = 256
GAMMA = 0.99           # Discount factor
TAU = 0.005            # Target network soft update rate
LR_ACTOR = 3e-4        # Actor learning rate
LR_CRITIC = 3e-4       # Critic learning rate
LR_ALPHA = 3e-4        # Temperature alpha learning rate
ALPHA = 0.2            # Initial temperature (if not auto-tuning)
AUTO_TUNE_ALPHA = True # Automatically tune alpha
TARGET_ENTROPY = -1.0  # Target entropy (often -action_dim)

HIDDEN_DIM = 256       # Hidden layer size for networks
TOTAL_TIMESTEPS = 100_000 # Total training steps
START_STEPS = 5000     # Steps before starting training updates
UPDATES_PER_STEP = 1     # How many gradient updates per env step
LOG_INTERVAL = 1000    # Log progress every X steps
EVAL_INTERVAL = 5000   # Evaluate policy every X steps
EVAL_EPISODES = 5      # Number of episodes for evaluation

# For reproducibility
# Note: CUDA operations can still introduce some non-determinism
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    # Consider torch.backends.cudnn.deterministic = True
    # and torch.backends.cudnn.benchmark = False
    # for more determinism, potentially at the cost of performance.


# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(states).to(DEVICE),
            torch.FloatTensor(actions).to(DEVICE),
            torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(next_states).to(DEVICE),
            torch.FloatTensor(dones).unsqueeze(1).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)

# --- Networks ---
LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPSILON = 1e-6 # For numerical stability in log_prob calculation

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, action_bound):
        super(Actor, self).__init__()

        # Convert action_bound to tensor if it's not already
        if isinstance(action_bound, (int, float)):
            action_bound = torch.tensor([action_bound], dtype=torch.float32)
        elif isinstance(action_bound, np.ndarray):
             action_bound = torch.from_numpy(action_bound.astype(np.float32))
        # Ensure action_bound is on the correct device later if using GPU
        self.register_buffer('action_bound', action_bound) # Register as buffer

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Define the separate output heads for mean and log_std
        self.mean_fc = nn.Linear(hidden_dim, action_dim)
        self.log_std_fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        # Pass state through the shared backbone
        x = self.backbone(state)
        # Calculate mean and log_std from the backbone output
        mean = self.mean_fc(x)
        log_std = self.log_std_fc(x)

        # Clamp log_std for stability
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def sample(self, state):
        # Get distribution parameters from the forward pass
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        # Sample using reparameterization trick (for gradient flow)
        x_t = normal.rsample()

        # Apply Tanh squashing to bound the action to [-1, 1]
        y_t = torch.tanh(x_t)

        # Scale the action to the environment's action range
        # Make sure action_bound is on the same device as y_t
        action = y_t * self.action_bound.to(y_t.device)

        # Calculate log probability with Tanh correction
        # Formula from Spinning Up / CleanRL implementation:
        log_prob = normal.log_prob(x_t)
        # Correcting the log probability for the Tanh transformation
        log_prob -= torch.log((1 - y_t.pow(2)) + EPSILON)
        # Sum log probabilities across action dimensions if action_dim > 1
        log_prob = log_prob.sum(1, keepdim=True)

        # Calculate the deterministic action (Tanh of mean) for evaluation
        mean_action = torch.tanh(mean) * self.action_bound.to(mean.device)

        return action, log_prob, mean_action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()

        # Q1 architecture defined using nn.Sequential
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 architecture defined using nn.Sequential
        # Note: This creates a separate set of layers and parameters from q1
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        # Concatenate state and action for input
        # Ensure tensors are on the same device and have compatible shapes
        # Usually state: [batch_size, state_dim], action: [batch_size, action_dim]
        sa = torch.cat([state, action], dim=1) # Use dim=1 for batch processing

        # Pass the concatenated input through each sequential network
        q1_val = self.q1(sa)
        q2_val = self.q2(sa)

        return q1_val, q2_val

    def Q1(self, state, action):
        """Helper method to get only the Q1 value."""
        sa = torch.cat([state, action], dim=1)
        q1_val = self.q1(sa)
        return q1_val


# --- SAC Agent ---
class SACAgent:
    def __init__(self, state_dim, action_dim, action_bound):
        self.action_bound = torch.FloatTensor([action_bound]).to(DEVICE)

        # Actor Network
        self.actor = Actor(state_dim, action_dim, HIDDEN_DIM, self.action_bound).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        # Critic Networks
        self.critic = Critic(state_dim, action_dim, HIDDEN_DIM).to(DEVICE)
        self.critic_target = Critic(state_dim, action_dim, HIDDEN_DIM).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict()) # Initialize target same as main
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # Temperature Alpha
        if AUTO_TUNE_ALPHA:
            self.target_entropy = torch.tensor(TARGET_ENTROPY, dtype=torch.float32, device=DEVICE)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ALPHA)
            self.alpha = self.log_alpha.exp().detach() # detach for critic update use
        else:
            self.alpha = torch.tensor(ALPHA, dtype=torch.float32, device=DEVICE)

        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

    def select_action(self, state, evaluate=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        if evaluate:
             # During evaluation, use the mean of the distribution for deterministic action
            _, _, action_mean = self.actor.sample(state_tensor)
            action = action_mean # Use Tanh'ed mean
        else:
            # During training, sample from the distribution
            action, _, _ = self.actor.sample(state_tensor)

        return action.detach().cpu().numpy()[0] # Return numpy array for env interaction

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return None, None, None # Not enough samples yet

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # --- Critic Update ---
        with torch.no_grad():
            # Get next actions and log probs from current policy
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            # Get next Q values from target critics
            q1_target_next, q2_target_next = self.critic_target(next_states, next_actions)
            min_q_target_next = torch.min(q1_target_next, q2_target_next)
            # Calculate target Q value (Bellman target)
            target_q = rewards + (1.0 - dones) * GAMMA * (min_q_target_next - self.alpha * next_log_probs)

        # Get current Q estimates
        current_q1, current_q2 = self.critic(states, actions)

        # Calculate Critic loss (MSE)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimize the Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        # --- Actor Update ---
        # Freeze Critic networks to avoid gradient flow during actor update
        for p in self.critic.parameters():
            p.requires_grad = False

        # Sample actions and log probs for actor loss calculation
        pi_actions, pi_log_probs, _ = self.actor.sample(states)
        # Get Q values for these actions from one of the current critics
        q1_pi, q2_pi = self.critic(states, pi_actions)
        min_q_pi = torch.min(q1_pi, q2_pi) # Use min for stability? common practice varies. Can use just q1_pi.

        # Calculate Actor loss
        actor_loss = (self.alpha * pi_log_probs - min_q_pi).mean() # Maximize Q - alpha*log_prob -> Minimize alpha*log_prob - Q

        # Optimize the Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze Critic networks
        for p in self.critic.parameters():
            p.requires_grad = True


        # --- Alpha (Temperature) Update ---
        alpha_loss = None
        if AUTO_TUNE_ALPHA:
            # Need gradients for log_alpha calculation
            # Use detached log_probs from actor update to prevent double backprop?
            # Re-evaluate log_probs without detaching for alpha loss calculation
            # Need actions generated for actor loss calculation
            # Use pi_actions and pi_log_probs from Actor update section

            # Make sure log_probs aren't detached here
            alpha_loss_tensor = -(self.log_alpha.exp() * (pi_log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss_tensor.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().detach() # Update alpha value for next critic update
            alpha_loss = alpha_loss_tensor.item() # for logging


        # --- Target Network Update (Soft Update) ---
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

        return critic_loss.item(), actor_loss.item(), alpha_loss


    def save(self, filename):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha if AUTO_TUNE_ALPHA else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if AUTO_TUNE_ALPHA else None,
        }, filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=DEVICE)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        if AUTO_TUNE_ALPHA and checkpoint['log_alpha'] is not None:
             self.log_alpha = checkpoint['log_alpha']
             self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
             self.alpha = self.log_alpha.exp().detach()
        print(f"Model loaded from {filename}")


# --- Evaluation Function ---
def evaluate_policy(env, agent, episodes=5):
    total_reward = 0.0
    for _ in range(episodes):
        state, _ = env.reset(seed=random.randint(0, 1e6)) # Use different seeds for eval episodes
        episode_reward = 0.0
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = agent.select_action(state, evaluate=True) # Use deterministic action
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
        total_reward += episode_reward
    return total_reward / episodes


# --- Main Training ---
if __name__ == "__main__":
    # Create environment
    env = gym.make(ENV_NAME) # Use render_mode="human" to watch
    eval_env = gym.make(ENV_NAME) # Separate env for evaluation

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0] # Assumes symmetric action space centered at 0

    print(f"State dim: {state_dim}, Action dim: {action_dim}, Action bound: {action_bound}")

    agent = SACAgent(state_dim, action_dim, action_bound)

    state, _ = env.reset(seed=SEED)
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    total_updates = 0

    rewards_history = []
    eval_rewards_history = []
    steps_history = []

    start_time = time.time()
    best_eval_reward = -10000

    for t in range(1, TOTAL_TIMESTEPS + 1):
        episode_timesteps += 1

        if t < START_STEPS:
            # Sample random action before training starts
            action = env.action_space.sample()
        else:
            # Sample action from policy
            action = agent.select_action(state)

        # Perform action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated # Treat truncation as done for buffer storage

        # Store data in replay buffer
        agent.replay_buffer.add(state, action, reward, next_state, float(done)) # Store done as float (0.0 or 1.0)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= START_STEPS:
            for _ in range(UPDATES_PER_STEP):
                 update_result = agent.update(BATCH_SIZE)
                 if update_result:
                     total_updates += 1
                     c_loss, a_loss, alpha_l = update_result


        if done:
            duration = time.time() - start_time
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Updates: {total_updates} Alpha: {agent.alpha.item():.4f} Duration: {duration:.2f}s")

            rewards_history.append(episode_reward)
            steps_history.append(t+1)

            # Reset environment
            state, _ = env.reset(seed=SEED + episode_num + 1) # Vary seed per episode slightly
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            start_time = time.time() # Reset timer for next episode duration

        # Evaluate policy periodically
        if t % EVAL_INTERVAL == 0 and t >= START_STEPS:
            avg_eval_reward = evaluate_policy(eval_env, agent, episodes=EVAL_EPISODES)
            eval_rewards_history.append(avg_eval_reward)
            print("-" * 40)
            print(f"Evaluation over {EVAL_EPISODES} episodes: {avg_eval_reward:.3f}")
            print("-" * 40)
            # Optionally save model
            if avg_eval_reward > best_eval_reward:
                agent.save(f"sac_pendulum.pth")
                best_eval_reward = avg_eval_reward


    env.close()
    eval_env.close()

    # --- Plotting ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(steps_history, rewards_history)
    plt.title('Episode Reward over Time Steps')
    plt.xlabel('Time Steps')
    plt.ylabel('Episode Reward')

    plt.subplot(1, 2, 2)
    eval_steps = np.arange(EVAL_INTERVAL, TOTAL_TIMESTEPS + 1, EVAL_INTERVAL)
    # Make sure lengths match if plotting stops before last eval interval
    eval_steps = eval_steps[:len(eval_rewards_history)]
    plt.plot(eval_steps, eval_rewards_history)
    plt.title('Average Evaluation Reward over Time Steps')
    plt.xlabel('Time Steps')
    plt.ylabel('Avg Eval Reward')

    plt.tight_layout()
    plt.show()