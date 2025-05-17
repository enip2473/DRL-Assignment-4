import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tqdm import tqdm

env = make_dmc_env("humanoid-walk", np.random.randint(0, 1000000), flatten=True, use_pixels=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_SIZE = 512
BUFFER_SIZE = 1000000
LR = 3e-4
GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 256

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(HIDDEN_SIZE, action_size)
        self.log_std_layer = nn.Linear(HIDDEN_SIZE, action_size)

    def forward(self, state):
        x = self.backbone(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_size + action_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_size + action_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)

class SACAgent:
    def __init__(self, state_size=67, action_size=21):
        self.actor = Actor(state_size, action_size).to(device)
        self.critic = Critic(state_size, action_size).to(device)
        self.target_critic = Critic(state_size, action_size).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.target_entropy = torch.tensor(-action_size, dtype=torch.float32).to(device)
        self.log_alpha = torch.tensor(-1.0, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-4)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR)
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(BUFFER_SIZE, device=device), 
            batch_size=BATCH_SIZE
        )

    def update(self):
        batch = self.replay_buffer.sample()
        state, action, reward, next_state, done = (
            batch["state"],
            batch["action"],
            batch["reward"],
            batch["next_state"],
            batch["done"],
        )
        alpha = self.log_alpha.exp()

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1, target_q2 = self.target_critic(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
            q_target = reward + (1 - done) * GAMMA * target_q.squeeze(-1)

        current_q1, current_q2 = self.critic(state, action)
        current_q1 = current_q1.squeeze(-1)
        current_q2 = current_q2.squeeze(-1)
        
        critic_loss = F.mse_loss(current_q1, q_target) + F.mse_loss(current_q2, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_action, log_prob = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        return actor_loss, critic_loss

    def select_action(self, state, deterministic=False):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state)
        return action[0].cpu().numpy()
    
    def save(self, filename):
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
        }
        torch.save(save_dict, filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        print(f"Model loaded from {filename}")

NUM_EPISODES = 5000
WARMUP_EPISODES = 50

def eval_actor(env, agent, episode=20):
    eval_scores = []
    for _ in range(episode):
        state, _ = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.select_action(state, deterministic=True)
            state, reward, done, truncated, _ = env.step(action)
            score += reward
            done = done or truncated
        eval_scores.append(score)
    mean = np.mean(eval_scores)
    std = np.std(eval_scores)
    final_score = mean - std
    print(f"Mean: {mean:.2f} | Std: {std:.2f} | Score: {final_score:.2f}")
    return final_score

if __name__ == "__nain__":
    agent = SACAgent(67, 21)
    score_deque = deque(maxlen=100)
    best_eval_score = -10000

    for episode in tqdm(range(NUM_EPISODES)):
        state, _ = env.reset()
        score = 0
        done = False
        while not done:
            if episode <= WARMUP_EPISODES:
                action = np.random.uniform(-1.0, 1.0, size=21)
            else:
                action = agent.select_action(state)

            next_state, reward, done, truncated, _ = env.step(action)
            agent.replay_buffer.add(
                {
                    "state": torch.from_numpy(state).float(),
                    "action": torch.from_numpy(action).float(),
                    "reward": torch.tensor(reward, dtype=torch.float32),
                    "next_state": torch.from_numpy(next_state).float(),
                    "done": torch.tensor(done, dtype=torch.int32),
                }
            )

            if episode > WARMUP_EPISODES:
                agent.update()

            state = next_state
            score += reward
            done = done or truncated

        score_deque.append(score)
        
        if (1 + episode) % 10 == 0:
            print(f"Episode {episode + 1} Score: {np.mean(score_deque):.2f}")

        if (1 + episode) % 100 == 0:
            eval_score = eval_actor(env, agent)
            if eval_score > best_eval_score:
                print("Saving models...")
                agent.save("sac_humanoid.pth")
                best_eval_score = eval_score