import gymnasium as gym
import numpy as np
from train import SACAgent

env = gym.make("Pendulum-v1") # Use render_mode="human" to watch
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0] # Assumes symmetric action space centered at 0
agent = SACAgent(state_dim, action_dim, action_bound)
agent.load("sac_pendulum.pth")

class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        # Pendulum-v1 has a Box action space with shape (1,)
        # Actions are in the range [-2.0, 2.0]
        self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)

    def act(self, observation):
        action = agent.select_action(observation, evaluate=True)
        return action