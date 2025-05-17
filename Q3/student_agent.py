import gymnasium as gym
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env
from train import SACAgent
ENV_NAME = "humanoid-walk"

env = make_dmc_env(ENV_NAME, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0] # Assumes symmetric action space centered at 0
agent = SACAgent(state_dim, action_dim)
agent.load("sac_humanoid.pth")
# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)

    def act(self, observation):
        return agent.select_action(observation, True)
