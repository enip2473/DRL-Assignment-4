import gymnasium
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env
from train import SACAgent
ENV_NAME = "cartpole-balance"

env = make_dmc_env(ENV_NAME, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0] # Assumes symmetric action space centered at 0
agent = SACAgent(state_dim, action_dim, action_bound)
agent.load("sac_cartpole.pth")
# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (1,), np.float64)

    def act(self, observation):
        return agent.select_action(observation, evaluate=True)
