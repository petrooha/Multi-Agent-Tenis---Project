# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network
import numpy as np
from agent import Agent, ReplayBuffer
import torch
from utilities import soft_update, transpose_to_tensor, transpose_list
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

BUFFER_SIZE = int(100000)  # replay buffer size
BATCH_SIZE = 64        # minibatch size
GAMMA = 0.99            # discount factor
UPDATE_EVERY = 4 

class MADDPG:
    def __init__(self, state_size, action_size, seed, discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions = 14+2+2+2=20
        self.maddpg_agent = [Agent(state_size, action_size, seed), 
                             Agent(state_size, action_size, seed)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
        self.t_step=0
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)


    def act(self, states):
        """get actions from all agents in the MADDPG object"""
        #actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        actions = [self.maddpg_agent[i].act(states[i]) for i in range(len(self.maddpg_agent))]
        return actions

   
    def reset(self):
        [agent.reset() for agent in self.maddpg_agent]

    def step(self, states, actions, rewards, next_states, done):
        for i in range(len(self.maddpg_agent)):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], done[i])
            
            # Learn every UPDATE_EVERY time steps.
            self.t_step += 1
            if (self.t_step + 1) % UPDATE_EVERY == 0:

            # If enough samples are available in memory, get random subset and learn
                if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample()
                    self.maddpg_agent[i].learn(experiences, GAMMA)
                
                
                
            #self.maddpg_agent[i].step(states[i], actions[i], rewards[i], next_states[i], done[i])



