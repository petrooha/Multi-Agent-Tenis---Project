import numpy as np
import torch
from agent import Agent, ReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size

class MadAgent():
    def __init__(self, num_agents, state_size, action_size, seed):
        self.num_agents = num_agents
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.agents = [Agent(state_size, action_size, self.memory, BATCH_SIZE, seed) for i in range(num_agents)]
    
    def reset(self):
        [agent.reset() for agent in self.agents]
    
    def step(self, states, actions, rewards, next_states, dones):
        [self.agents[i].step(states[i], actions[i], rewards[i], next_states[i], dones[i]) for i in range(self.num_agents)]
    
    def act(self, states):
        actions = [self.agents[i].act(np.array([states[i]])) for i in range(self.num_agents)]
        return actions
    
    def save(self):
        torch.save(self.agents[0].actor_local.state_dict(), 'checkpoint_actor_1.pth')
        torch.save(self.agents[0].critic_local.state_dict(), 'checkpoint_critic_1.pth')
        torch.save(self.agents[1].actor_local.state_dict(), 'checkpoint_actor_2.pth')
        torch.save(self.agents[1].critic_local.state_dict(), 'checkpoint_critic_2.pth')

