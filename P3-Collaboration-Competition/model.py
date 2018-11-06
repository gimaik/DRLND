import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=[512, 256]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1_units = fc_units[0]
        self.fc2_units = fc_units[1]

        self.fc1 = nn.Linear(state_size, self.fc1_units)
        self.fc2 = nn.Linear(self.fc1_units, self.fc2_units)
        self.fc3 = nn.Linear(self.fc2_units, action_size)
        self.bn1 = nn.BatchNorm1d(state_size)                
        self.reset_parameters()

    def reset_parameters(self):
        init_weights(self.fc1)
        init_weights(self.fc2)
        init_weights(self.fc3)
        
                
    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.elu(self.fc1(state))
        x = F.elu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=[512, 256]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.fc1_units = fc_units[0]
        self.fc2_units = fc_units[1]
        
        self.bn1 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, self.fc1_units)        
        self.fc2 = nn.Linear(self.fc1_units + action_size, self.fc2_units)        
        self.fc3 = nn.Linear(self.fc2_units, 1)  
        self.reset_parameters()

    def reset_parameters(self):
        init_weights(self.fc1)
        init_weights(self.fc2)
        init_weights(self.fc3)

    def forward(self, state, actions):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""        
        xs = F.elu(self.fc1(state))
        x = torch.cat((xs, actions), dim=1)
        x = F.elu(self.fc2(x))
        return self.fc3(x)