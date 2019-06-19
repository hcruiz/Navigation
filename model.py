import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            
        NOTE: skeleton provided by Udacity
        """
        super(QNetwork, self).__init__()
        try:
            self.seed = torch.manual_seed(seed)
        except:
            print('Seed is set to default None.')
        self.l_in = nn.Linear(state_size,50)
        self.hidden = nn.Linear(50,50)
        self.l_out = nn.Linear(50,action_size)
        self.activ = nn.ReLU()

    def forward(self, state):
        """Build a network that maps state -> action values."""
        state = self.l_in(state)
        state = self.activ(state)
        state = self.hidden(state)
        state = self.activ(state)
        state = self.l_out(state)
        return state
