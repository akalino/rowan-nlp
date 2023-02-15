import torch
import torch.nn as nn


class Perceptron(nn.Module):  # nn.Module creates the computational graph

    def __init__(self, input_dim):
        super(Perceptron, self).__init__()
        self.fully_connected = nn.Linear(input_dim, 1)  # Linear layers organized as (input, output)

    def forward(self, _input):
        # the forward pass is doing the following 4 operations, written as a
        # one liner in the return line (line 18)
        #x = _input
        #x = self.fully_connected(x)
        #x = torch.sigmoid(x)
        #x = x.squeeze()
        return torch.sigmoid(self.fully_connected(_input)).squeeze()

