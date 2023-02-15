import torch
import torch.nn as nn


class SimpleNet(nn.Module):

    def __init__(self, _input_size):
        super(SimpleNet, self).__init__()
        self.in_size = _input_size

        # First linear layer and non-linearity
        self.linear1 = nn.Linear(self.in_size, 1024)
        self.relu1 = nn.ReLU()

        # Second linear layer and non-linearity
        self.linear2 = nn.Linear(1024, 2048)
        self.relu2 = nn.ReLU()

        # Third layer and non-linearity
        self.linear3 = nn.Linear(2048, 128)
        self.relu3 = nn.ReLU()

        # Classification layer - here 2 classes, but binary predictor, thus 0/1
        self.linear4 = nn.Linear(128, 1)
        self.activation = nn.Sigmoid()

    def forward(self, _input):
        x = self.linear1(_input)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        out = self.activation(x)
        return out


class SimpleDropNet(nn.Module):

    def __init__(self, _input_size, _drop):
        super(SimpleDropNet, self).__init__()
        self.in_size = _input_size

        # First linear layer and non-linearity
        self.linear1 = nn.Linear(self.in_size, 1024)
        self.drop1 = nn.Dropout(_drop)
        self.relu1 = nn.ReLU()

        # Second linear layer and non-linearity
        self.linear2 = nn.Linear(1024, 2048)
        self.drop2 = nn.Dropout(_drop)
        self.relu2 = nn.ReLU()

        # Third layer and non-linearity
        self.linear3 = nn.Linear(2048, 128)
        self.drop3 = nn.Dropout(_drop)
        self.relu3 = nn.ReLU()

        # Classification layer - here 2 classes, but binary predictor, thus 0/1
        self.linear4 = nn.Linear(128, 1)
        self.activation = nn.Sigmoid()

    def forward(self, _input):
        x = self.linear1(_input)
        x = self.drop1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.drop3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        out = self.activation(x)
        return out
