import torch
import torch.nn as nn


def bce_loss():
    bce = nn.BCELoss()
    sigmoid = nn.Sigmoid()
    x = torch.randn(4, 1, requires_grad=True)
    #print(x)
    probs = sigmoid(x)
    #print(probs)
    trgt = torch.tensor([1, 0, 1, 0], dtype=torch.float32).view(4, 1)
    #print(trgt)
    loss = bce(probs, trgt)
    print(loss)


if __name__ == "__main__":
    bce_loss()
