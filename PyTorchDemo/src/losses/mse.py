import torch
import torch.nn as nn


def mse_loss():
    mse = nn.MSELoss()
    outs = torch.randn(3, 5, requires_grad=True)
    print(outs)
    trgt = torch.randn(3, 5)
    print(trgt)
    loss = mse(outs, trgt)
    print(loss.detach())


if __name__ == "__main__":
    mse_loss()
