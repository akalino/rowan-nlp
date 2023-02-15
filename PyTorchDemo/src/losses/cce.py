import torch
import torch.nn as nn


def cce_loss():
    cce = nn.CrossEntropyLoss()
    outs = torch.randn(3, 5, requires_grad=True)
    trgt = torch.tensor([1, 0, 3], dtype=torch.int64)
    loss = cce(outs, trgt)
    print(loss)


if __name__ == "__main__":
    cce_loss()
