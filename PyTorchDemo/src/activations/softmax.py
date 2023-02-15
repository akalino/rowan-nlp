import torch
import matplotlib.pyplot as plt


def plot_softmax():
    softmax = torch.nn.Softmax(dim=1)
    x = torch.randn(1, 3)
    y = softmax(x)
    print(x)
    print(y)
    print(torch.sum(y, dim=1))


if __name__ == "__main__":
    plot_softmax()
