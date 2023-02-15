import torch
import matplotlib.pyplot as plt


def plot_tanh():
    x = torch.range(-5.0, 5.0, 0.1)
    y = torch.tanh(x)
    plt.plot(x.numpy(), y.numpy())
    plt.show()


if __name__ == "__main__":
    plot_tanh()
