import torch
import matplotlib.pyplot as plt


def plot_prelu():
    prelu = torch.nn.PReLU(num_parameters=1)
    x = torch.range(-5.0, 5.0, 0.1)
    y = prelu(x)
    plt.plot(x.numpy(), y.detach().numpy())  # need to detach because PReLU has a gradient parameter!
    plt.show()


if __name__ == "__main__":
    plot_prelu()
