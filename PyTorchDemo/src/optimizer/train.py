import torch.nn as nn
import torch.optim as op

from tqdm import tqdm

from data import get_toy_data
from model import Perceptron


def run():
    # define hyperparameters
    input_dim = 2
    lr = 0.001

    # define model, loss, optimizer
    perceptron = Perceptron(input_dim=input_dim)
    bce_loss = nn.BCELoss()
    optimizer = op.Adam(params=perceptron.parameters(), lr=lr)

    # define training process
    epochs = 100
    num_batches = 50
    batch_size = 200
    for i in tqdm(range(epochs)):
        for j in range(num_batches):
            # get the current batch
            data_x, data_y = get_toy_data(batch_size)

            # clear gradient bookkeeping
            perceptron.zero_grad()

            # pass data forward through computational graph
            pred_y = perceptron(data_x)

            # compare predictions to labels and compute loss
            loss = bce_loss(pred_y, data_y)
            if i == 0:
                print('Initial loss of {}'.format(loss))
            elif i == epochs - 1:
                print('Final loss of {}'.format(loss))

            # push loss back through the network
            loss.backward()

            # tell the optimizer to take a step in the gradient direction for the model parameters
            optimizer.step()


if __name__ == "__main__":
    run()
