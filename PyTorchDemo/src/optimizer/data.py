import numpy as np
import torch

LEFT_CENTER = (3, 3)
RIGHT_CENTER = (3, -2)


def get_toy_data(batch_size, left_center=LEFT_CENTER, right_center=RIGHT_CENTER):
    x_data = []
    y_targets = np.zeros(batch_size)
    for batch_i in range(batch_size):
        if np.random.random() > 0.5:
            x_data.append(np.random.normal(loc=left_center))
        else:
            x_data.append(np.random.normal(loc=right_center))
            y_targets[batch_i] = 1
    return torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_targets, dtype=torch.float32)


if __name__ =="__main__":
    x, y = get_toy_data(8, LEFT_CENTER, RIGHT_CENTER)
    print(x.shape)
    print(x)
    print(y.shape)
    print(y)
