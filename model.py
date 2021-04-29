import torch
import torch.nn as nn


class neuralNet(nn.Module):
    def __init__(self, inputsize, hiddensize, num_classes):
        super(neuralNet, self).__init__()
        self.t1 = nn.Linear(inputsize, hiddensize)
        self.t2= nn.Linear(hiddensize, hiddensize)
        self.t3 = nn.Linear(hiddensize, hiddensize)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.t1(x)
        out = self.relu(out)
        out = self.t2(out)
        out = self.relu(out)
        out = self.t3(out)

        return out