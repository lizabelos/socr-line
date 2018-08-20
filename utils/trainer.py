import torch

class MovingAverage:
    def __init__(self, moving_average_size=256):
        self.count = 0
        self.i = 0
        self.moving_average_size = moving_average_size
        self.values = self.moving_average_size * [0]

    def reset(self):
        self.count = 0
        self.i = 0

    def moving_average(self):
        if self.count == 0:
            return 0
        return sum(self.values) / self.count

    def addn(self, value):
        self.values[self.i] = value
        self.count = min(self.count + 1, self.moving_average_size)
        self.i = (self.i + 1) % self.moving_average_size


class CPUParallel(torch.nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *input):
        return self.module(*input)
