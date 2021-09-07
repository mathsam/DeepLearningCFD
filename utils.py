import torch
from torch import nn
from torch.nn import Conv2d

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DownSampler(nn.Module):

    def __init__(self, scaling_factor):
        super().__init__()
        self.conv_filter = Conv2d(1, 1, scaling_factor, stride=scaling_factor, padding=0)
        self.conv_filter.weight.data = torch.ones_like(self.conv_filter.weight.data) / scaling_factor ** 2
        self.conv_filter.bias.data = torch.zeros_like(self.conv_filter.bias.data)
        self.conv_filter = self.conv_filter
    
    def forward(self, field2d):
        if len(field2d.shape) == 3:
            bs, h, w = field2d.shape
        elif len(field2d.shape) == 2:
            h, w = field2d.shape
            bs = 1
        else:
            raise ValueError(f"shape of field2d not supported: {field2d.shape}")
        
        field2d = field2d.reshape((bs, 1, h, w))
        with torch.no_grad():
            output = self.conv_filter(field2d)
        
        bs, _, ho, wo = output.shape
        if bs == 1:
            output = output.reshape((ho, wo))    
        else:
            output = output.reshape((bs, ho, wo))
        return output