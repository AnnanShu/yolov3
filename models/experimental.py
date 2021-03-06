# This file contains experimental modules

import numpy as np
import torch
import torch.nn as nn

from models.common import Conv, DWConv

class CrossConv(nn.Module):
    # Cross convolution downsampling
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        super(CrossConv, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1))
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Sum(nn.Module):
    # weight sum of 2 or more layers
    def __init__(self, n, weight=False):
        super(Sum, self).__init__()
        self.weight = weight  # apply weight boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1., n) / 2, requires_grad=True)  # layer weight



class MixConv2d:
    pass