# This file contains experimental modules

import numpy as np
import torch
import torch.nn as nn

from models.common import Conv, DWConv

class CrossConv(nn.Module):
    # Cross convolution downsampling
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        super(CrossConv, self).__init__()
        pass

    def forward(self, x):
        return x

class MixConv2d:
    pass