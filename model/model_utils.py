import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
