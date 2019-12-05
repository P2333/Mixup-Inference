import torch
import numpy as np
from torch.autograd import Variable


def to_Variable(x, cuda=True, dtype=np.float32, requires_grad=False):
    x = torch.from_numpy(x.astype(dtype))
    if cuda:
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)