from . import Models
from . import shortcuts
from . import dataset
from . import loss_function

from .shortcuts import *
import torch
import numpy as np
import os

torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(12345)

os.environ["TORCH_HOME"] = "/home/LargeData/torch_home"
