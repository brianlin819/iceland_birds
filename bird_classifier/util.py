import random
import torch
from torch.backends import cudnn
import numpy

def init_seed(seed):

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.xpu.manual_seed(seed)
        cudnn.benchmark = True
        cudnn.deterministic = True