import random
import torch
from torch.backends import cudnn
import yaml
import argparse
import numpy

def init_seed(seed):
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='configs/exp_efficientnet.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r'))
    device = cfg['device']
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        if(device == 'xpu' and torch.xpu.is_available):
            torch.xpu.manual_seed(seed)
        elif(device == 'cuda' and torch.cuda.is_available()):
            cuda.xpu.manual_seed(seed)