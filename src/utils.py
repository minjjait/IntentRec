# src/utils.py
from __future__ import annotations
import random, numpy as np, torch
import torch.backends.cudnn as cudnn

def set_seed(seed: int = 20250908):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def glorot(t):
    if t is not None:
        torch.nn.init.xavier_uniform_(t)
