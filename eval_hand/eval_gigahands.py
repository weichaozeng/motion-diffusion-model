import torch
from tqdm import tqdm
import numpy as np


def evaluate(data, model, diffusion, device=None):
    model.eval()
    