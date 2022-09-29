from .model_wrapper import ModelWrapper

import torch
import torch.nn as nn

def create_model(cfg):
    return ModelWrapper(cfg)