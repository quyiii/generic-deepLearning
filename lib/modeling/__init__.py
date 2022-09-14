from .model_wrapper import ModelWrapper
from .lr_scheduler

def create_model(cfg):
    return ModelWrapper(cfg)