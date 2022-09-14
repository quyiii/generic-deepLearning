from .model_wrapper import ModelWrapper

def create_model(cfg):
    return ModelWrapper(cfg)