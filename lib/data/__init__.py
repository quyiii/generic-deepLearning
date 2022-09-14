import importlib
from torch.utils.data import DataLoader
from .base_dataset import BaseDataset

def get_dataset_class(dataset_type):
    dataset = None
    file_name = "lib.data." + dataset_type + "_dataset"
    datasetlib = importlib.import_module(file_name)

    target_dataset_type = dataset_type.replace('_', '').lower() + "dataset"
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_type and issubclass(cls, BaseDataset):
            dataset = cls
            break
    if dataset is None:
        raise NotImplementError("no {}.py in lib/data".format(dataset_type + "_dataset"))
    return dataset


def get_dataLoader(cfg):
