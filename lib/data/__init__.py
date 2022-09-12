import importlib
from torch.utils.data import DataLoader
from data.base_dataset import BaseDataset

def get_dataset_by_name(dataset_type, dataset_name):
    dataset = None
    file_name = "data." + dataset_type + "_dataset"
    datasetlib = importlib.import_module(file_name)

    target_dataset_type = dataset_type.replace('_', '').lower() + "dataset"
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_type and issubclass(cls, BaseDataset):
            dataset = cls
            break
    if dataset is None:
        raise NotImplementError("no {}.py in lib/data".format(target_dataset_type))
    return dataset


def get_dataLoader(cfg):
