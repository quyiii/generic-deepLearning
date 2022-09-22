import importlib
from torch.utils.data import DataLoader
from .base_dataset import BaseDataset

def get_dataset_class(cfg):
    dataset = None
    input_type = cfg.INPUT.TYPE.lower()
    dataset_type = cfg.DATASET.TYPE.lower()
    file_name = "lib.data." + input_type + "_datasets." + dataset_type + "_dataset"
    datasetlib = importlib.import_module(file_name)

    target_dataset_type = dataset_type.replace('_', '').lower() + "dataset"
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_type and issubclass(cls, BaseDataset):
            dataset = cls
            break
    if dataset is None:
        raise NotImplementedError("no {}.py in lib/data".format(dataset_type + "_dataset"))
    return dataset


def get_dataLoader(cfg):
    dataset = None
    dataset_type = cfg.DATASET.TYPE.lower()
    if dataset_type in ["unaligned"]:
        dataset = get_dataset_class(cfg)(cfg)
    else:
        raise NotImplementedError("dataloader {} is not implemented".format(dataset_type))
    return DataLoader(dataset, 
                      batch_size=cfg.DATALOADER.BATCH_SIZE,
                      shuffle=cfg.DATALOADER.SHUFFLE, 
                      num_workers=cfg.DATALOADER.NUM_WORKERS,
                      drop_last=True)
