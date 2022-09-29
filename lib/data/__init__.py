import importlib
import pdb
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
            # print("find "+ name)
            dataset = cls
            break
    if dataset is None:
        raise NotImplementedError("no {}.py in lib/data".format(dataset_type + "_dataset"))
    return dataset


def get_dataLoader(cfg):
    dataset = None
    dataset_type = cfg.DATASET.TYPE.lower()
    if dataset_type in ["unaligned", "aligned"]:
        dataset_class = get_dataset_class(cfg)
        dataset = dataset_class(cfg)
    else:
        raise NotImplementedError("dataloader {} is not implemented".format(dataset_type))
    print("create dataset with {} data".format(len(dataset)))
    # 根据源码阅读得知: dataloader返回根据collect_func返回数据
    # collect_func 在batch元素为 tensor 或 int 或 float时转换数据为tensor
    # 并torch.stack(batch, 0) 将单个数据拼接为(batch, data)
    # 在数据为dict 即 {}时 将每个batch元素内部对应的各个key组成的list递归调用collect_func
    # 因此 dict通过dataloader得到 {'A':tensor of batch A / As}
    # 在数据为string时 直接返回[string, string]
    return DataLoader(dataset, 
                      batch_size=cfg.DATALOADER.BATCH_SIZE,
                      shuffle=cfg.DATALOADER.SHUFFLE, 
                      num_workers=cfg.DATALOADER.NUM_WORKERS,
                      drop_last=True)
