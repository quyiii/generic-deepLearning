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


def get_dataLoader(cfg, split='train'):
    dataset = None
    dataset_type = cfg.DATASET.TYPE.lower()
    if dataset_type in ["unaligned", "aligned"]:
        dataset_class = get_dataset_class(cfg)
        dataset = dataset_class(cfg)
    elif dataset_type in ["caption"]:
        dataset_class = get_dataset_class(cfg)
        dataset = dataset_class(cfg, split)
    else:
        raise NotImplementedError("dataloader {} is not implemented".format(dataset_type))
    print("create dataset with {} data".format(len(dataset)))
    # 根据源码阅读得知: dataloader根据collect_fun()对batch数据进行处理并返回数据
    # collect_fun(batch)
    # 在batch里的数据为string时 直接返回[string * batch_size]
    # 在batch里的数据为 tensor 或 int 或 float 时转换数据为tensor
    # 并torch.stack(batch, 0) (新添加一个维度并进行拼接)将单个数据拼接为(batch_size, data)
    # 在batch里的数据为dict 即 {} 时 
    # 将batch元素内部对应的各个key组成的各自的list并各自递归调用collect_func 返回{key:调用结果....}
    # 因此 dict通过dataloader得到 {'A':batch A (A is convert to its mapping type)...}
    return DataLoader(dataset, 
                      batch_size=cfg.DATALOADER.BATCH_SIZE,
                      shuffle=cfg.DATALOADER.SHUFFLE, 
                      num_workers=cfg.DATALOADER.NUM_WORKERS,
                      drop_last=True)
