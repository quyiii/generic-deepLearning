from torch.utils.data import Dataset
from abc import ABC, abstractmethod

class BaseDataset(Dataset, ABC):
    """
    This is an abstract base class for all datasets

    To create a subclass, you must implement these following functions
    __int__
    __len__
    __getitem__
    """

    def __init__(self, args):
        self.args = args

