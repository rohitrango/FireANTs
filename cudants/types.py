''' Collection of all types here '''
from typing import Union, List, TypeVar
import torch
T = TypeVar("T")

devicetype = Union[str, torch.device]
ItemOrList = Union[T, List[T]]
