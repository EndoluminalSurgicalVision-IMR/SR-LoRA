from .medical_datasets import Chest19, Colon, Endoscopy
# from .natural_datasets import CIFAR100_Fewshot
from .natural_datasets import VTAB1kCIFAR100, VTAB1k
from mmcls.datasets import CIFAR100

__all__ = ['Chest19', 'Endoscopy', 'Colon', 'CIFAR100', 'CIFAR100_Fewshot','VTAB1kCIFAR100', 'VTAB1k']
