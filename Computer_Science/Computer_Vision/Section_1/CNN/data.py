import torch 

from torchvision import datasets, transforms


class CIFAR10Data:
    
    
    def __init__(self, 
                 data_dir: str = "./data",
                 batch_size :int = 128,
                 data_augmentation: bool = False,
                 num_workers: int = 2):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data_augmentation = data_augmentation
        self.num_workers = num_workers
        
        self.train_dataset = datasets.CIFAR10(
            self.data_dir, train=True, download=True,
            transform=self._get_transforms(data_augmentation),
        )
        
        self.val_dataset = datasets.CIFAR10(
            self.data_dir, train=False, download=True,
            transform=self._get_transforms(data_augmentation=False),
        )
        
    def _get_transforms(self, data_augmentation: bool) -> transforms.Compose:
        pass