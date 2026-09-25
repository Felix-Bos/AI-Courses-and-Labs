from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CIFAR10Data:

    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2470, 0.2435, 0.2616)

    def __init__(
        self,
        data_dir: str | Path = "./data",
        batch_size: int = 128,
        augment: bool = False,
        num_workers: int = 2,
    ):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.augment = augment
        self.num_workers = num_workers

        self.train_dataset = datasets.CIFAR10(
            self.data_dir, train=True, download=True,
            transform=self._get_transforms(augment),
        )
        self.val_dataset = datasets.CIFAR10(
            self.data_dir, train=False, download=True,
            transform=self._get_transforms(augment=False),
        )

    def _get_transforms(self, augment: bool) -> transforms.Compose:
        tfs = []
        if augment:
            tfs += [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        tfs += [
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ]
        return transforms.Compose(tfs)

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    @property
    def train_loader(self) -> DataLoader:
        return self._make_loader(self.train_dataset, shuffle=True)

    @property
    def val_loader(self) -> DataLoader:
        return self._make_loader(self.val_dataset, shuffle=False)