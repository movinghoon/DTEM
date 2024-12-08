import os
import torch
import torchvision.datasets as datasets

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms


DATA_DIR = None # example) /path1/path2/ILSVRC2012


def build_train_but_val_loader(data_dir=DATA_DIR,
                                is_train=True,
                                distributed=True,
                                batch_size=128,
                                workers=4,
                                input_size=224,
                                mean=IMAGENET_DEFAULT_MEAN,
                                std=IMAGENET_DEFAULT_STD):
    val_transform = transforms.Compose([
        transforms.Resize(256 * input_size // 224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_dset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transform)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dset) if distributed else None
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        sampler=val_sampler,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    return val_loader

def build_base_dataloader(data_dir=DATA_DIR,
                     is_train=True,
                     distributed=True,
                     batch_size=128,
                     workers=4,
                     input_size=224,
                     mean=IMAGENET_DEFAULT_MEAN,
                     std=IMAGENET_DEFAULT_STD):
    if is_train:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_dset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
        sampler = torch.utils.data.distributed.DistributedSampler(train_dset) if distributed else None
        train_loader = torch.utils.data.DataLoader(train_dset,
                                                sampler=sampler,
                                                batch_size=batch_size,
                                                num_workers=workers,
                                                pin_memory=True,
                                                drop_last=True,)
        return train_loader
    
    val_transform = transforms.Compose([
        transforms.Resize(int((256 / 224) * input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_dset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transform)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dset) if distributed else None
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        sampler=val_sampler,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )
    return val_loader