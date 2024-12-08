import os
import torch
import torchvision.datasets as datasets

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from timm.data import create_transform


def build_dataloader(data_dir=None,
                     is_train=True,
                     distributed=True,
                     batch_size=128,
                     workers=4,
                     input_size=224,
                     mean=IMAGENET_DEFAULT_MEAN,
                     std=IMAGENET_DEFAULT_STD):
    if is_train:
        train_transform = create_transform(
                input_size=input_size,
                is_training=True,
                color_jitter=0.4,
                auto_augment='rand-m9-mstd0.5-inc1',
                interpolation='bicubic',
                re_prob=0.25,
                re_mode='pixel',
                re_count=1,
                mean=mean,
                std=std,
        )
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
    )
    return val_loader