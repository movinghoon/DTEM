import os
import math
import timm
import wandb
import torch
import random
import argparse
from tqdm import tqdm
from functools import partial
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.datasets as datasets
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms

# timm
from timm.data import create_transform, Mixup
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

# utils
from src.logger import Logger, DummyLogger
from src.utils import accuracy_at_k, dist_init, adjust_learning_rate, get_params_groups, warmup_adjust_learning_rate

# for mae
from model.mae import vit_base_patch16_mae, vit_large_patch16_mae

# dataloader
from data import build_base_dataloader, build_dataloader, build_train_but_val_loader

# ltm
from dtem import patch


MODEL_NAME_DICT = {
    'deit-tiny': 'deit_tiny_patch16_224.fb_in1k',
    'deit-small': 'deit_small_patch16_224.fb_in1k',
    'deit-base': 'deit_base_patch16_224.fb_in1k',
}


# args
parser = argparse.ArgumentParser()
parser.add_argument("--arch", default='deit-small', type=str, choices=list(MODEL_NAME_DICT.keys()))
parser.add_argument('--prop_pool', default=True, type=lambda s: s.lower() in ['true', 't', 'yes', '1'])    # for mae output

# num reduction
parser.add_argument('--reduction', default=16, type=int)
parser.add_argument("--val-reduction", default=16, type=int)

# differentiable Argumenets
parser.add_argument("--k2", default=3, type=int)
parser.add_argument("--tau1", default=0.1, type=float)
parser.add_argument("--tau2", default=0.1, type=float)
parser.add_argument("--feat-dim", default=None, type=int)

# loader
parser.add_argument('--aug', default='deit', choices=['base', 'deit'])
parser.add_argument('--data-dir', required=True, type=str)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--workers', default=4, type=int, help='per gpu # of workers')

# mixup
parser.add_argument('--mixup', type=float, default=0.8, help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
parser.add_argument('--cutmix', type=float, default=1., help='cutmix alpha, cutmix enabled if > 0. default (1.0)')
parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup_prob', type=float, default=1.0, help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup_switch_prob', type=float, default=0.5, help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup_mode', type=str, default='batch', help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

# loss
parser.add_argument('--smoothing', default=0., type=float)

# optimizer
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--min-lr', default=1e-6, type=float)
parser.add_argument('--weight-decay', default=1e-4, type=float)
parser.add_argument('--clip-grad', default=5., type=float)

# epoch
parser.add_argument('--warmup_epochs', default=0, type=int)
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--gradient-accumulation', default=1, type=int)
parser.add_argument("--lr-scheduler", default='step', type=str, choices=['step', 'epoch'])

# wandb
parser.add_argument('--wandb', default=True, type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument("--name", default=None, required=True, type=str)
parser.add_argument("--entity", default=None, required=True, type=str)
parser.add_argument("--project", default=None, required=True, type=str)

# distributed
parser.add_argument('--distributed', default=True, type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument('--fp16', action='store_true')

# other options
parser.add_argument('--seed', default=42, type=int, help='seed')
parser.add_argument('--val', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--no-save', action='store_true')
args = parser.parse_args()

# for gradient accumulation
global counter
counter = 0


@torch.no_grad()
def evaluate(model, loader, epoch, logger):
    acc1, acc5, counts = [], [], []
    pbar = loader
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            pbar = tqdm(loader, total=len(loader), desc="Epoch:{}".format(epoch))
    else:
        pbar = tqdm(loader, total=len(loader), desc="Epoch:{}".format(epoch))

    # set reduction
    if args.distributed:
        model.module.update_r(args.val_reduction)
    else:
        model.update_r(args.val_reduction)
    
    # eval
    model.eval()
    for (images, labels) in pbar:
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # inference
        logits = model(images)
        _acc1, _acc5 = accuracy_at_k(logits, targets=labels, top_k=(1, 5))

        # log
        num = logits.size(0)
        counts.append(num)
        acc1.append(num * _acc1.item())
        acc5.append(num * _acc5.item())
    counts = sum(counts)
    acc1, acc5 = np.sum(acc1) / counts, np.sum(acc5) / counts

    # log
    logger.update({"count": counts,
                   "Val/acc1": acc1,
                   "Val/acc5": acc5})


def train(model, optimizer, loader, epoch, logger, scaler=None, mixup_fn=None):
    # Loss
    CRITERION = SoftTargetCrossEntropy() if args.mixup > 0. else LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    
    # logging
    losses, acc1, acc5, counts = [], [], [], []
    
    # pbar at rank 0
    pbar = loader
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            pbar = tqdm(loader, total=len(loader), desc="Epoch:{}".format(epoch + 1))
    else:
        pbar = tqdm(loader, total=len(loader), desc="Epoch:{}".format(epoch + 1))

    # train
    model.train()
    global counter
    for i, (images, labels) in enumerate(pbar):
        # set reduction
        if args.distributed:
            model.module.update_r(args.reduction)
        else:
            model.update_r(args.reduction)
        
        # to cuda
        images = images.cuda(non_blocking=True) if torch.is_tensor(images) else images[0].cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)
            
        # loss
        _loss, _acc1, _acc5, _n = 0, 0, 0, 0
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=args.fp16):
            logits = model(images)
            _loss += CRITERION(logits, labels)
            if mixup_fn is None:
                a, b = accuracy_at_k(logits, targets=labels, top_k=(1, 5))
                _acc1 += a.item()
                _acc5 += b.item()
            _n += 1
            loss = _loss / args.gradient_accumulation

        # compute the gradients
        if args.fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # learning rate schedule
        if args.lr_scheduler == 'step' and counter % args.gradient_accumulation == 0:
            warmup_adjust_learning_rate(optimizer, i / len(loader) + epoch, args.lr, args.min_lr, args.warmup_epochs, args.epochs)

        # step
        counter += 1
        if counter % args.gradient_accumulation == 0:
            if args.fp16:
                # gradient clipping
                scaler.unscale_(optimizer)
                if args.clip_grad is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)

                # update
                scaler.step(optimizer)
                scaler.update()
            else:
                # gradient clipping
                if args.clip_grad is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
                
                # update
                optimizer.step()
            optimizer.zero_grad()

        # for logging
        num = labels.size(0)
        counts.append(num)
        acc1.append(_acc1 * num / _n)
        acc5.append(_acc5 * num / _n)
        losses.append(num * _loss.item())

    # log
    counts = sum(counts)
    if mixup_fn is None:
        acc1, acc5, losses = np.sum(acc1) / counts, np.sum(acc5) / counts, np.sum(losses) / counts
        print("Loss:{:5.4g}, Acc1:{:5.4g}, Acc5:{:5.4g}".format(losses, acc1, acc5))
        out_dict = {"count": counts,
                    "Train/loss": losses,
                    "Train/acc1": acc1,
                    "Train/acc5": acc5}
    else:
        losses = np.sum(losses) / counts
        print("Loss:{:5.4g}".format(losses))
        out_dict = {"count": counts,
                    "Train/loss": losses,}
    
    logger.update(out_dict)


def main():
    if args.test: # for debugging
        args.epochs = 1
        args.wandb = False
        args.distributed = False

    # distributed handling
    args.local_rank = int(os.environ['LOCAL_RANK']) if args.distributed else 0
    if args.distributed:
        dist_init(args.local_rank)
    
    # set seed
    random.seed(args.seed + args.local_rank)
    torch.manual_seed(args.seed + args.local_rank)
    
    # model
    if 'deit' in args.arch.lower() or 'vit' in args.arch.lower():
        model = timm.create_model(MODEL_NAME_DICT[args.arch], pretrained=True)
        
        # image cfg
        INPUT_SIZE = model.default_cfg["input_size"][1]
        IMG_MEAN = model.default_cfg["mean"]
        IMG_STD = model.default_cfg["std"]
    
    elif 'mae' in args.arch.lower():
        model = MODEL_NAME_DICT[args.arch](pretrained=True) # timm.create_model(model_name, pretrained=True)
        
        # image cfg
        INPUT_SIZE = 224
        IMG_MEAN = IMAGENET_DEFAULT_MEAN
        IMG_STD = IMAGENET_DEFAULT_STD
    
    else:
        raise NotImplementedError
    
    # Patch model
    mae = True if 'mae' in args.arch.lower() else False
    model = patch(model,
                  k2=args.k2,
                  tau1=args.tau1,
                  tau2=args.tau2,
                  feat_dim=args.feat_dim,
                  mae=mae,
                  prop_pool=args.prop_pool)
    model.update_r(args.val_reduction)

    # Freeze params
    for n, p in model.named_parameters():
        if 'metric' not in n.lower():
            p.requires_grad = False
    
    if args.aug == 'base':
        train_loader = build_base_dataloader(data_dir=args.data_dir,
                                                     is_train=True,
                                                     distributed=args.distributed,
                                                     batch_size=args.batch_size,
                                                     workers=args.workers,
                                                     input_size=INPUT_SIZE,
                                                     mean=IMG_MEAN,
                                                     std=IMG_STD)
    elif args.aug == 'deit':
        train_loader = build_dataloader(data_dir=args.data_dir,
                                                is_train=True,
                                                distributed=args.distributed,
                                                batch_size=args.batch_size,
                                                workers=args.workers,
                                                input_size=INPUT_SIZE,
                                                mean=IMG_MEAN,
                                                std=IMG_STD)
    else:
        raise NotImplementedError

    val_loader = build_base_dataloader(data_dir=args.data_dir,
                                       is_train=False,
                                       distributed=args.distributed,
                                       batch_size=args.batch_size,
                                       workers=args.workers,
                                       input_size=INPUT_SIZE,
                                       mean=IMG_MEAN,
                                       std=IMG_STD)
    
    # DDP
    model = model.cuda()
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        
    # optimizer & scheduler
    _model = model.module if args.distributed else model
    optimizer = torch.optim.AdamW(get_params_groups(_model), lr=args.lr, weight_decay=args.weight_decay)
    
    # fp16
    scaler = None
    if args.fp16:
        scaler = GradScaler()

    # logger
    wandb_logger = None
    if args.local_rank == 0 and args.wandb:
        wandb_logger = wandb.init(entity=args.entity,
                                  project=args.project,
                                  name=args.name,
                                  config=args,
                                  reinit=False)
    logger = Logger(wandb_logger=wandb_logger) if args.wandb else DummyLogger()

    # mixup
    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0 or args.cutmix_minmax is not None:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=len(train_loader.dataset.classes))
        
    # train
    best_acc = 0
    for epoch in range(args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        # learning rate schedule
        if args.lr_scheduler == 'epoch':
            adjust_learning_rate(optimizer, args.lr, args.min_lr, epoch=epoch, epochs=args.epochs)

        # train
        train(model, optimizer, train_loader, epoch, logger, scaler, mixup_fn)
        
        # eval
        evaluate(model, val_loader, epoch + 1, logger)
        
        # log
        metric = logger.log({"epoch": epoch + 1})    

        # print
        acc1 = metric['Val/acc1'].item()
        acc5 = metric['Val/acc5'].item()
        print("Val Acc1:{:5.4g}, Acc5:{:5.4g}".format(acc1, acc5))
        
        # save
        acc = metric['Val/acc1'].item() if args.wandb else 0    # no wandb, no saving
        if args.local_rank == 0 and acc > best_acc and args.no_save is not None:
            best_acc = acc
            save_dict = {
                    "epoch": epoch + 1,
                    "best_acc": best_acc,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
            }
            name = './' + args.name + '.pth.tar'
            torch.save(save_dict, name)

    # end
    if args.local_rank == 0 and args.wandb:
        logger.wandb_logger.finish()

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()