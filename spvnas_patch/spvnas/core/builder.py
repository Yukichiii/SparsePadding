from typing import Callable

import torch
import torch.optim
from torch import nn
import torchpack.distributed as dist
from torchpack.utils.config import configs
from torchpack.utils.typing import Dataset, Optimizer, Scheduler

__all__ = [
    'make_dataset', 'make_model', 'make_criterion', 'make_optimizer',
    'make_scheduler'
]


def make_dataset() -> Dataset:
    voxel_pad_stride = [int(i) for i in configs.dataset.voxel_pad_stride.split(',') if i!='']
    voxel_pad_stride = [i for i in voxel_pad_stride if i>0]
    voxel_pad_stride.sort()
    if configs.dataset.name == 'semantic_kitti':
        from core.datasets import SemanticKITTI as SegDataset
    elif configs.dataset.name == 'scannet':
        from core.datasets import Scannet as SegDataset
    else:
        raise NotImplementedError(configs.dataset.name)
    dataset = SegDataset(
                submit=configs.dataset.submit,
                root=configs.dataset.root,
                num_points=configs.dataset.num_points,
                voxel_size=configs.dataset.voxel_size,
                voxel_pad_stride=voxel_pad_stride,
                voxel_pad_method=configs.dataset.voxel_pad_method)
    return dataset


def make_model() -> nn.Module:
    if 'cr' in configs.model:
        cr = configs.model.cr
    else:
        cr = 1.0
    if configs.model.name == 'spvcnn':
        from core.models.semantic_kitti import SPVCNN
        model = SPVCNN(
            num_classes=configs.data.num_classes,
            cr=cr,
            pres=configs.dataset.voxel_size,
            vres=configs.dataset.voxel_size
        )
        return model
    
    if configs.model.name == 'minkunet':
        from core.models.semantic_kitti import MinkUNet as MINK_Model
    elif configs.model.name == 'minkunet_low_2':
        from core.models.semantic_kitti import MinkUNet_Low_2 as MINK_Model
    elif configs.model.name == 'minkunet_scannet':
        from core.models.scannet import MinkUNet as MINK_Model
    elif configs.model.name == 'minkunet_low_2_scannet':
        from core.models.scannet import MinkUNet_Low_2 as MINK_Model
    else:
        raise NotImplementedError(configs.model.name)
    model = MINK_Model(
        num_classes=configs.data.num_classes,
        in_channels=configs.data.in_channels,
        cr=cr
    )
    return model


def make_criterion() -> Callable:
    if configs.criterion.name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=configs.criterion.ignore_index)
    elif configs.criterion.name == 'lovasz':
        from core.criterions import MixLovaszCrossEntropy
        criterion = MixLovaszCrossEntropy(
            ignore_index=configs.criterion.ignore_index
        )
    else:
        raise NotImplementedError(configs.criterion.name)
    return criterion


def make_optimizer(model: nn.Module) -> Optimizer:
    if configs.optimizer.name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=configs.optimizer.lr,
            momentum=configs.optimizer.momentum,
            weight_decay=configs.optimizer.weight_decay,
            nesterov=configs.optimizer.nesterov)
    elif configs.optimizer.name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay)
    elif configs.optimizer.name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay)
    else:
        raise NotImplementedError(configs.optimizer.name)
    return optimizer


def make_scheduler(optimizer: Optimizer) -> Scheduler:
    batch_size = configs.batch_size * dist.size()
    iter_per_epoch = (configs.data.training_size + batch_size - 1) // batch_size
    if configs.scheduler.name == 'none':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1)
    elif configs.scheduler.name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=configs.num_epochs)
    elif configs.scheduler.name == 'cosine_warmup':
        from core.schedulers import cosine_schedule_with_warmup
        from functools import partial
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=partial(
                cosine_schedule_with_warmup,
                num_epochs=configs.num_epochs,
                batch_size=configs.batch_size,
                dataset_size=configs.data.training_size
            )
        )
    elif configs.scheduler.name == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[iter_per_epoch * e for e in [configs.num_epochs // 2, configs.num_epochs //4 * 3]],
            gamma=0.1
        )
    elif configs.scheduler.name == 'poly':
        from core.schedulers import PolyLR
        scheduler = PolyLR(optimizer, max_iter=iter_per_epoch * configs.num_epochs, power=0.9, last_step=-1)
    else:
        raise NotImplementedError(configs.scheduler.name)
    return scheduler

