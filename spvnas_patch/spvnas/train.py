import argparse
import sys

import random
import numpy as np

import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
from torchpack import distributed as dist
from torchpack.callbacks.triggers import PeriodicCallback
from torchpack.callbacks import (InferenceRunner, MaxSaver,
                                 Saver)
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from core import builder
from core.trainers import SemanticKITTITrainer
from core.callbacks import MeanIoU

def count_parameters(model):
  num_params = 0
  for param in model.parameters():
      num_params += param.numel()
  return num_params

def main() -> None:
    dist.init()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)

    if dist.local_rank() == 0:
        logger.info(' '.join([sys.executable] + sys.argv))
        logger.info(f'Experiment started: "{args.run_dir}".' + '\n' + f'{configs}')
    
    # seed
    if ('seed' not in configs.train) or (configs.train.seed is None):
        configs.train.seed = torch.initial_seed() % (2**32 - 1)
        
    seed = configs.train.seed + dist.rank() * configs.workers_per_gpu * configs.num_epochs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset = builder.make_dataset()
    dataflow = dict()
    for split in dataset:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset[split],
            num_replicas=dist.size(),
            rank=dist.rank(),
            shuffle=(split == 'train'))
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=configs.batch_size,
            sampler=sampler,
            num_workers=configs.workers_per_gpu,
            pin_memory=True,
            collate_fn=dataset[split].collate_fn)

    model = builder.make_model()
    if dist.local_rank() == 0:
        logger.info(model)
        logger.info("#parameters number#: %.2fM" % (count_parameters(model) / 1000 / 1000))

    model = torch.nn.parallel.DistributedDataParallel(
        model.cuda(),
        device_ids=[dist.local_rank()],
        find_unused_parameters=True)

    criterion = builder.make_criterion()
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)

    trainer = SemanticKITTITrainer(model=model,
                          criterion=criterion,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          num_workers=configs.workers_per_gpu,
                          seed=seed,
                          point_wise=configs.model.point_wise,
                          nearest=configs.model.nearest)
    trainer.train_with_defaults(
        dataflow['train'],
        num_epochs=configs.num_epochs,
        callbacks=[
            PeriodicCallback(InferenceRunner(
                dataflow[split],
                callbacks=[MeanIoU(
                    name=f'iou/{split}',
                    num_classes=configs.data.num_classes,
                    ignore_label=configs.data.ignore_label
                )]), every_k_epochs=configs.train.eval_epoch)
            for split in ['test']
        ] + [
            PeriodicCallback(MaxSaver('iou/test'), every_k_epochs=configs.train.eval_epoch),
            PeriodicCallback(Saver(max_to_keep=3), every_k_epochs=configs.train.save_epoch),
        ])


if __name__ == '__main__':
    main()