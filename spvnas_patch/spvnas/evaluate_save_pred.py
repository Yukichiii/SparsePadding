import argparse
import sys

from tqdm import tqdm

import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
from torchpack import distributed as dist
from torchpack.callbacks import (InferenceRunner, MaxSaver,
                                 Saver, SaverRestore, Callbacks)
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from core import builder
from core.trainers import SemanticKITTITrainer
from core.callbacks import MeanIoU
import os
from model_zoo import spvnas_specialized, minkunet, spvcnn
import numpy as np 

def main() -> None:
    dist.init()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    parser.add_argument('--ckpt_dir', help='ckpt directory')
    parser.add_argument('--name', type=str, help='model name')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)

    logger.info(' '.join([sys.executable] + sys.argv))
    logger.info(f'Experiment started: "{args.run_dir}".' + '\n' + f'{configs}')

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
            batch_size=configs.batch_size if split == 'train' else 1,
            sampler=sampler,
            num_workers=configs.workers_per_gpu,
            pin_memory=True,
            collate_fn=dataset[split].collate_fn)

    
    # if 'spvnas' in args.name.lower():
    #     model = spvnas_specialized(args.name)
    # elif 'spvcnn' in args.name.lower():
    #     model = spvcnn(args.name)
    # elif 'mink' in args.name.lower():
    #     model = minkunet(args.name)
    # else:
    #     raise NotImplementedError
    
    model = builder.make_model()

    model = torch.nn.parallel.DistributedDataParallel(
        model.cuda(),
        device_ids=[dist.local_rank()],
        find_unused_parameters=True)

    init = torch.load(
            args.ckpt_dir + "/checkpoints/max-iou-test.pt",
            map_location='cuda:%d'%dist.local_rank() if torch.cuda.is_available() else 'cpu'
        )['model']
    model.load_state_dict(init)

    print("load success!")
    model.eval()

    criterion = builder.make_criterion()
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)
    meter = MeanIoU(configs.data.num_classes, 
                    configs.data.ignore_label)

    trainer = SemanticKITTITrainer(model=model,
                          criterion=criterion,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          num_workers=configs.workers_per_gpu,
                          seed=configs.train.seed,
                          point_wise=configs.model.point_wise,
                          nearest=configs.model.nearest
                          )
    callbacks=Callbacks([
                          SaverRestore(),
                          MeanIoU(
                              configs.data.num_classes, 
                              configs.data.ignore_label
                          )
                      ])
    callbacks._set_trainer(trainer)
    trainer.callbacks = callbacks
    trainer.dataflow = dataflow['test']
    
    
    trainer.before_train()
    trainer.before_epoch()
    
    # important
    model.eval()
    predict_path = args.ckpt_dir + "/predict/"
    if not os.path.exists(predict_path):
        os.mkdir(predict_path)
    print("start eval:")
    for feed_dict in tqdm(dataflow['test'], desc='eval'):
        _inputs = dict()
        for key, value in feed_dict.items():
            if not 'name' in key:
                _inputs[key] = value.cuda()

        sequence = feed_dict['file_name'][0].split('/')[-3]
        frame = feed_dict['file_name'][0].split('/')[-1].split('.')[0]

        inputs = _inputs['lidar']
        targets = feed_dict['targets'].F.long().cuda(non_blocking=True)
        outputs = model(inputs, pts=None, nearest=False)
       
        invs = feed_dict['inverse_map']
        all_labels = feed_dict['targets_mapped']
        _outputs = []
        _targets = []
        for idx in range(invs.C[:, -1].max()+1):
            cur_scene_pts = (inputs.C[:, -1] == idx).cpu().numpy()
            cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
            cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
            outputs_mapped = outputs[cur_scene_pts][
                cur_inv].argmax(1)
            targets_mapped = all_labels.F[cur_label]
            _outputs.append(outputs_mapped)
            _targets.append(targets_mapped)
        outputs = torch.cat(_outputs, 0)
        targets = torch.cat(_targets, 0)

        #out_fn = predict_path + "%s_%s.npy" % (sequence, frame)
        #np.save(out_fn, outputs.cpu().numpy().astype(np.int32))
        out_fn = predict_path + "%s_%s.txt" % (sequence, frame)
        np.savetxt(out_fn, outputs.cpu().numpy(), fmt="%d")

        output_dict = {
            'outputs': outputs,
            'targets': targets
        }
        trainer.after_step(output_dict)
    
    trainer.after_epoch()


if __name__ == '__main__':
    main()