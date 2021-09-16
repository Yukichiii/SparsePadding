from typing import Any, Callable, Dict
import numpy as np
import torch
from torch import nn
from torchpack.train import Trainer
from torchpack.utils.typing import Optimizer, Scheduler
from torchsparse.point_tensor import PointTensor

__all__ = ['SemanticKITTITrainer']


class SemanticKITTITrainer(Trainer):
    def __init__(self, model: nn.Module, criterion: Callable,
                 optimizer: Optimizer, scheduler: Scheduler,
                 num_workers: int, seed: int, point_wise: bool, nearest: bool) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_workers = num_workers
        self.seed = seed
        self.epoch_num = 1
        self.point_wise = point_wise
        self.nearest = nearest

    def _before_epoch(self) -> None:
        self.model.train()
        self.dataflow.sampler.set_epoch(self.epoch_num-1)
        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
                self.seed + (self.epoch_num-1) * self.num_workers + worker_id)

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:   
        _inputs = dict()
        for key, value in feed_dict.items():
            if not 'name' in key:
                _inputs[key] = value.cuda()

        inputs = _inputs['lidar']
        if self.point_wise:
            pts = _inputs['targets_mapped']
            z = PointTensor(pts.F, pts.C.double())
            targets = pts.F.long().cuda(non_blocking=True)
            outputs = self.model(inputs, pts=z, nearest=False)
        else:
            targets = feed_dict['targets'].F.long().cuda(non_blocking=True)
            outputs = self.model(inputs, pts=None, nearest=self.nearest)

        if outputs.requires_grad:
            loss = self.criterion(outputs, targets)
            self.summary.add_scalar('loss', loss.item())
            pred = outputs.argmax(1)
            ignore_mask = (targets != 255)
            targets = targets[ignore_mask]
            pred = pred[ignore_mask]
            acc = (pred==targets).cpu().numpy().mean()
            self.summary.add_scalar('acc', acc)
            
            lr = self.scheduler.get_last_lr()[0]
            self.summary.add_scalar('lr', lr)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        else:
            if self.point_wise:
                outputs = outputs.argmax(1)
            else:
                invs = feed_dict['inverse_map']
                all_labels = feed_dict['targets_mapped']
                _outputs = []
                _targets = []
                for idx in range(invs.C[:, -1].max()+1):
                    cur_scene_pts = (inputs.C[:, -1] == idx).cpu().numpy()
                    cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                    cur_label = (all_labels.C[:, -1].int() == idx).cpu().numpy()
                    outputs_mapped = outputs[cur_scene_pts][
                        cur_inv].argmax(1)
                    targets_mapped = all_labels.F[cur_label]
                    _outputs.append(outputs_mapped)
                    _targets.append(targets_mapped)
                outputs = torch.cat(_outputs, 0)
                targets = torch.cat(_targets, 0)
        
        #print("%.3fMB" % (torch.cuda.max_memory_allocated() / 1024 / 1024))
        return {'outputs': outputs, 'targets': targets}

    def _after_epoch(self) -> None:
        self.model.eval()
        

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = dict()
        state_dict['model'] = self.model.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        
    def _load_previous_checkpoint(self, checkpoint_path: str) -> None:
        pass