import logging
import os
import sys
from pathlib import Path

import numpy as np
from scipy import spatial

from core.datasets.mink_dataset import VoxelizationDataset, DatasetPhase, str2datasetphase_type
import core.datasets.mink_transforms as t


__all__ = ['Scannet']

CLASS_LABELS = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')
VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
SCANNET_COLOR_MAP = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    14: (247., 182., 210.),
    15: (66., 188., 102.),
    16: (219., 219., 141.),
    17: (140., 57., 197.),
    18: (202., 185., 52.),
    19: (51., 176., 203.),
    20: (200., 54., 131.),
    21: (92., 193., 61.),
    22: (78., 71., 183.),
    23: (172., 114., 82.),
    24: (255., 127., 14.),
    25: (91., 163., 138.),
    26: (153., 98., 156.),
    27: (140., 153., 101.),
    28: (158., 218., 229.),
    29: (100., 125., 154.),
    30: (178., 127., 135.),
    32: (146., 111., 194.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    35: (96., 207., 209.),
    36: (227., 119., 194.),
    37: (213., 92., 176.),
    38: (94., 106., 211.),
    39: (82., 84., 163.),
    40: (100., 85., 144.),
}

def read_txt(path):
  """Read txt file into lines.
  """
  with open(path) as f:
    lines = f.readlines()
  lines = [x.strip() for x in lines]
  return lines


class Scannet(dict):
    def __init__(self, root, voxel_size, num_points, **kwargs):
        submit_to_server = kwargs.get('submit', False)
        sample_stride = kwargs.get('sample_stride', 1)
        voxel_pad_stride = kwargs.get('voxel_pad_stride', [])
        voxel_pad_method = kwargs.get('voxel_pad_method', 'none')
        if submit_to_server:
            super(Scannet, self).__init__({
                'train':
                ScannetVoxelizationDataset(root,
                                      voxel_size,
                                      num_points,
                                      split='trainval',
                                      augment_data=True,
                                      voxel_pad_stride=voxel_pad_stride,
                                      voxel_pad_method=voxel_pad_method),
                'test':
                ScannetVoxelizationDataset(root,
                                      voxel_size,
                                      num_points,
                                      split='test',
                                      augment_data=False,
                                      voxel_pad_stride=voxel_pad_stride,
                                      voxel_pad_method=voxel_pad_method)
            })
        else:
            super(Scannet, self).__init__({
                'train':
                ScannetVoxelizationDataset(root,
                                      voxel_size,
                                      num_points,
                                      split='train',
                                      augment_data=True,
                                      voxel_pad_stride=voxel_pad_stride,
                                      voxel_pad_method=voxel_pad_method),
                'test':
                ScannetVoxelizationDataset(root,
                                      voxel_size,
                                      num_points,
                                      split='val',
                                      augment_data=False,
                                      voxel_pad_stride=voxel_pad_stride,
                                      voxel_pad_method=voxel_pad_method)  #,
            })




class ScannetVoxelizationDataset(VoxelizationDataset):

  # Voxelization arguments
  CLIP_BOUND = None
  TEST_CLIP_BOUND = None

  # Augmentation arguments
  ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                        np.pi))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
  ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

  ROTATION_AXIS = 'z'
  LOCFEAT_IDX = 2
  NUM_LABELS = 41  # Will be converted to 20 as defined in IGNORE_LABELS.
  IGNORE_LABELS = tuple(set(range(41)) - set(VALID_CLASS_IDS))
  IS_FULL_POINTCLOUD_EVAL = True

  # If trainval.txt does not exist, copy train.txt and add contents from val.txt
  DATA_PATH_FILE = {
      DatasetPhase.Train: 'scannetv2_train.txt',
      DatasetPhase.Val: 'scannetv2_val.txt',
      DatasetPhase.TrainVal: 'scannetv2_trainval.txt',
      DatasetPhase.Test: 'scannetv2_test.txt'
  }

  def __init__(self,
               root_path,
               voxel_size,
               num_points,
               split='train',
               augment_data=False,
               voxel_pad_stride=[],
               voxel_pad_method='none'):
    if isinstance(split, str):
      phase = str2datasetphase_type(split)
    # Use cropped rooms for train/val
    data_root = root_path
    self.VOXEL_SIZE = voxel_size
    if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
      self.CLIP_BOUND = self.TEST_CLIP_BOUND
    data_paths = read_txt(os.path.join(data_root, self.DATA_PATH_FILE[phase]))
    logging.info('Loading {}: {}'.format(self.__class__.__name__, self.DATA_PATH_FILE[phase]))

    if augment_data:
      prevoxel_transform_train = [t.ElasticDistortion(self.ELASTIC_DISTORT_PARAMS)]
      prevoxel_transforms = t.Compose(prevoxel_transform_train)
    else:
      prevoxel_transforms = None

    # input transform cannot keep consistency of sp_coords and pc_coords
    if augment_data:
      input_transforms = [
          #t.RandomDropout(0.2),
          #t.RandomHorizontalFlip(self.ROTATION_AXIS, False),
          t.ChromaticAutoContrast(),
          t.ChromaticTranslation(0.10),
          t.ChromaticJitter(0.05),
          # t.HueSaturationTranslation(config.data_aug_hue_max, config.data_aug_saturation_max),
      ]
      input_transforms = t.Compose(input_transforms)
    else:
      input_transforms = None

    super().__init__(
        data_paths,
        data_root=data_root,
        prevoxel_transform=prevoxel_transforms,
        input_transform=input_transforms,
        target_transform=None,
        ignore_label=255,
        return_transformation=False,
        augment_data=augment_data,
        voxel_pad_stride=voxel_pad_stride,
        voxel_pad_method=voxel_pad_method)


