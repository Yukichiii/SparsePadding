import collections

import numpy as np
from scipy.linalg import expm, norm
from torchsparse.utils import sparse_quantize
from core.datasets.pad_voxel_for_ti import pad_voxel

# Rotation matrix along axis with angle theta
def M(axis, theta):
  return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


class Voxelizer:

  def __init__(self,
               voxel_size=1,
               clip_bound=None,
               use_augmentation=False,
               scale_augmentation_bound=None,
               rotation_augmentation_bound=None,
               translation_augmentation_ratio_bound=None,
               ignore_label=255,
               voxel_pad_stride=[],
               voxel_pad_method='none'):
    """
    Args:
      voxel_size: side length of a voxel
      clip_bound: boundary of the voxelizer. Points outside the bound will be deleted
        expects either None or an array like ((-100, 100), (-100, 100), (-100, 100)).
      scale_augmentation_bound: None or (0.9, 1.1)
      rotation_augmentation_bound: None or ((np.pi / 6, np.pi / 6), None, None) for 3 axis.
        Use random order of x, y, z to prevent bias.
      translation_augmentation_bound: ((-5, 5), (0, 0), (-10, 10))
      ignore_label: label assigned for ignore (not a training label).
    """
    self.voxel_size = voxel_size
    self.clip_bound = clip_bound
    self.ignore_label = ignore_label
    self.voxel_pad_stride = voxel_pad_stride
    self.voxel_pad_method = voxel_pad_method

    # Augmentation
    self.use_augmentation = use_augmentation
    self.scale_augmentation_bound = scale_augmentation_bound
    self.rotation_augmentation_bound = rotation_augmentation_bound
    self.translation_augmentation_ratio_bound = translation_augmentation_ratio_bound

  def get_transformation_matrix(self):
    voxelization_matrix, rotation_matrix = np.eye(4), np.eye(4)
    # Get clip boundary from config or pointcloud.
    # Get inner clip bound to crop from.

    # Transform pointcloud coordinate to voxel coordinate.
    # 1. Random rotation
    rot_mat = np.eye(3)
    if self.use_augmentation and self.rotation_augmentation_bound is not None:
      if isinstance(self.rotation_augmentation_bound, collections.Iterable):
        rot_mats = []
        for axis_ind, rot_bound in enumerate(self.rotation_augmentation_bound):
          theta = 0
          axis = np.zeros(3)
          axis[axis_ind] = 1
          if rot_bound is not None:
            theta = np.random.uniform(*rot_bound)
          rot_mats.append(M(axis, theta))
        # Use random order
        np.random.shuffle(rot_mats)
        rot_mat = rot_mats[0] @ rot_mats[1] @ rot_mats[2]
      else:
        raise ValueError()
    rotation_matrix[:3, :3] = rot_mat
    # 2. Scale and translate to the voxel space.
    scale = 1 / self.voxel_size
    if self.use_augmentation and self.scale_augmentation_bound is not None:
      scale *= np.random.uniform(*self.scale_augmentation_bound)
    np.fill_diagonal(voxelization_matrix[:3, :3], scale)
    # Get final transformation matrix.
    return voxelization_matrix, rotation_matrix

  def clip(self, coords, center=None, trans_aug_ratio=None):
    bound_min = np.min(coords, 0).astype(float)
    bound_max = np.max(coords, 0).astype(float)
    bound_size = bound_max - bound_min
    if center is None:
      center = bound_min + bound_size * 0.5
    if trans_aug_ratio is not None:
      trans = np.multiply(trans_aug_ratio, bound_size)
      center += trans
    lim = self.clip_bound

    if isinstance(self.clip_bound, (int, float)):
      if bound_size.max() < self.clip_bound:
        return None
      else:
        clip_inds = ((coords[:, 0] >= (-lim + center[0])) & \
            (coords[:, 0] < (lim + center[0])) & \
            (coords[:, 1] >= (-lim + center[1])) & \
            (coords[:, 1] < (lim + center[1])) & \
            (coords[:, 2] >= (-lim + center[2])) & \
            (coords[:, 2] < (lim + center[2])))
        return clip_inds

    # Clip points outside the limit
    clip_inds = ((coords[:, 0] >= (lim[0][0] + center[0])) & \
        (coords[:, 0] < (lim[0][1] + center[0])) & \
        (coords[:, 1] >= (lim[1][0] + center[1])) & \
        (coords[:, 1] < (lim[1][1] + center[1])) & \
        (coords[:, 2] >= (lim[2][0] + center[2])) & \
        (coords[:, 2] < (lim[2][1] + center[2])))
    return clip_inds

  def voxelize(self, coords, feats, labels, center=None):
    assert coords.shape[1] == 3 and coords.shape[0] == feats.shape[0] and coords.shape[0]
    if self.clip_bound is not None:
      trans_aug_ratio = np.zeros(3)
      if self.use_augmentation and self.translation_augmentation_ratio_bound is not None:
        for axis_ind, trans_ratio_bound in enumerate(self.translation_augmentation_ratio_bound):
          trans_aug_ratio[axis_ind] = np.random.uniform(*trans_ratio_bound)

      clip_inds = self.clip(coords, center, trans_aug_ratio)
      if clip_inds is not None:
        coords, feats = coords[clip_inds], feats[clip_inds]
        if labels is not None:
          labels = labels[clip_inds]

    # Get rotation and scale
    M_v, M_r = self.get_transformation_matrix()
    # Apply transformations
    rigid_transformation = M_v
    if self.use_augmentation:
      rigid_transformation = M_r @ rigid_transformation

    homo_coords = np.hstack((coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)))

    coords_aug_f = homo_coords @ rigid_transformation.T[:, :3]
    lower_bound = np.min(coords_aug_f, axis=0)
    coords_aug_f -= lower_bound
    coords_aug_round = np.round(coords_aug_f)

    inds, sp_labels, inverse_map = sparse_quantize(coords_aug_round,
                                                feats,
                                                labels,
                                                return_index=True,
                                                return_invs=True)
    sp_coords = coords_aug_round[inds]
    sp_feats = feats[inds]
    sp_labels = labels[inds]

    # add a channel as non_padded flag
    # if len(self.voxel_pad_stride) > 0:
    #   sp_feats = np.pad(sp_feats, ((0, 0), (0, 1)), 'constant', constant_values=1)
    
    init_sp_coords = sp_coords.copy()
    for stride in self.voxel_pad_stride:
      # pad for all initial sp_coords: init_sp_coords
      # do not pad for new paddd voxels
      if self.voxel_pad_method == 'full':
        coords_aug_floor = np.floor(init_sp_coords / stride) * stride
        floor_inds = sparse_quantize(coords_aug_floor, return_index=True)
        sp_coords_floor = coords_aug_floor[floor_inds]
        pad_num = 27
      elif self.voxel_pad_method == 'octree':
        coords_aug_floor = np.floor(init_sp_coords / (stride*2)) * (stride*2)
        floor_inds = sparse_quantize(coords_aug_floor, return_index=True)
        sp_coords_floor = coords_aug_floor[floor_inds]
        pad_num = 8
      elif self.voxel_pad_method == 'trilinear':
        coords_aug_floor = np.floor(coords_aug_f / stride) * stride
        floor_inds = sparse_quantize(coords_aug_floor, return_index=True)
        sp_coords_floor = coords_aug_floor[floor_inds]
        pad_num = 8
      else:
        break
      
      # use current all sp_coords(can be padded) to build flag for padding: sp_coords_at_stride
      if stride > 1:
        sp_coords_at_stride_ = np.floor(sp_coords / stride) * stride
        down_inds = sparse_quantize(sp_coords_at_stride_, return_index=True)
        sp_coords_at_stride = sp_coords_at_stride_[down_inds]
      else:
        sp_coords_at_stride = sp_coords

      padded_sp_coords = pad_voxel(sp_coords_at_stride.astype(np.int32), sp_coords_floor.astype(np.int32), pad_num=pad_num, stride=stride).astype(np.float32)
      padded_num = len(padded_sp_coords)
      sp_feats = np.pad(sp_feats, ((0, padded_num), (0, 0)), 'constant', constant_values=0)
      sp_labels = np.pad(sp_labels, (0, padded_num), 'constant', constant_values=self.ignore_label)
      sp_coords = np.concatenate([sp_coords, padded_sp_coords], axis=0)

    if len(self.voxel_pad_stride) > 0:
      # if reorder the sp_coords, inverse_map should also be mapped use reorder_inverse
      voxel_num_after_padded = len(sp_coords)
      reorder_inds, reorder_inverse = sparse_quantize(sp_coords, return_index=True, return_invs=True)
      assert(voxel_num_after_padded == len(reorder_inds))        
      sp_coords = sp_coords[reorder_inds]
      sp_labels = sp_labels[reorder_inds]
      sp_feats = sp_feats[reorder_inds]
      inverse_map = reorder_inverse[inverse_map]

      assert((sp_coords[inverse_map]==coords_aug_round).all())

    return sp_coords, sp_feats, sp_labels, rigid_transformation.flatten(), inverse_map, coords_aug_f, labels
