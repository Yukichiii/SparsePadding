import json
import os
import os.path
import random
import sys
from collections import Sequence

import h5py
import numpy as np
import scipy
import scipy.interpolate
import scipy.ndimage
import torch
from numba import jit


from core.datasets.pad_voxel_for_ti import pad_voxel
from torchsparse import SparseTensor, PointTensor
from torchsparse.utils import sparse_collate_fn, sparse_quantize

__all__ = ['SemanticKITTI']

label_name_mapping = {
    0: "unlabeled",
    1: "outlier",
    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    16: "on-rails",
    18: "truck",
    20: "other-vehicle",
    30: "person",
    31: "bicyclist",
    32: "motorcyclist",
    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other-ground",
    50: "building",
    51: "fence",
    52: "other-structure",
    60: "lane-marking",
    70: "vegetation",
    71: "trunk",
    72: "terrain",
    80: "pole",
    81: "traffic-sign",
    99: "other-object",
    252: "moving-car",
    253: "moving-bicyclist",
    254: "moving-person",
    255: "moving-motorcyclist",
    256: "moving-on-rails",
    257: "moving-bus",
    258: "moving-truck",
    259: "moving-other-vehicle"
}

kept_labels = [
    'road', 'sidewalk', 'parking', 'other-ground', 'building', 'car', 'truck',
    'bicycle', 'motorcycle', 'other-vehicle', 'vegetation', 'trunk', 'terrain',
    'person', 'bicyclist', 'motorcyclist', 'fence', 'pole', 'traffic-sign'
]


class SemanticKITTI(dict):
    def __init__(self, root, voxel_size, num_points, **kwargs):
        submit_to_server = kwargs.get('submit', False)
        sample_stride = kwargs.get('sample_stride', 1)
        google_mode = kwargs.get('google_mode', False)
        voxel_pad_stride = kwargs.get('voxel_pad_stride', [])
        voxel_pad_method = kwargs.get('voxel_pad_method', 'none')
        print("========")
        print("submit:", submit_to_server)
        print("========")
        if submit_to_server:
            super(SemanticKITTI, self).__init__({
                'train':
                SemanticKITTIInternal(root,
                                      voxel_size,
                                      num_points,
                                      sample_stride=1,
                                      split='train',
                                      submit=True,
                                      voxel_pad_stride=voxel_pad_stride),
                'test':
                SemanticKITTIInternal(root,
                                      voxel_size,
                                      num_points,
                                      sample_stride=1,
                                      split='test',
                                      voxel_pad_stride=voxel_pad_stride)
            })
        else:
            super(SemanticKITTI, self).__init__({
                'train':
                SemanticKITTIInternal(root,
                                      voxel_size,
                                      num_points,
                                      sample_stride=1,
                                      split='train',
                                      google_mode=google_mode,
                                      voxel_pad_stride=voxel_pad_stride),
                'test':
                SemanticKITTIInternal(root,
                                      voxel_size,
                                      num_points,
                                      sample_stride=sample_stride,
                                      split='val',
                                      voxel_pad_stride=voxel_pad_stride)  #,
                #'real_test': SemanticKITTIInternal(root, voxel_size, num_points, split='test')
            })


class SemanticKITTIInternal:
    def __init__(self,
                 root,
                 voxel_size,
                 num_points,
                 split,
                 sample_stride=1,
                 submit=False,
                 google_mode=True,
                 voxel_pad_stride=[]):
        if submit:
            trainval = True
        else:
            trainval = False
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.sample_stride = sample_stride
        self.google_mode = google_mode
        self.voxel_pad_stride = voxel_pad_stride
        self.seqs = []
        if split == 'train':
            self.seqs = [
                '00', '01', '02', '03', '04', '05', '06', '07', '09', '10'
            ]
            if self.google_mode or trainval:
                self.seqs.append('08')
            #if trainval is True:
            #    self.seqs.append('08')
        elif self.split == 'val':
            self.seqs = ['08']
        elif self.split == 'test':
            self.seqs = [
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                '21'
            ]

        self.files = []
        for seq in self.seqs:
            seq_files = sorted(
                os.listdir(os.path.join(self.root, seq, 'velodyne')))
            seq_files = [
                os.path.join(self.root, seq, 'velodyne', x) for x in seq_files
            ]
            self.files.extend(seq_files)

        #self.files = self.files[:40]
        if self.sample_stride > 1:
            self.files = self.files[::self.sample_stride]

        reverse_label_name_mapping = {}
        self.label_map = np.zeros(260)
        cnt = 0
        for label_id in label_name_mapping:
            if label_id > 250:
                if label_name_mapping[label_id].replace('moving-',
                                                        '') in kept_labels:
                    self.label_map[label_id] = reverse_label_name_mapping[
                        label_name_mapping[label_id].replace('moving-', '')]
                else:
                    self.label_map[label_id] = 255
            elif label_id == 0:
                self.label_map[label_id] = 255
            else:
                if label_name_mapping[label_id] in kept_labels:
                    self.label_map[label_id] = cnt
                    reverse_label_name_mapping[
                        label_name_mapping[label_id]] = cnt
                    cnt += 1
                else:
                    self.label_map[label_id] = 255

        self.reverse_label_name_mapping = reverse_label_name_mapping
        self.num_classes = cnt
        self.angle = 0.0
        # print('There are %d classes.' % (cnt))

    def set_angle(self, angle):
        self.angle = angle

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with open(self.files[index], 'rb') as b:
            block_ = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        block = np.zeros_like(block_)

        if 'train' in self.split:
            theta = np.random.uniform(0, 2 * np.pi)
            scale_factor = np.random.uniform(0.95, 1.05)
            rot_mat = np.array([[np.cos(theta),
                                 np.sin(theta), 0],
                                [-np.sin(theta),
                                 np.cos(theta), 0], [0, 0, 1]])

            block[:, :3] = np.dot(block_[:, :3], rot_mat) * scale_factor
            #block[:, 3:] = block_[:, 3:] + np.random.randn(3) * 0.1
        else:
            theta = self.angle
            transform_mat = np.array([[np.cos(theta),
                                       np.sin(theta), 0],
                                      [-np.sin(theta),
                                       np.cos(theta), 0], [0, 0, 1]])
            block[...] = block_[...]
            block[:, :3] = np.dot(block[:, :3], transform_mat)

        block[:, 3] = block_[:, 3]
        orig_pc_ = block[:, :3] / self.voxel_size
        orig_pc_ -= orig_pc_.min(0, keepdims=1)

        pc_ = np.round(orig_pc_)
        pc_floor_ = np.floor(orig_pc_)
        #inds = self.inds[index]

        label_file = self.files[index].replace('velodyne', 'labels').replace(
            '.bin', '.label')
        if os.path.exists(label_file):
            with open(label_file, 'rb') as a:
                all_labels = np.fromfile(a, dtype=np.int32).reshape(-1)
        else:
            all_labels = np.zeros((pc_.shape[0])).astype(np.int32)

        labels_ = self.label_map[all_labels & 0xFFFF].astype(
            np.int64)  # semantic labels
        inst_labels_ = (all_labels >> 16).astype(np.int64)  # instance labels

        feat_ = block
        if 'train' in self.split:
            if len(pc_) > self.num_points:
                selected = np.random.choice(np.arange(len(pc_)), self.num_points, replace=False)
                orig_pc_ = orig_pc_[selected]
                pc_ = pc_[selected]
                pc_floor_ = pc_floor_[selected]
                feat_ = feat_[selected]
                labels_ = labels_[selected]
                inst_labels_ = inst_labels_[selected]
        inds, labels, inverse_map = sparse_quantize(pc_,
                                                    feat_,
                                                    labels_,
                                                    return_index=True,
                                                    return_invs=True)

        pc = pc_[inds]
        feat = feat_[inds]
        labels = labels_[inds]
        
        for stride in self.voxel_pad_stride:
            if stride > 0:
                pc_floor_ = np.floor(orig_pc_ / stride) * stride
                floor_inds = sparse_quantize(pc_floor_, return_index=True)
                pc_floor = pc_floor_[floor_inds]

                if stride > 1:
                    pc_at_stride_ = np.floor(pc / stride) * stride
                    down_inds = sparse_quantize(pc_at_stride_, return_index=True)
                    pc_at_stride = pc_at_stride_[down_inds]
                else:
                    pc_at_stride = pc

                padded_pc = pad_voxel(pc_at_stride.astype(np.int32), pc_floor.astype(np.int32), stride=stride).astype(np.float32)
                padded_num = len(padded_pc)
                feat = np.pad(feat, ((0, padded_num), (0, 0)), 'constant', constant_values=0)
                labels = np.pad(labels, (0, padded_num), 'constant', constant_values=255)
                #print(len(pc), '+', padded_num, '->', len(pc) + padded_num)
                pc = np.concatenate([pc, padded_pc], axis=0)

        if len(self.voxel_pad_stride) > 0:
            # if reorder the pc, inverse_map should also be mapped use reorder_inverse
            voxel_num_after_padded = len(pc)
            reorder_inds, reorder_inverse = sparse_quantize(pc, return_index=True, return_invs=True)
            assert(voxel_num_after_padded == len(reorder_inds))        
            pc = pc[reorder_inds]
            labels = labels[reorder_inds]
            feat = feat[reorder_inds]
            inverse_map = reorder_inverse[inverse_map]

            assert((pc[inverse_map]==pc_).all())

        lidar = SparseTensor(feat, pc)
        labels = SparseTensor(labels, pc)
        labels_ = PointTensor(labels_, orig_pc_)
        inverse_map = SparseTensor(inverse_map, pc_)
        
        return {
            'lidar': lidar,
            'targets': labels,
            'targets_mapped': labels_,
            'inverse_map': inverse_map,
            'file_name': self.files[index]
        }

    
    @staticmethod
    def collate_fn(inputs):
        collated = sparse_collate_fn(inputs)
        labels_ = collated['targets_mapped']
        batchsize = len(labels_)
        all_labels_ = [l.F for l in labels_]
        all_pts_ = [l.C for l in labels_]
        all_labels_ = np.concatenate(all_labels_)
        for i in range(batchsize):
            all_pts_[i] = np.pad(all_pts_[i], ((0, 0), (0, 1)), 'constant', constant_values=i)
        all_pts_ = np.concatenate(all_pts_)
        collated['targets_mapped'] = PointTensor(torch.from_numpy(all_labels_), torch.from_numpy(all_pts_))
        return collated