# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

from pointnet2_utils import furthest_point_sample
from pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule
from models.minkunet import MinkUNet
from torchsparse import SparseTensor, PointTensor
from torchsparse.utils import sparse_quantize
from models.pad_voxel_for_ti import pad_voxel

class Pointnet2Backbone(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """
    def __init__(self, input_feature_dim=0):
        super().__init__()

        self.sa1 = PointnetSAModuleVotes(
                npoint=2048,
                radius=0.2,
                nsample=64,
                mlp=[input_feature_dim, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa2 = PointnetSAModuleVotes(
                npoint=1024,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa3 = PointnetSAModuleVotes(
                npoint=512,
                radius=0.8,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa4 = PointnetSAModuleVotes(
                npoint=256,
                radius=1.2,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.fp1 = PointnetFPModule(mlp=[256+256,256,256])
        self.fp2 = PointnetFPModule(mlp=[256+256,256,256])

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        if not end_points: end_points = {}
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        end_points['sa1_inds'] = fps_inds
        end_points['sa1_xyz'] = xyz
        end_points['sa1_features'] = features

        xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
        end_points['sa2_inds'] = fps_inds
        end_points['sa2_xyz'] = xyz
        end_points['sa2_features'] = features

        xyz, features, fps_inds = self.sa3(xyz, features) # this fps_inds is just 0,1,...,511
        end_points['sa3_xyz'] = xyz
        end_points['sa3_features'] = features

        xyz, features, fps_inds = self.sa4(xyz, features) # this fps_inds is just 0,1,...,255
        end_points['sa4_xyz'] = xyz
        end_points['sa4_features'] = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'], end_points['sa4_features'])
        features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)
        end_points['fp2_features'] = features
        end_points['fp2_xyz'] = end_points['sa2_xyz']
        num_seed = end_points['fp2_xyz'].shape[1]
        end_points['fp2_inds'] = end_points['sa1_inds'][:,0:num_seed] # indices among the entire input point clouds
        return end_points

class MINKUNetBackbone(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """
    def __init__(self, input_feature_dim=0, pad_voxel_stride=[], pad_voxel_method='none', normalize_weights=True, nearest=True):
        super().__init__()
        self.voxel_size = 0.02
        self.seed_num = 1024
        self.out_channels = 256
        self.pad_voxel_stride = pad_voxel_stride
        assert(type(self.pad_voxel_stride) is list)
        self.pad_voxel_stride.sort()
        self.pad_voxel_method = pad_voxel_method
        assert(type(self.pad_voxel_method) is str)
        self.nearest = nearest
        print("====================================")
        print("normalize_weights:", normalize_weights)
        print("pad_voxel_stride:", pad_voxel_stride)
        print("pad_voxel_method:", pad_voxel_method)
        print("nearest: ", nearest)
        print("====================================")
        self.mink_unet = MinkUNet(in_channels=input_feature_dim, out_channels=self.out_channels, normalize_weights=normalize_weights)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def transform_input(self, xyz, features, seed_xyz):
        batch_size = xyz.shape[0]
        pc_list = []
        feats_list = []
        pts_list = []
        xyz = xyz.cpu().numpy()
        features = features.cpu().numpy()
        seed_xyz = seed_xyz.cpu().numpy()
        for i in range(batch_size):
            pc_ = xyz[i].astype(np.float64)
            lower_bound = pc_.min(axis=0)
            pc_ -= lower_bound
            pc_ /= self.voxel_size
            pc_r_ = np.round(pc_)
            inds = sparse_quantize(pc_r_, return_index=True)
            pc = pc_r_[inds]
            feats = features[i][inds]

            init_pc = pc.copy()
            for stride in self.pad_voxel_stride:
                if stride > 0:
                    if self.pad_voxel_method == 'full':
                        pc_floor_ = np.floor(init_pc / stride) * stride
                        floor_inds = sparse_quantize(pc_floor_, return_index=True)
                        pc_floor = pc_floor_[floor_inds]
                        pad_num = 27
                    elif self.pad_voxel_method == 'octree':
                        pc_floor_ = np.floor(init_pc / (stride*2)) * (stride*2)
                        floor_inds = sparse_quantize(pc_floor_, return_index=True)
                        pc_floor = pc_floor_[floor_inds]
                        pad_num = 8
                    elif self.pad_voxel_method == 'trilinear':
                        pc_floor_ = np.floor(pc_ / stride) * stride
                        floor_inds = sparse_quantize(pc_floor_, return_index=True)
                        pc_floor = pc_floor_[floor_inds]
                        pad_num = 8
                    elif self.pad_voxel_method == '' or self.pad_voxel_method == 'none':
                        break
                    else:
                        raise NotImplemented(self.pad_voxel_method)

                    pc_at_stride_ = np.floor(pc / stride) * stride
                    qt_inds = sparse_quantize(pc_at_stride_, return_index=True)
                    pc_at_stride = pc_at_stride_[qt_inds]

                    padded_pc = pad_voxel(pc_at_stride.astype(np.int32), pc_floor.astype(np.int32), pad_num=pad_num, stride=stride).astype(np.float32)
                    padded_num = len(padded_pc)
                    feats = np.pad(feats, ((0, padded_num), (0, 0)), 'constant', constant_values=0)
                    pc = np.concatenate([pc, padded_pc], axis=0)
            
            if len(self.pad_voxel_stride) > 0:
                # if reorder the sp_coords, inverse_map should also be mapped use reorder_inverse
                voxel_num_after_padded = len(pc)
                reorder_inds = sparse_quantize(pc, return_index=True)
                assert(voxel_num_after_padded == len(reorder_inds))        
                pc = pc[reorder_inds]
                feats = feats[reorder_inds]

            pc = np.pad(pc, ((0, 0), (0, 1)), 'constant', constant_values=i)
            pc_list.append(pc)
            feats_list.append(feats)
            pts = (seed_xyz[i] - lower_bound) / self.voxel_size
            pts = np.pad(pts, ((0, 0), (0, 1)), 'constant', constant_values=i)
            pts_list.append(pts)

        pc_list = np.concatenate(pc_list, axis=0).astype(np.int32)
        feats_list = np.concatenate(feats_list, axis=0)
        pts_list = np.concatenate(pts_list, axis=0)
        
        pc_list = torch.from_numpy(pc_list).cuda()
        feats_list = torch.from_numpy(feats_list).cuda()
        pts_list = torch.from_numpy(pts_list).cuda()

        sp_input = SparseTensor(feats_list, pc_list)
        pts = PointTensor(pts_list, pts_list.double())
        return sp_input, pts

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        if not end_points: end_points = {}
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)
        features = features.permute(0, 2, 1)
        inds = furthest_point_sample(xyz, self.seed_num).long()
        
        seed_xyz = []
        for i in range(batch_size):
            seed_xyz.append(xyz[i][inds[i]])
        seed_xyz = torch.stack(seed_xyz)
        
        sp_input, pts = self.transform_input(xyz, features, seed_xyz)
        output = self.mink_unet(sp_input, pts, nearest=self.nearest)

        end_points['fp2_features'] = output.view(-1, self.seed_num, self.out_channels).permute(0, 2, 1)
        end_points['fp2_xyz'] = seed_xyz
        num_seed = self.seed_num
        end_points['fp2_inds'] = inds
        return end_points

if __name__=='__main__':
    backbone_net = Pointnet2Backbone(input_feature_dim=3).cuda()
    print(backbone_net)
    backbone_net.eval()
    out = backbone_net(torch.rand(16,20000,6).cuda())
    for key in sorted(out.keys()):
        print(key, '\t', out[key].shape)
