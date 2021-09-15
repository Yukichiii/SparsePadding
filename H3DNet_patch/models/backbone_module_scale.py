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
from pointnet2_modules import PointnetSAModuleVotes, PointnetSAModuleVotesWith, PointnetFPModule, PointnetPlaneVotes
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
    def __init__(self, input_feature_dim=0, scale=1):
        super().__init__()

        self.sa1 = PointnetSAModuleVotes(
                npoint=2048,
                radius=0.2,
                nsample=64,
                mlp=[input_feature_dim, 64*scale, 64*scale, 128*scale],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa2 = PointnetSAModuleVotes(
                npoint=1024,
                radius=0.4,
                nsample=32,
                mlp=[128*scale, 128*scale, 128*scale, 256*scale],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa3 = PointnetSAModuleVotes(
                npoint=512,
                radius=0.8,
                nsample=16,
                mlp=[256*scale, 128*scale, 128*scale, 256*scale],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa4 = PointnetSAModuleVotes(
                npoint=256,
                radius=1.2,
                nsample=16,
                mlp=[256*scale, 128*scale, 128*scale, 256*scale],
                use_xyz=True,
                normalize_xyz=True
            )

        if scale == 1:
            self.fp1 = PointnetFPModule(mlp=[256+256,512,512])
            self.fp2 = PointnetFPModule(mlp=[512+256,512,512])
        else:
            self.fp1 = PointnetFPModule(mlp=[256*scale+256*scale,256*scale,256*scale])
            self.fp2 = PointnetFPModule(mlp=[256*scale+256*scale,256*scale,256*scale])

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None, mode=''):
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

        end_points['sa0_xyz'+mode] = xyz
        end_points['sa0_features'+mode] = features
        
        # --------- 4 SET ABSTRACTION LAYERS ---------
        if mode != '':
            ### Reuse inds from point
            xyz, features, fps_inds = self.sa1(xyz, features, inds=end_points['sa1_inds'])
        else:
            xyz, features, fps_inds = self.sa1(xyz, features)
        end_points['sa1_inds'+mode] = fps_inds
        end_points['sa1_xyz'+mode] = xyz
        end_points['sa1_features'+mode] = features

        if mode != '':
            xyz, features, fps_inds = self.sa2(xyz, features, inds=end_points['sa2_inds']) # this fps_inds is just 0,1,...,1023
        else:
            xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
        end_points['sa2_inds'+mode] = fps_inds
        end_points['sa2_xyz'+mode] = xyz
        end_points['sa2_features'+mode] = features

        if mode != '':
            xyz, features, fps_inds = self.sa3(xyz, features, inds=end_points['sa3_inds']) # this fps_inds is just 0,1,...,511
        else:
            xyz, features, fps_inds = self.sa3(xyz, features) # this fps_inds is just 0,1,...,1023
        end_points['sa3_inds'+mode] = fps_inds
        end_points['sa3_xyz'+mode] = xyz
        end_points['sa3_features'+mode] = features

        if mode != '':
            xyz, features, fps_inds = self.sa4(xyz, features, inds=end_points['sa4_inds']) # this fps_inds is just 0,1,...,255
        else:
            xyz, features, fps_inds = self.sa4(xyz, features) # this fps_inds is just 0,1,...,255
        end_points['sa4_inds'+mode] = fps_inds
        end_points['sa4_xyz'+mode] = xyz
        end_points['sa4_features'+mode] = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(end_points['sa3_xyz'+mode], end_points['sa4_xyz'+mode], end_points['sa3_features'+mode], end_points['sa4_features'+mode])
        features = self.fp2(end_points['sa2_xyz'+mode], end_points['sa3_xyz'+mode], end_points['sa2_features'+mode], features)
        end_points['fp2_features'+mode] = features
        end_points['fp2_xyz'+mode] = end_points['sa2_xyz'+mode]
        num_seed = end_points['fp2_xyz'+mode].shape[1]
        end_points['fp2_inds'+mode] = end_points['sa1_inds'+mode][:,0:num_seed] # indices among the entire input point clouds
        return end_points

class Pointnet2BackboneRefine(nn.Module):
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

        self.sa1 = PointnetSAModuleVotesWith(
                npoint=2048,
                radius=0.2,
                nsample=64,
                mlp=[input_feature_dim+18+1, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa2 = PointnetSAModuleVotesWith(
                npoint=1024,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa3 = PointnetSAModuleVotesWith(
                npoint=512,
                radius=0.8,
                nsample=16,
                mlp=[256, 128, 128, 256],### Add the indicator info here
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa4 = PointnetSAModuleVotesWith(
                npoint=256,
                radius=1.2,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.fp1 = PointnetFPModule(mlp=[256+256,256,256])
        self.fp2 = PointnetFPModule(mlp=[256+256,256,256])
        #self.fp1 = PointnetFPModule(mlp=[128+128,128,128])
        #self.fp2 = PointnetFPModule(mlp=[128+128,128,128])
        #self.fp3 = PointnetFPModule(mlp=[256+128,256,256])
        #self.fp4 = PointnetFPModule(mlp=[256,128,128])

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, center_points: torch.cuda.FloatTensor, cue_points: torch.cuda.FloatTensor, matching: torch.cuda.FloatTensor, matching_sem: torch.cuda.FloatTensor, floor_height: torch.cuda.FloatTensor, end_points=None, mode=''):
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

        end_points['sa0_xyz'+mode] = xyz
        end_points['sa0_features'+mode] = features

        #center_points = end_points['center_points']
        #cue_points = end_points['cue_points']#.view(batch_size, -1, 3).float()
        
        obj_points = torch.cat((center_points, cue_points), dim=1)
        #center_matching = torch.max(matching.view(batch_size, 18, 256), dim=1)[0]
        center_matching = end_points['match_center']

        center_sem = torch.cuda.FloatTensor(batch_size, 256, 18).zero_()### Need to change to config sem later
        center_sem.scatter_(2, matching_sem[:,:256].unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_size_cluster)
        cue_sem = torch.cuda.FloatTensor(batch_size, 256*18, 18).zero_()
        cue_sem.scatter_(2, matching_sem[:,256:].unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_size_cluster)

        center_feature = torch.cat(((center_points[:,:,2] - floor_height.unsqueeze(-1)).unsqueeze(1), center_matching.unsqueeze(1), center_sem.transpose(2,1).contiguous()), dim=1) ### Need to make the floor height an option
        cue_feature = torch.cat(((cue_points[:,:,2] - floor_height.unsqueeze(-1)).unsqueeze(1), matching.unsqueeze(1), cue_sem.transpose(2,1).contiguous()), dim=1)
        other_features = torch.cat((features, torch.cuda.FloatTensor(batch_size, 19, features.shape[-1]).zero_()), dim=1)
        
        features = torch.cat((center_feature, cue_feature, other_features), dim=2)
        #features = torch.cat((cue_feature, other_features), dim=2)
        
        # --------- 4 SET ABSTRACTION LAYERS ---------
        ### Concatenate the 
        #xyz, features, fps_inds = self.sa1(obj_points, xyz, features, inds=end_points['sa1_inds'])
        xyz, features, fps_inds = self.sa1(obj_points, xyz, features)
        end_points['sa1_inds'+mode] = fps_inds
        end_points['sa1_xyz'+mode] = xyz
        end_points['sa1_features'+mode] = features
        
        #xyz, features, fps_inds = self.sa2(xyz[:,:256*18,:].contiguous(), xyz[:,256*18:,:].contiguous(), features) # this fps_inds is just 0,1,...,1023
        xyz, features, fps_inds = self.sa2(xyz[:,:256*19,:].contiguous(), xyz[:,256*19:,:].contiguous(), features, inds=end_points['sa2_inds']) # this fps_inds is just 0,1,...,1023
        end_points['sa2_inds'+mode] = fps_inds
        end_points['sa2_xyz'+mode] = xyz
        end_points['sa2_features'+mode] = features

        ### Append the surface and line info here
        '''
        center_ind = torch.cuda.FloatTensor(batch_size, 4, 256).zero_()
        center_ind[:,0,:] = 1.0
        surfacez_ind = torch.cuda.FloatTensor(batch_size, 4, 256*2).zero_()
        surfacez_ind[:,1,:] = 1.0
        surfacexy_ind = torch.cuda.FloatTensor(batch_size, 4, 256*4).zero_()
        surfacexy_ind[:,2,:] = 1.0
        line_ind = torch.cuda.FloatTensor(batch_size, 4, 256*12).zero_()
        line_ind[:,3,:] = 1.0
        cue_ind = torch.cat((torch.cuda.FloatTensor(batch_size, 1, 1024).zero_(), end_points["pred_z_ind"].unsqueeze(1), end_points["pred_xy_ind"].unsqueeze(1), end_points["pred_line_ind"].unsqueeze(1)), dim=1)
        ind_feature = torch.cat((center_ind, surfacez_ind, surfacexy_ind, line_ind, cue_ind), dim=2)
        features = torch.cat((features, ind_feature), dim=1)
        '''
        #xyz, features, fps_inds = self.sa3(xyz[:,:256*18,:].contiguous(), xyz[:,256*18:,:].contiguous(), features) # this fps_inds is just 0,1,...,1023
        xyz, features, fps_inds = self.sa3(xyz[:,:256*19,:].contiguous(), xyz[:,256*19:,:].contiguous(), features, inds=end_points['sa3_inds']) # this fps_inds is just 0,1,...,1023
        end_points['sa3_inds'+mode] = fps_inds
        end_points['sa3_xyz'+mode] = xyz
        end_points['sa3_features'+mode] = features

        #xyz, features, fps_inds = self.sa4(xyz[:,:256*18,:].contiguous(), xyz[:,256*18:,:].contiguous(), features) # this fps_inds is just 0,1,...,1023
        xyz, features, fps_inds = self.sa4(xyz[:,:256*19,:].contiguous(), xyz[:,256*19:,:].contiguous(), features, inds=end_points['sa4_inds']) # this fps_inds is just 0,1,...,1023
        end_points['sa4_inds'+mode] = fps_inds
        end_points['sa4_xyz'+mode] = xyz
        end_points['sa4_features'+mode] = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        #features = self.fp1(end_points['sa3_xyz'+mode], end_points['sa4_xyz'+mode], end_points['sa3_features'+mode], end_points['sa4_features'+mode])
        #features = self.fp2(end_points['sa2_xyz'+mode], end_points['sa3_xyz'+mode], end_points['sa2_features'+mode], features)
        features = self.fp1(end_points['sa3_xyz'+mode], end_points['sa4_xyz'+mode][:,256*19:,:].contiguous(), end_points['sa3_features'+mode], end_points['sa4_features'+mode][:,:,256*19:].contiguous())
        features = self.fp2(end_points['sa2_xyz'+mode][:,:256*19,:].contiguous(), end_points['sa3_xyz'+mode][:,256*19:,:].contiguous(), end_points['sa2_features'+mode][:,:,:256*19].contiguous(), features[:,:,256*19:].contiguous())
        end_points['fp2_features'+mode] = features
        end_points['fp2_xyz'+mode] = end_points['sa2_xyz'+mode][:,:256*19,:].contiguous()
        num_seed = end_points['fp2_xyz'+mode].shape[1]
        end_points['fp2_inds'+mode] = end_points['sa1_inds'+mode][:,0:num_seed] # indices among the entire input point clouds
        return end_points
    
class Pointnet2BackbonePlane(nn.Module):
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

        self.sa1 = PointnetPlaneVotes(
                npoint=2048,
                radius=0.2,
                nsample=64,
                mlp=[input_feature_dim, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa2 = PointnetPlaneVotes(
                npoint=1024,
                radius=0.4,
                nsample=32,
                mlp=[128*2, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa3 = PointnetPlaneVotes(
                npoint=512,
                radius=0.8,
                nsample=16,
                mlp=[256, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa4 = PointnetPlaneVotes(
                npoint=256,
                radius=1.2,
                nsample=16,
                mlp=[256, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        self.fp1 = PointnetFPModule(mlp=[256+256,256,256])
        self.fp2 = PointnetFPModule(mlp=[256+256,256,256])
        #self.fp3 = PointnetFPModule(mlp=[256+128,256,256])
        #self.fp4 = PointnetFPModule(mlp=[256,128,128])

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None, mode='plane'):
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

        end_points['sa0_xyz'+mode] = xyz
        end_points['sa0_features'+mode] = features
        
        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        end_points['sa1_inds'+mode] = fps_inds
        end_points['sa1_xyz'+mode] = xyz
        end_points['sa1_features'+mode] = features

        xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
        end_points['sa2_inds'+mode] = fps_inds
        end_points['sa2_xyz'+mode] = xyz
        end_points['sa2_features'+mode] = features

        xyz, features, fps_inds = self.sa3(xyz, features) # this fps_inds is just 0,1,...,511
        end_points['sa3_xyz'+mode] = xyz
        end_points['sa3_features'+mode] = features

        xyz, features, fps_inds = self.sa4(xyz, features) # this fps_inds is just 0,1,...,255
        end_points['sa4_xyz'+mode] = xyz
        end_points['sa4_features'+mode] = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(end_points['sa3_xyz'+mode], end_points['sa4_xyz'+mode], end_points['sa3_features'+mode], end_points['sa4_features'+mode])
        features = self.fp2(end_points['sa2_xyz'+mode], end_points['sa3_xyz'+mode], end_points['sa2_features'+mode], features)
        end_points['fp2_features'+mode] = features
        end_points['fp2_xyz'+mode] = end_points['sa2_xyz'+mode]
        num_seed = end_points['fp2_xyz'+mode].shape[1]
        end_points['fp2_inds'+mode] = end_points['sa1_inds'+mode][:,0:num_seed] # indices among the entire input point clouds
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
    def __init__(self, input_feature_dim=0, pad_voxel_stride=[], pad_voxel_method='none', normalize_weights=True, nearest=True, scale=1):
        super().__init__()
        self.voxel_size = 0.02
        self.seed_num = 1024
        if scale==1:
            self.out_channels = 512
        else:
            self.out_channels = 256 * scale
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

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None, mode=''):
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

        end_points['fp2_features'+mode] = output.view(-1, self.seed_num, self.out_channels).permute(0, 2, 1)
        end_points['fp2_xyz'+mode] = seed_xyz
        num_seed = self.seed_num
        end_points['fp2_inds'+mode] = inds
        return end_points


if __name__=='__main__':
    backbone_net = Pointnet2Backbone(input_feature_dim=3).cuda()
    print(backbone_net)
    backbone_net.eval()
    out = backbone_net(torch.rand(4,8192,6).cuda())
    for key in sorted(out.keys()):
        print(key, '\t', out[key].shape)
