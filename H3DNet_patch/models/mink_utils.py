import torchsparse
import torchsparse.nn as spnn
import torchsparse.nn.functional as spf
from torchsparse.sparse_tensor import SparseTensor
from torchsparse.point_tensor import PointTensor
from torchsparse.utils.kernel_region import *
from torchsparse.utils.helpers import *


__all__ = ['initial_voxelize', 'point_to_voxel', 'voxel_to_point']


# z: PointTensor
# return: SparseTensor
def initial_voxelize(z, init_res, after_res):
    new_float_coord = torch.cat(
        [(z.C[:, :3] * init_res) / after_res, z.C[:, -1].view(-1, 1)], 1)

    pc_hash = spf.sphash(torch.floor(new_float_coord).int())
    sparse_hash = torch.unique(pc_hash)
    idx_query = spf.sphashquery(pc_hash, sparse_hash)
    counts = spf.spcount(idx_query.int(), len(sparse_hash))

    inserted_coords = spf.spvoxelize(torch.floor(new_float_coord), idx_query,
                                   counts)
    inserted_coords = torch.round(inserted_coords).int()
    inserted_feat = spf.spvoxelize(z.F, idx_query, counts)

    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)
    new_tensor.check()
    z.additional_features['idx_query'][1] = idx_query
    z.additional_features['counts'][1] = counts
    z.C = new_float_coord

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: SparseTensor
def point_to_voxel(x, z):
    if z.additional_features is None or z.additional_features.get('idx_query') is None\
       or z.additional_features['idx_query'].get(x.s) is None:
        #pc_hash = hash_gpu(torch.floor(z.C).int())
        pc_hash = spf.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s).int() * x.s,
                z.C[:, -1].int().view(-1, 1)
            ], 1))
        sparse_hash = spf.sphash(x.C)
        idx_query = spf.sphashquery(pc_hash, sparse_hash)
        counts = spf.spcount(idx_query.int(), x.C.shape[0])
        z.additional_features['idx_query'][x.s] = idx_query
        z.additional_features['counts'][x.s] = counts
    else:
        idx_query = z.additional_features['idx_query'][x.s]
        counts = z.additional_features['counts'][x.s]

    inserted_feat = spf.spvoxelize(z.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_feat, x.C, x.s)
    new_tensor.coord_maps = x.coord_maps
    new_tensor.kernel_maps = x.kernel_maps

    return new_tensor


# # x: SparseTensor, z: PointTensor
# # return: PointTensor
# def voxel_to_point(x, z, nearest=False):
#     if z.idx_query is None or z.weights is None or z.idx_query.get(
#             x.s) is None or z.weights.get(x.s) is None:
#         kr = KernelRegion(2, x.s, 1)
#         off = kr.get_kernel_offset().to(z.F.device)
#         #old_hash = kernel_hash_gpu(torch.floor(z.C).int(), off)
#         old_hash = spf.sphash(
#             torch.cat([
#                 torch.floor(z.C[:, :3] / x.s).int() * x.s,
#                 z.C[:, -1].int().view(-1, 1)
#             ], 1), off)
#         pc_hash = spf.sphash(x.C.to(z.F.device))
#         idx_query = spf.sphashquery(old_hash, pc_hash)
#         weights = spf.calc_ti_weights(z.C, idx_query,
#                                   scale=x.s).transpose(0, 1).contiguous().float()
#         idx_query = idx_query.transpose(0, 1).contiguous()
#         if nearest:
#             weights[:, 1:] = 0.
#             idx_query[:, 1:] = -1
#         new_feat = spf.spdevoxelize(x.F, idx_query, weights)
#         new_tensor = PointTensor(new_feat,
#                                  z.C,
#                                  idx_query=z.idx_query,
#                                  weights=z.weights)
#         new_tensor.additional_features = z.additional_features
#         new_tensor.idx_query[x.s] = idx_query
#         new_tensor.weights[x.s] = weights
#         z.idx_query[x.s] = idx_query
#         z.weights[x.s] = weights

#     else:
#         new_feat = spf.spdevoxelize(x.F, z.idx_query.get(x.s), z.weights.get(x.s))
#         new_tensor = PointTensor(new_feat,
#                                  z.C,
#                                  idx_query=z.idx_query,
#                                  weights=z.weights)
#         new_tensor.additional_features = z.additional_features

#     return new_tensor


def calc_ti_weights(coords, idx_query, normalize_weights=True):
    mask = torch.cuda.FloatTensor(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    frac = coords - torch.floor(coords)
    frac = frac[:, 0:3]
    frac = torch.cuda.FloatTensor([1, 1, 1]) - mask - torch.unsqueeze(frac, dim=1)
    weights = torch.abs(torch.prod(frac, dim=2)).t()
    weights[idx_query == -1] = 0
    if normalize_weights:
        weights /= weights.sum(0) + 1e-8
    return weights

# x: SparseTensor, z: PointTensor
# return: PointTensor
def voxel_to_point(x, z, normalize_weights=True):
    h = x.C.shape[0]
    npt = z.C.shape[0]  
    if z.idx_query is None or z.weights is None or z.idx_query.get(
            x.s) is None or z.weights.get(x.s) is None:
        kr = KernelRegion(2, x.s, 1)
        off = kr.get_kernel_offset().to(z.F.device)
        #old_hash = kernel_hash_gpu(torch.floor(z.C).int(), off)
        old_hash = spf.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s).int() * x.s,
                z.C[:, -1].int().view(-1, 1)
            ], 1), off)
        pc_hash = spf.sphash(x.C.to(z.F.device))
        idx_query = spf.sphashquery(old_hash, pc_hash)
        weights = calc_ti_weights(z.C, idx_query, normalize_weights).transpose(0, 1).contiguous()

        not_found_all_neighbor_flag = torch.prod(idx_query!=-1, dim=0)==0
        not_found_all_neighbor_coords = z.C[not_found_all_neighbor_flag]

        idx_query = idx_query.transpose(0, 1).contiguous()

        ids = torch.arange(npt).view(npt, 1).cuda()
        ids = ids.repeat(1, 8).view(-1)
        idx = idx_query.view(-1)
        flgs = idx > -1
        ids = ids[flgs]
        idx = idx[flgs]
        weights = weights.view(-1)[flgs].float()
        
        indices = torch.cat([torch.unsqueeze(ids, dim=1), torch.unsqueeze(idx, dim=1)], dim=1).long()

        mat = torch.sparse.FloatTensor(indices.t(), weights, torch.Size([npt, h])).cuda()

        new_feat = torch.sparse.mm(mat, x.F)

        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features
        new_tensor.idx_query[x.s] = idx_query
        new_tensor.weights[x.s] = weights
        z.idx_query[x.s] = idx_query
        z.weights[x.s] = weights

    else:
        weights = z.weights.get(x.s)
        idx_query = z.idx_query.get(x.s)
             
        ids = torch.arange(npt).view(npt, 1).cuda()
        ids = ids.repeat(1, 8).view(-1)
        idx = idx_query.view(-1)
        flgs = idx > -1
        ids = ids[flgs]
        idx = idx[flgs]
        weights = weights.view(-1)[flgs]
        indices = torch.cat([torch.unsqueeze(ids, dim=1), torch.unsqueeze(idx, dim=1)], dim=1).long()

        mat = torch.sparse.FloatTensor(indices.t(), weights, torch.Size([npt, h])).cuda()
        new_feat = torch.sparse.mm(mat, x.F)
    
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features

    return new_tensor

def nearest_voxel(x, z):
    #old_hash = kernel_hash_gpu(torch.floor(z.C).int(), off)
    old_hash = spf.sphash(
        torch.cat([
            torch.floor(torch.round(z.C[:, :3]) / x.s).int() * x.s,
            z.C[:, -1].int().view(-1, 1)
        ], 1))
    pc_hash = spf.sphash(x.C.to(z.F.device))
    idx_query = spf.sphashquery(old_hash, pc_hash)
    assert((idx_query!=-1).all())
    new_feat = x.F[idx_query, :]
    new_tensor = PointTensor(new_feat,
                             z.C)

    return new_tensor