import numpy as np

__all__ = ['pad_voxel']
def calc_key(points, bound):
    points = points.astype(np.int64)
    bound = bound.astype(np.int64)
    key = points[:, 0] * bound[1] * bound[2] + points[:, 1] * bound[2] + points[:, 2]
    return key

def calc_unique_pts(pts):
    bound = (np.max(pts, axis=0) + 1).astype(np.int64)
    key = calc_key(pts, bound)
    _, index = np.unique(key, return_index=True)
    unique_pts = pts[index]
    return unique_pts

# pad num == 27:
# pad for 27 neighbors
# pad num == 8:
# pad for octree or tri_itp
def pad_neighbors(points, bound, flag, pad_num=8, stride=1):
    if pad_num == 27:
        direction = np.array([[-1, -1, -1], [-1, -1, 0], [-1, -1, 1], [-1, 0, -1], [-1, 0, 0], [-1, 0, 1], [-1, 1, -1], [-1, 1, 0], [-1, 1, 1], [0, -1, -1], [0, -1, 0], [0, -1, 1], [0, 0, -1], [0, 0, 0], [0, 0, 1], [0, 1, -1], [0, 1, 0], [0, 1, 1], [1, -1, -1], [1, -1, 0], [1, -1, 1], [1, 0, -1], [1, 0, 0], [1, 0, 1], [1, 1, -1], [1, 1, 0], [1, 1, 1]]) * stride
    elif pad_num == 8:
        direction = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]) * stride
    neighbors = points.repeat(len(direction), 0).reshape(-1, len(direction), 3) + direction
    neighbors = neighbors.reshape(-1, 3)
    mask = np.prod(neighbors >= 0, axis=1).astype(bool)
    neighbors = neighbors[mask]
    mask = np.prod(neighbors < bound, axis=1).astype(bool)
    neighbors = neighbors[mask]
    key = calc_key(neighbors, bound)
    key, unique_index = np.unique(key, return_index=True)
    neighbors = neighbors[unique_index]
    neighbors = neighbors[flag[key]]
    return neighbors

# pc -> flag
# pc_floor -> new_padded
def pad_voxel(pc, pc_floor, pad_num=8, stride=1):
    num_before_pad = len(pc)

    bound = (np.max(pc, axis=0) + stride + 1).astype(np.int64)

    flag = np.ones(bound[0] * bound[1] * bound[2], dtype=np.bool)
    key = calc_key(pc, bound)
    flag[key] = False

    new_padded = pad_neighbors(pc_floor, bound, flag, pad_num=pad_num, stride=stride)
    
    num_padded = len(new_padded)
    num_after_pad = num_before_pad + num_padded
    #print("stride:", stride, ", pad_num:", pad_num, ":", num_before_pad, '->', num_after_pad)
    return new_padded
