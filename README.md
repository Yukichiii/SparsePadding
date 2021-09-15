# Interpolation-Aware Padding for 3D Sparse Convolutional Neural Networks
Created by <a href="https://yukichiii.github.io/" target="_blank">Yu-Qi Yang</a>, <a href="https://wang-ps.github.io/" target="_blank">Peng-Shuai Wang</a> and <a href="https://xueyuhanlang.github.io/" target="_blank">Yang Liu</a>.

![overview](overview/overview.pdf)

## Introduction
This repository is code release for our paper (arXiv report [here](https://arxiv.org/abs/2108.06925)).

Sparse voxel-based 3D convolutional neural networks (CNNs) are widely used for various 3D vision tasks. Sparse voxel-based 3D CNNs create sparse non-empty voxels from the 3D input and perform 3D convolution operations on them only. We propose a simple yet effective padding scheme --- interpolation-aware padding to pad a few empty voxels adjacent to the non-empty voxels and involve them in the 3D CNN computation so that all neighboring voxels exist when computing point-wise features via the trilinear interpolation. For fine-grained 3D vision tasks where point-wise features are essential, like semantic segmentation and 3D detection, our network achieves higher prediction accuracy than the existing networks using the nearest neighbor interpolation or the normalized trilinear interpolation with the zero-padding or the octree-padding scheme. Through extensive comparisons on various 3D segmentation and detection tasks, we demonstrate the superiority of 3D sparse CNNs with our padding scheme in conjunction with feature interpolation.

The core code of our padding methods is shown in pad_voxel_for_ti.py, and the normalized trilinear interpolation is implemented in mink_utils.py. You can use it as a plug-in in any frameworks based on sparse 3D CNNs. You can follow the usage in our experiments below.

## Environment
Our Experiments are based on torch_sparse, it is a re-implementation of MinkowskiEngine. It use Google's sparse hash map project to accelerate the sparse operations. So we choose this library as our tool.

We test our code in an environment with PyTorch1.7 + CUDA10.1.

All the libraries we used for experiments are third-party open-source codes.

You can create the environment following this:

    conda create --name padding python=3.7 -y
    conda activate padding
    conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch -y

    sudo apt-get install libsparsehash-dev
    pip install git+https://github.com/mit-han-lab/torchsparse.git@v1.1.0

## Example Usage
### Semantic Segmentation


### Object Detection Task
For Object Detection Task On ScanNet, we use VoteNet and H3DNet as our baseline. First, you should clone the code of VoteNet and H3DNet.
For VoteNet:

    git clone https://github.com/facebookresearch/votenet.git

Prepare the boundingbox data of ScanNet and install the Python dependencies by following `votenet/README.md` and `H3DNet/README.md`. The processed data should be in this folder `votenet/scannet/scannet_train_detection_data` and `H3DNet/scannet/scannet_train_detection_data`.

To train the detection network which uses sparse 3D CNN backbone with our padding scheme, you should copy the code of our modified model to the cloned repo.

    cp -rf votenet_patch votenet

Then compile pointnet++ used by both VoteNet and H3DNet.

    cd VoteNet/pointnet2
    python setup.py install

To train the model with our interpolation-aware padding and trilinear interpolation:

    python train_fixed.py --dataset scannet --log_dir logs/scannet_detection_minkunet_points_pad --num_point 40000 --use_color --model votenet_mink_unet --weight_decay 0.0001 --pad_voxel_stride 2,4 --pad_voxel_method trilinear --rand_seed 123456789

For H3DNet, it's similar to prepare the code.

    git clone https://github.com/zaiweizhang/H3DNet.git
    cp -rf H3DNet_patch H3DNet
    cd H3DNet/pointnet2
    python setup.py install

Then you can train the model with:

    python train_1bb_minkunet.py --data_path scannet/scannet_train_detection_data --dataset scannet --log_dir logs/minkunet_pad --num_point 40000 --model hdnet_1bb_minkunet --batch_size 6  --pad_voxel_stride 4 --pad_voxel_method trilinear
