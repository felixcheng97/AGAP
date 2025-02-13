import os
import torch
import numpy as np
import imageio
import torch.nn.functional as F
import glob
import scipy
from PIL import Image


def load_neuvf_data(basedir, factor, x, y, z):
    meta = np.load(os.path.join(basedir, 'poses_bounds.npz'))
    poses = meta['poses']
    K = meta['intrinsics']
    xyz_min = meta['box_min']
    xyz_max = meta['box_max']

    poses = np.concatenate([np.concatenate([ poses[..., :, 0:1], -poses[..., :, 1:2], -poses[..., :, 2:3]], axis=-1), poses[..., :, 3:]], axis=-1)

    center = np.array([0., 0., 0.])
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    poses_center = viewmatrix(vec2, up, center)
    poses, xyz_min, xyz_max = recenter_pose(poses, poses_center, xyz_min, xyz_max)

    center = np.array([x, y, z])
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    poses_center = viewmatrix(vec2, up, center)
    poses, xyz_min, xyz_max = recenter_pose(poses, poses_center, xyz_min, xyz_max)

    xyz_min, xyz_max = np.amin([xyz_min, xyz_max], axis=0), np.amax([xyz_min, xyz_max], axis=0)

    images = [sorted(glob.glob(os.path.join(cam, '*.png')))[0] for cam in sorted(glob.glob(os.path.join(basedir, 'images_rgba_8x/cam_*')))]
    images = [np.array(Image.open(image).convert('RGB')) / 255. for image in images]
    images = np.stack(images, axis=0)

    K[:, 0, 0] /= factor
    K[:, 1, 1] /= factor
    K[:, 0, 2] /= factor
    K[:, 1, 2] /= factor
    hwf = [images.shape[1], images.shape[2], K[0][0][0]]

    render_poses = torch.Tensor(poses)

    i_train = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    i_val = [4]
    i_test = [4]
    i_split = [i_train, i_val, i_test]

    return images, poses, render_poses, hwf, i_split, K, xyz_min, xyz_max

def recenter_pose(poses, poses_center, xyz_min, xyz_max):
    bottom = np.reshape([0,0,0,1.], [1,4])
    poses_center = np.concatenate([poses_center[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)
    poses = np.linalg.inv(poses_center) @ poses
    xyz_min = np.linalg.inv(poses_center) @ np.concatenate([xyz_min, np.array([1])])
    xyz_max = np.linalg.inv(poses_center) @ np.concatenate([xyz_max, np.array([1])])
    return poses[:,:3,:4], xyz_min[:3], xyz_max[:3]

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m