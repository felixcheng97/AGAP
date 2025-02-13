import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
import glob
from PIL import Image
from scipy.optimize import least_squares


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose


def load_dtu_data(basedir, width, height, factor, x, y, z):    
    # images = sorted(glob.glob(os.path.join(basedir, 'image', '*.png')))[:49]
    # masks = sorted(glob.glob(os.path.join(basedir, 'mask', '*.png')))[:49]
    images = sorted(glob.glob(os.path.join(basedir, 'image', '*.png')))
    masks = sorted(glob.glob(os.path.join(basedir, 'mask', '*.png')))
    camera_dict = np.load(os.path.join(basedir, 'cameras.npz'))
    n_images = len(images)

    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    intrinsics_all = []
    pose_all = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(None, P)
        intrinsics_all.append(intrinsics)
        pose_all.append(pose)

    K = np.stack(intrinsics_all, axis=0)[:, :3, :3]
    K[:, 0, 0] /= factor
    K[:, 1, 1] /= factor
    K[:, 0, 2] /= factor
    K[:, 1, 2] /= factor
    poses = np.stack(pose_all, axis=0)

    opencv2opengl = lambda c2w: np.concatenate([np.concatenate([ c2w[..., :, 0:1], -c2w[..., :, 1:2], -c2w[..., :, 2:3]], axis=-1), c2w[..., :, 3:]], axis=-1)
    poses = opencv2opengl(poses)    

    masked_images = []
    # images_all = []
    masks_all = []
    for image, mask in zip(images, masks):
        image = np.array(Image.open(image).convert('RGB').resize((width, height), Image.LANCZOS)) / 255.
        mask = np.array(Image.open(mask).convert('RGB').resize((width, height), Image.LANCZOS)) / 255.
        # mask = np.array(Image.open(mask).resize((width, height), Image.LANCZOS).convert('L'))[..., None] > 127.5
        masked_image = image * mask
        masked_images.append(masked_image)
        # images_all.append(image)
        masks_all.append(np.max(mask, axis=2) > 0.5)
    images = np.stack(masked_images, axis=0)
    masks = np.stack(masks_all, axis=0)

    # Fit the sphere's center and radius
    # Ts = poses[:49, :3, -1]
    # import pdb; pdb.set_trace()
    # from . import utils; utils.plot_camera_poses('./', poses[:,:3,:4])
    # Ts = poses[:, :3, -1]
    # center, radius = fit_sphere(Ts)
    center = np.array([0., 0., 0.])
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    poses_center = viewmatrix(vec2, up, center)
    poses = recenter_pose(poses, poses_center)

    #     
    # import pdb; pdb.set_trace()
    center = np.array([x, y, z])
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    poses_center = viewmatrix(vec2, up, center)
    poses = recenter_pose(poses, poses_center)

    render_poses = torch.Tensor(poses)

    hwf = [height, width, K[0][0][0]]

    i_train = [i for i in range(n_images)]
    i_val = [23]
    i_test = [23]
    i_split = [i_train, i_val, i_test]
    
    return images, masks, poses, render_poses, hwf, i_split, K

def recenter_pose(poses, poses_center):
    bottom = np.reshape([0,0,0,1.], [1,4])
    poses_center = np.concatenate([poses_center[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)
    poses = np.linalg.inv(poses_center) @ poses
    return poses[:,:3,:4]

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def sphere_residuals(params, points):
    # Calculate the squared difference between the distances from each point to the center and the radius
    center = params[:3]
    radius = params[3]
    distances = np.linalg.norm(points - center, axis=1)
    residuals = distances - radius
    return residuals

def fit_sphere(points):
    # Initialize the fitting parameters
    initial_center = np.mean(points, axis=0)
    initial_radius = np.max(np.linalg.norm(points - initial_center, axis=1))
    initial_params = np.concatenate([initial_center, [initial_radius]])

    # Fit the center and radius using the least squares method
    result = least_squares(sphere_residuals, initial_params, args=(points,))
    center = result.x[:3]
    radius = result.x[3]
    return center, radius

