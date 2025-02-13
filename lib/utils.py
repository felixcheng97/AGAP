import os, math
import numpy as np
import scipy.signal
from typing import List, Optional

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from .masked_adam import MaskedAdam


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d


''' Misc
'''
mse2psnr = lambda x : -10. * torch.log10(x)
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def create_optimizer_or_freeze_model(model, cfg_train, global_step):
    decay_steps = cfg_train.lrate_decay * 1000
    decay_factor = 0.1 ** (global_step/decay_steps)

    param_group = []
    for k in cfg_train.keys():
        if not k.startswith('lrate_'):
            continue
        k = k[len('lrate_'):]

        if not hasattr(model, k):
            continue

        param = getattr(model, k)
        if param is None:
            print(f'create_optimizer_or_freeze_model: param {k} not exist')
            continue

        lr = getattr(cfg_train, f'lrate_{k}') * decay_factor
        if lr > 0:
            print(f'create_optimizer_or_freeze_model: param {k} lr {lr}')
            if isinstance(param, nn.Module):
                param = param.parameters()
            param_group.append({'params': param, 'lr': lr, 'skip_zero_grad': (k in cfg_train.skip_zero_grad_fields)})
        else:
            print(f'create_optimizer_or_freeze_model: param {k} freeze')
            param.requires_grad = False
    return MaskedAdam(param_group)


''' Checkpoint utils
'''
def load_checkpoint(model, optimizer, ckpt_path, no_reload_optimizer):
    ckpt = torch.load(ckpt_path)
    start = ckpt['global_step']
    model.load_state_dict(ckpt['model_state_dict'])
    if not no_reload_optimizer:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return model, optimizer, start


def load_model(model_class, ckpt_path):
    ckpt = torch.load(ckpt_path)
    model = model_class(**ckpt['model_kwargs'])
    model.load_state_dict(ckpt['model_state_dict'])
    return model


''' Evaluation metrics (ssim)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim



def plot_camera_poses(savedir, c2w, xyz_min, xyz_max):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract the camera positions (origins of the camera poses)
    camera_positions = c2w[:, :3, 3]
    max_val_x = np.max(np.abs(camera_positions[:, 0]))
    max_val_y = np.max(np.abs(camera_positions[:, 1]))
    max_val_z = np.max(np.abs(camera_positions[:, 2]))
    max_val = np.min([max_val_x, max_val_y, max_val_z])
    
    # Plot the global coordinate system
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', length=max_val)
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', length=max_val)
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', length=max_val)
    
    for pose in c2w:
        origin = pose[:3, 3]
        x_axis = pose[:3, 0]
        y_axis = pose[:3, 1]
        z_axis = pose[:3, 2]
        
        ax.quiver(*origin, *x_axis, color='r', length=max_val / 5)
        ax.quiver(*origin, *y_axis, color='g', length=max_val / 5)
        ax.quiver(*origin, *z_axis, color='b', length=max_val / 5)
    
    # Set labels and limits based on the maximum absolute value of the positions
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-max_val_x, max_val_x])
    ax.set_ylim([-max_val_y, max_val_y])
    ax.set_zlim([-max_val_z, max_val_z])

    # Draw the cube (edges only) indicated by the xyz_min and xyz_max
    corners = np.array([[xyz_min[0], xyz_min[1], xyz_min[2]],
                        [xyz_max[0], xyz_min[1], xyz_min[2]],
                        [xyz_max[0], xyz_max[1], xyz_min[2]],
                        [xyz_min[0], xyz_max[1], xyz_min[2]],
                        [xyz_min[0], xyz_min[1], xyz_max[2]],
                        [xyz_max[0], xyz_min[1], xyz_max[2]],
                        [xyz_max[0], xyz_max[1], xyz_max[2]],
                        [xyz_min[0], xyz_max[1], xyz_max[2]]])

    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]

    for edge in edges:
        p1, p2 = corners[edge]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-', linewidth=0.5)
    
    # Add a title with a message
    title_msg = f"Camera Poses Visualization (Max Values: X={max_val_x:.2f}, Y={max_val_y:.2f}, Z={max_val_z:.2f})"
    plt.title(title_msg)
    
    # Save the plot
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    plt.savefig(os.path.join(savedir, 'camera_poses.png'))


def create_line_set(c2w):
    points = []
    lines = []
    colors = []
    
    line_idx = 0
    for pose in c2w:
        origin = pose[:3, 3]
        x_axis = origin + pose[:3, 0] * 0.1
        y_axis = origin + pose[:3, 1] * 0.1
        z_axis = origin + pose[:3, 2] * 0.1
        
        points.append(origin)
        points.append(x_axis)
        points.append(origin)
        points.append(y_axis)
        points.append(origin)
        points.append(z_axis)
        
        lines.append([line_idx, line_idx + 1])
        lines.append([line_idx + 2, line_idx + 3])
        lines.append([line_idx + 4, line_idx + 5])
        
        colors.append([1, 0, 0])  # Red for x-axis
        colors.append([0, 1, 0])  # Green for y-axis
        colors.append([0, 0, 1])  # Blue for z-axis
        
        line_idx += 6
    
    return points, lines, colors

def save_as_ply(c2w, filename='camera_poses.ply'):
    points, lines, colors = create_line_set(c2w)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_line_set(filename, line_set)