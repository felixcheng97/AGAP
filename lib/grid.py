import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import itertools
from .networks import *

from torch.utils.cpp_extension import load
parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
        name='render_utils_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
        verbose=True)

total_variation_cuda = load(
        name='total_variation_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/total_variation.cpp', 'cuda/total_variation_kernel.cu']],
        verbose=True)

total_variation_2d_cuda = load(
        name='total_variation_2d_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/total_variation_2d.cpp', 'cuda/total_variation_2d_kernel.cu']],
        verbose=True)


def create_grid(type, **kwargs):
    if type == 'DenseGrid':
        return DenseGrid(**kwargs)
    else:
        raise NotImplementedError


''' Dense MSI explicit
'''
class DenseMSIExplicit(nn.Module):
    def __init__(self, explicit_grid, explicit_mlp, uv_min, uv_max, sigmoid=True, **kwargs):
        super(DenseMSIExplicit, self).__init__()
        self.explicit_grid = explicit_grid
        self.explicit_mlp = explicit_mlp
        self.register_buffer('uv_min', torch.tensor(uv_min))
        self.register_buffer('uv_max', torch.tensor(uv_max))
        self.msi_size = explicit_grid.msi_size
        self.sigmoid = sigmoid
        self.inference = False
        self.loaded = False
    
    def forward(self, xyz, dudv=None, rays_mask=None, ray_id=None):
        # canonical projection initialization (P_c)
        xyz = xyz / xyz.norm(dim=-1, keepdim=True)
        xyz = xyz.clamp(-1., 1.)
        # xyz[xyz[..., 2] < 0] *= -1
        x = xyz[..., 0]
        y = xyz[..., 1]
        z = xyz[..., 2]
        phi = torch.asin(y)
        theta = torch.atan2(x, z)
        uv = torch.stack([theta, phi], dim=-1)

        # projection offset (P_o)
        if dudv is not None:
            uv = uv + dudv

        # circular rounding for phi
        # uv[:, 0] = torch.remainder(uv[:, 0] - self.uv_min[0], self.uv_max[0] - self.uv_min[0]) + self.uv_min[0]

        # normalize uv to range [-1, 1]
        ind_norm = ((uv.contiguous() - self.uv_min) / (self.uv_max - self.uv_min)) * 2 - 1
        
        # if rays_mask is not None:
        #     bg_mask = rays_mask[ray_id] == 0
        #     detached_part = ind_norm[bg_mask].detach()
        #     new_ind_norm = ind_norm.clone()
        #     new_ind_norm[bg_mask] = detached_part
        #     ind_norm = new_ind_norm.reshape(1,1,-1,2)
        # else:
        #     ind_norm = ind_norm.reshape(1,1,-1,2)
        ind_norm = ind_norm.reshape(1,1,-1,2)

        # get the grid for grid sample
        if not self.loaded:
            grid = self.explicit_grid(uv, self.explicit_mlp)
        else:
            grid = self.grid_loaded

        # perform grid sample
        out = F.grid_sample(grid, ind_norm, mode='bicubic', align_corners=True)
        out = out.reshape(3, -1).T

        # sigmoid for RGB
        if not self.loaded and self.sigmoid:
            out = F.sigmoid(out)

        return out

    def set_inference(self, inference):
        self.inference = inference

    def get_current_msi(self):
        if self.explicit_mlp:
            raise NotImplementedError
        else:
            rgb_logit = self.explicit_grid.grid
        return F.sigmoid(rgb_logit) if self.sigmoid else rgb_logit
        
    def load(self, msi_path):
        device = next(self.explicit_grid.parameters()).device
        img = np.array(Image.open(msi_path))[..., :3]
        img = torch.from_numpy(img).permute(2, 0, 1).flip((1,)).unsqueeze(0).to(device) / 255.
        img = F.interpolate(img, size=self.msi_size, mode='bilinear', align_corners=True)
        self.grid_loaded = img
        self.loaded = True

    def scale_msi_grid(self, new_msi_size, upsample=False):
        self.explicit_grid.scale_msi_grid(new_msi_size, upsample)
        self.msi_size = new_msi_size

    def total_variation_2d_add_grad(self, wx, wy, dense_mode):
        self.explicit_grid.total_variation_2d_add_grad(wx, wy, dense_mode)

    def get_dense_grid(self):
        return self.get_current_msi()

    @torch.no_grad()
    def __isub__(self, val):
        self.grid.data -= val
        return self

    def extra_repr(self):
        return f'inference={self.inference}, loaded={self.loaded}, sigmoid={self.sigmoid}'
    

''' Dense MSI explicit grid
'''
class DenseMSIExplicitGrid(nn.Module):
    def __init__(self, channels, msi_size, **kwargs):
        super(DenseMSIExplicitGrid, self).__init__()
        self.channels = channels
        self.msi_size = msi_size
        self.grid = nn.Parameter(torch.zeros([1, channels, *msi_size]))

    def forward(self, ind_norm, rgbnet=None):
        '''
        xyz: global coordinates to query
        '''
        if rgbnet:
            raise NotImplementedError
        else:
            grid = self.grid
        return grid

    def scale_msi_grid(self, new_msi_size, upsample=False):
        if not upsample:
            self.grid = nn.Parameter(self.grid.new_zeros((1, self.channels, *new_msi_size)))
        else:
            self.grid = nn.Parameter(F.interpolate(self.grid.data, size=tuple(new_msi_size), mode='bicubic', align_corners=True))
        self.msi_size = new_msi_size

    def total_variation_2d_add_grad(self, wx, wy, dense_mode):
        '''Add gradients by total variation loss in-place'''
        total_variation_2d_cuda.total_variation_2d_add_grad(
            self.grid, self.grid.grad, wx, wy, dense_mode)

    def get_dense_grid(self):
        return self.grid

    @torch.no_grad()
    def __isub__(self, val):
        self.grid.data -= val
        return self

    def extra_repr(self):
        return f'channels={self.channels}, msi_size={self.msi_size}'


''' Dense Equ explicit
'''
class DenseEquExplicit(nn.Module):
    def __init__(self, explicit_grid, explicit_mlp, sigmoid=True, **kwargs):
        super(DenseEquExplicit, self).__init__()
        self.explicit_grid = explicit_grid
        self.explicit_mlp = explicit_mlp
        self.register_buffer('uv_min', torch.Tensor([-np.pi, -np.pi/2]))
        self.register_buffer('uv_max', torch.Tensor([ np.pi,  np.pi/2]))
        self.equ_size = explicit_grid.equ_size
        self.padding_size = explicit_grid.padding_size
        
        self.calculate_inv_kb()
        self.sigmoid = sigmoid
        self.inference = False
        self.loaded = False
    
    def calculate_inv_kb(self):
        # left point : (x, y) = (uv_min[0] - (inv * padding_size - inv/2), -1)
        # right point: (x, y) = (uv_max[0] + (inv * padding_size - inv/2), +1)
        # solve the following two equations:
        ### -1 = k * (uv_min[0] - (inv * padding_size - inv/2)) + b
        ### +1 = k * (uv_max[0] + (inv * padding_size - inv/2)) + b
        self.inv = (self.uv_max[0] - self.uv_min[0]) / self.equ_size[1]
        self.k = torch.Tensor([2 / ((self.uv_max[0] - self.uv_min[0]) + 2 * (self.inv * self.padding_size - self.inv/2)), 2 / (self.uv_max[1] - self.uv_min[1])])
        self.b = torch.Tensor([-0.5 * (self.uv_max[0] + self.uv_min[0]) * self.k[0], -(self.uv_max[1] + self.uv_min[1]) / (self.uv_max[1] - self.uv_min[1])])

    def forward(self, xyz, dudv=None):
        # canonical projection initialization (P_c)
        xyz = xyz / xyz.norm(dim=-1, keepdim=True)
        xyz = xyz.clamp(-1., 1.)
        x = xyz[..., 0]
        y = xyz[..., 1]
        z = xyz[..., 2]
        phi = torch.asin(z)
        theta = torch.atan2(y, x)
        uv = torch.stack([theta, phi], dim=-1)

        # projection offset (P_o)
        if dudv is not None:
            uv = uv + dudv

        # circular rounding for phi     
        uv[:, 0] = torch.remainder(uv[:, 0] - self.uv_min[0], self.uv_max[0] - self.uv_min[0]) + self.uv_min[0]

        # normalize phi and theta to range [-1, 1]
        ind_norm = self.k * uv + self.b
        ind_norm = ind_norm.reshape(1,1,-1,2)

        # get the grid for grid sample
        if not self.loaded:
            grid = self.explicit_grid(ind_norm, self.explicit_mlp)
        else:
            grid = self.grid_loaded

        # perform grid sample
        out = F.grid_sample(grid, ind_norm, mode='bicubic', align_corners=True)
        out = out.reshape(3, -1).T

        # sigmoid for RGB
        if not self.loaded and self.sigmoid:
            out = F.sigmoid(out)

        return out

    def set_inference(self, inference):
        self.inference = inference

    def get_current_equ(self):
        if self.explicit_mlp:
            raise NotImplementedError
        else:
            rgb_logit = self.explicit_grid.grid
        return F.sigmoid(rgb_logit) if self.sigmoid else rgb_logit
        
    def load(self, equ_path):
        device = next(self.explicit_grid.parameters()).device
        img = np.array(Image.open(equ_path))
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device) / 255.
        img = F.interpolate(img, size=self.equ_size, mode='bilinear', align_corners=True)
        self.grid_loaded = img
        self.loaded = True

    def scale_equ_grid(self, new_equ_size, upsample=False):
        self.explicit_grid.scale_equ_grid(new_equ_size, upsample)
        self.equ_size = new_equ_size
        self.calculate_inv_kb()

    def total_variation_2d_add_grad(self, wx, wy, dense_mode):
        self.explicit_grid.total_variation_2d_add_grad(wx, wy, dense_mode)

    def get_dense_grid(self):
        return self.get_current_equ()

    @torch.no_grad()
    def __isub__(self, val):
        self.grid.data -= val
        return self

    def extra_repr(self):
        return f'inference={self.inference}, loaded={self.loaded}, sigmoid={self.sigmoid}'


''' Dense Equ explicit grid
'''
class DenseEquExplicitGrid(nn.Module):
    def __init__(self, channels, equ_size, **kwargs):
        super(DenseEquExplicitGrid, self).__init__()
        self.channels = channels
        self.equ_size = equ_size
        self.padding_size = 16
        self.grid = nn.Parameter(torch.zeros([1, channels, *equ_size]))

    def forward(self, ind_norm, rgbnet=None):
        '''
        xyz: global coordinates to query
        '''
        if rgbnet:
            raise NotImplementedError
        else:
            grid = self.grid
        grid = self.circular_pad(grid)
        return grid

    def scale_equ_grid(self, new_equ_size, upsample=False):
        if not upsample:
            self.grid = nn.Parameter(self.grid.new_zeros((1, self.channels, *new_equ_size)))
        else:
            self.grid = nn.Parameter(F.interpolate(self.grid.data, size=tuple(new_equ_size), mode='bicubic', align_corners=True))
        self.equ_size = new_equ_size
    
    def circular_pad(self, grid):
        assert grid.shape[-1] > self.padding_size
        return torch.cat([grid[..., -self.padding_size:], grid, grid[..., :self.padding_size]], dim=-1)

    def total_variation_2d_add_grad(self, wx, wy, dense_mode):
        '''Add gradients by total variation loss in-place'''
        total_variation_2d_cuda.total_variation_2d_add_grad(
            self.grid, self.grid.grad, wx, wy, dense_mode)

    def get_dense_grid(self):
        return self.grid

    @torch.no_grad()
    def __isub__(self, val):
        self.grid.data -= val
        return self

    def extra_repr(self):
        return f'channels={self.channels}, equ_size={self.equ_size}'


''' Dense 2D explicit
'''
class Dense2DExplicit(nn.Module):
    def __init__(self, explicit_grid, explicit_mlp, uv_min, uv_max, sigmoid=True, **kwargs):
        super(Dense2DExplicit, self).__init__()
        self.explicit_grid = explicit_grid
        self.explicit_mlp = explicit_mlp
        self.register_buffer('uv_min', torch.tensor(uv_min))
        self.register_buffer('uv_max', torch.tensor(uv_max))
        self.image_size = explicit_grid.image_size
        self.sigmoid = sigmoid
        self.inference = False
        self.loaded = False

    def forward(self, xyz, dudv=None):
        # canonical projection initialization (P_c)
        uv = xyz[..., :2]

        # projection offset (P_o)
        if dudv is not None:
            uv = uv + dudv

        # normalize phi and theta to range [-1, 1]
        ind_norm = ((uv.contiguous() - self.uv_min) / (self.uv_max - self.uv_min)) * 2 - 1
        ind_norm = ind_norm.reshape(1,1,-1,2)

        # get the grid for grid sample
        if not self.loaded:
            grid = self.explicit_grid(uv, self.explicit_mlp)
        else:
            grid = self.grid_loaded

        # perform grid sample
        out = F.grid_sample(grid, ind_norm, mode='bicubic', align_corners=True)
        out = out.reshape(3, -1).T

        # sigmoid for RGB
        if not self.loaded and self.sigmoid:
            out = F.sigmoid(out)

        return out

    def set_inference(self, inference):
        self.inference = inference

    def get_current_image(self):
        if self.explicit_mlp:
            raise NotImplementedError
        else:
            rgb_logit = self.explicit_grid.grid
        return F.sigmoid(rgb_logit) if self.sigmoid else rgb_logit
        
    def load(self, img_path):
        device = next(self.explicit_grid.parameters()).device
        img = np.array(Image.open(img_path))[..., :3]
        img = torch.from_numpy(img).permute(2, 0, 1).flip((1,)).unsqueeze(0).to(device) / 255.
        img = F.interpolate(img, size=self.image_size, mode='bilinear', align_corners=True)
        self.grid_loaded = img
        self.loaded = True

    def scale_image_grid(self, new_image_size, upsample=False):
        self.explicit_grid.scale_image_grid(new_image_size, upsample)
        self.image_size = new_image_size

    def total_variation_2d_add_grad(self, wx, wy, dense_mode):
        self.explicit_grid.total_variation_2d_add_grad(wx, wy, dense_mode)

    def get_dense_grid(self):
        return self.get_current_image()

    @torch.no_grad()
    def __isub__(self, val):
        self.grid.data -= val
        return self

    def extra_repr(self):
        return f'inference={self.inference}, loaded={self.loaded}, sigmoid={self.sigmoid}'


''' Dense 2D explicit grid
'''
class Dense2DExplicitGrid(nn.Module):
    def __init__(self, channels, image_size, **kwargs):
        super(Dense2DExplicitGrid, self).__init__()
        self.channels = channels
        self.image_size = image_size
        self.grid = nn.Parameter(torch.zeros([1, channels, *image_size]))

    def forward(self, uv, rgbnet=None):
        '''
        xyz: global coordinates to query
        '''
        if rgbnet:
            grid = rgbnet(self.grid[0].reshape(self.channels, -1).permute(1, 0)).permute(1, 0).reshape(1, 3, *self.image_size)
        else:
            grid = self.grid
        return grid

    def scale_image_grid(self, new_image_size, upsample=False):
        if not upsample:
            self.grid = nn.Parameter(self.grid.new_zeros((1, self.channels, *new_image_size)))
        else:
            self.grid = nn.Parameter(F.interpolate(self.grid.data, size=tuple(new_image_size), mode='bicubic', align_corners=True))
        self.image_size = new_image_size

    def total_variation_2d_add_grad(self, wx, wy, dense_mode):
        '''Add gradients by total variation loss in-place'''
        total_variation_2d_cuda.total_variation_2d_add_grad(
            self.grid, self.grid.grad, wx, wy, dense_mode)

    def get_dense_grid(self):
        return self.grid

    @torch.no_grad()
    def __isub__(self, val):
        self.grid.data -= val
        return self

    def extra_repr(self):
        return f'channels={self.channels}, image_size={self.image_size}'


''' Dense 3D grid
'''
class DenseGrid(nn.Module):
    def __init__(self, channels, world_size, xyz_min, xyz_max, **kwargs):
        super(DenseGrid, self).__init__()
        self.channels = channels
        self.world_size = world_size
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.grid = nn.Parameter(torch.zeros([1, channels, *world_size]))

    def forward(self, xyz):
        '''
        xyz: global coordinates to query
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        out = F.grid_sample(self.grid, ind_norm, mode='bilinear', align_corners=True)
        out = out.reshape(self.channels,-1).T.reshape(*shape,self.channels)
        if self.channels == 1:
            out = out.squeeze(-1)
        return out

    def scale_volume_grid(self, new_world_size):
        if self.channels == 0:
            self.grid = nn.Parameter(torch.zeros([1, self.channels, *new_world_size]))
        else:
            self.grid = nn.Parameter(
                F.interpolate(self.grid.data, size=tuple(new_world_size), mode='trilinear', align_corners=True))

    def total_variation_add_grad(self, wx, wy, wz, dense_mode):
        '''Add gradients by total variation loss in-place'''
        total_variation_cuda.total_variation_add_grad(
            self.grid, self.grid.grad, wx, wy, wz, dense_mode)

    def get_dense_grid(self):
        return self.grid

    @torch.no_grad()
    def __isub__(self, val):
        self.grid.data -= val
        return self

    def extra_repr(self):
        return f'channels={self.channels}, world_size={self.world_size.tolist()}'


''' Mask grid
It supports query for the known free space and unknown space.
'''
class MaskGrid(nn.Module):
    def __init__(self, path=None, mask_cache_thres=None, mask=None, xyz_min=None, xyz_max=None):
        super(MaskGrid, self).__init__()
        if path is not None:
            st = torch.load(path)
            self.mask_cache_thres = mask_cache_thres
            density = F.max_pool3d(st['model_state_dict']['density.grid'], kernel_size=3, padding=1, stride=1)
            alpha = 1 - torch.exp(-F.softplus(density + st['model_state_dict']['act_shift']) * st['model_kwargs']['voxel_size_ratio'])
            mask = (alpha >= self.mask_cache_thres).squeeze(0).squeeze(0)
            xyz_min = torch.Tensor(st['model_kwargs']['xyz_min'])
            xyz_max = torch.Tensor(st['model_kwargs']['xyz_max'])
        else:
            mask = mask.bool()
            xyz_min = torch.Tensor(xyz_min)
            xyz_max = torch.Tensor(xyz_max)

        self.register_buffer('mask', mask)
        xyz_len = xyz_max - xyz_min
        self.register_buffer('xyz2ijk_scale', (torch.Tensor(list(mask.shape)) - 1) / xyz_len)
        self.register_buffer('xyz2ijk_shift', -xyz_min * self.xyz2ijk_scale)

    @torch.no_grad()
    def forward(self, xyz):
        '''Skip know freespace
        @xyz:   [..., 3] the xyz in global coordinate.
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(-1, 3)
        mask = render_utils_cuda.maskcache_lookup(self.mask, xyz, self.xyz2ijk_scale, self.xyz2ijk_shift)
        mask = mask.reshape(shape)
        return mask

    def extra_repr(self):
        return f'mask.shape=list(self.mask.shape)'

