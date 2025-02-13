import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import segment_coo

from . import grid
from .dvgo import Raw2Alpha, Alphas2Weights
from .dmpigo import create_full_step_id
from .networks import *

from torch.utils.cpp_extension import load
parent_dir = os.path.dirname(os.path.abspath(__file__))
ub360_utils_cuda = load(
        name='ub360_utils_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/ub360_utils.cpp', 'cuda/ub360_utils_kernel.cu']],
        verbose=True)


'''Model'''
class DirectPanoramaVoxGO(nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, num_voxels_base=0,
                 alpha_init=None,
                 mask_cache_world_size=None,
                 fast_color_thres=0,
                 contracted_norm='l2',
                 density_type='DenseGrid', k0_type='DenseGrid',
                 density_config={}, k0_config={},
                 rgbnet_dim=0,
                 rgbnet_depth=3, rgbnet_width=128,
                 equ_size=(768,1536),
                 xyz_config={},
                 viewdirs_config={},
                 deformation_config={},
                 **kwargs):
        super(DirectPanoramaVoxGO, self).__init__()
        self.register_buffer('xyz_min', torch.Tensor([-1,-1,-1]))
        self.register_buffer('xyz_max', torch.Tensor([1,1,1]))
        if isinstance(fast_color_thres, dict):
            self._fast_color_thres = fast_color_thres
            self.fast_color_thres = fast_color_thres[0]
        else:
            self._fast_color_thres = None
            self.fast_color_thres = fast_color_thres
        self.contracted_norm = contracted_norm

        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)

        # determine init grid resolution
        self._set_grid_resolution(num_voxels)
        self._set_equ_resolution(equ_size)

        # determine the density bias shift
        self.alpha_init = alpha_init
        self.register_buffer('act_shift', torch.FloatTensor([np.log(1/(1-alpha_init) - 1)]))
        print('dpvgo: set density bias shift to', self.act_shift)

        # init density voxel grid
        self.density_type = density_type
        self.density_config = density_config
        self.density = grid.create_grid(
            density_type, channels=1, world_size=self.world_size,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max,
            config=self.density_config)

        # init color representation
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
        }
        self.k0_type = k0_type
        self.k0_config = k0_config

        if rgbnet_dim == 0:
            self.k0_explicit_grid = grid.DenseEquExplicitGrid(channels=3, equ_size=self.equ_size)
            self.k0_explicit_mlp = None
        else:
            self.k0_explicit_grid = grid.DenseEquExplicitGrid(channels=rgbnet_dim, equ_size=self.equ_size)
            self.k0_explicit_mlp = nn.Sequential(
                nn.Linear(rgbnet_dim, rgbnet_width), nn.ReLU(inplace=True),
                *[
                    nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                    for _ in range(rgbnet_depth-2)
                ],
                nn.Linear(rgbnet_width, 3),
            )
            nn.init.constant_(self.k0_explicit_mlp[-1].bias, 0)
        self.k0_explicit = grid.DenseEquExplicit(explicit_grid=self.k0_explicit_grid, explicit_mlp=self.k0_explicit_mlp, sigmoid=self.k0_type=='DenseEquExplicit')
        self.k0 = self.k0_explicit

        self.xyz_config = xyz_config
        self.viewdirs_config = viewdirs_config
        self.deformation_config = deformation_config

        self.xyz_enc_type = xyz_config['enc_type']
        if self.xyz_enc_type == 'pe':
            self.embedding_xyz = PositionalEncoding(in_channels=3, **xyz_config[self.xyz_enc_type])
        elif self.xyz_enc_type == 'hash':
            self.embedding_xyz = HashEncoding(**xyz_config[self.xyz_enc_type])
        else:
            raise NotImplementedError

        self.viewdirs_enc_type = viewdirs_config['enc_type']
        if self.viewdirs_enc_type == 'pe':
            self.embedding_viewdirs = ViewdirEncoding(in_channels=3, **viewdirs_config[self.viewdirs_enc_type])
        elif self.viewdirs_enc_type == 'hash':
            self.embedding_viewdirs = HashEncoding(**viewdirs_config[self.viewdirs_enc_type])
        else:
            raise NotImplementedError

        self.deform_type = deformation_config['deform_type']
        in_channels = self.embedding_xyz.out_channels + self.embedding_viewdirs.out_channels
        if self.deform_type == 'mlp':
            self.deformation_field = DeformationMLP(in_channels=in_channels, **deformation_config[self.deform_type])
        else:
            self.deformation_field = DeformationTCNN(in_channels=in_channels, **deformation_config[self.deform_type])

        print('dpvgo: densitye grid', self.density)
        print('dpvgo: k0', self.k0)
        print('dpvgo: deformation field', self.deformation_field)
        print('dpvgo: embedding_xyz', self.embedding_xyz)
        print('dpvgo: embedding_viewdirs', self.embedding_viewdirs)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        # Re-implement as occupancy grid (2021/1/31)
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size
        mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)
        self.mask_cache = grid.MaskGrid(
            path=None, mask=mask,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max)

    def get_k0_grid_rgb(self):
        rgb = self.k0.get_current_equ()[0].permute(1,2,0).detach().cpu()
        rgb_np = np.uint8(rgb.numpy() * 255)
        return rgb_np

    def _set_equ_resolution(self, equ_size):
        self.equ_size = equ_size
        print('dpvgo equ_size      ', self.equ_size)

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.world_len = self.world_size[0].item()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('dpvgo voxel_size      ', self.voxel_size)
        print('dpvgo world_size      ', self.world_size)
        print('dpvgo voxel_size_base ', self.voxel_size_base)
        print('dpvgo voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'alpha_init': self.alpha_init,
            'voxel_size_ratio': self.voxel_size_ratio,
            'mask_cache_world_size': list(self.mask_cache.mask.shape),
            'fast_color_thres': self.fast_color_thres,
            'contracted_norm': self.contracted_norm,
            'density_type': self.density_type,
            'k0_type': self.k0_type,
            'density_config': self.density_config,
            'k0_config': self.k0_config,
            **self.rgbnet_kwargs,
            'equ_size': self.equ_size,
            'xyz_config': self.xyz_config,
            'viewdirs_config': self.viewdirs_config,
            'deformation_config': self.deformation_config,
        }
    
    @torch.no_grad()
    def scale_equ_grid(self, equ_size, upsample):
        print('dpvgo scale_equ_grid start')
        ori_equ_size = self.equ_size
        self._set_equ_resolution(equ_size)
        print('dpvgo scale_equ_grid scale equ_size from', ori_equ_size, 'to', self.equ_size)

        self.k0.scale_equ_grid(self.equ_size, upsample)
        print('dpvgo k0 scale_image_grid finish')

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('dpvgo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('dpvgo: scale_volume_grid scale world_size from', ori_world_size.tolist(), 'to', self.world_size.tolist())

        self.density.scale_volume_grid(self.world_size)

        if np.prod(self.world_size.tolist()) <= 256**3:
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
            ), -1)
            self_alpha = F.max_pool3d(self.activate_density(self.density.get_dense_grid()), kernel_size=3, padding=1, stride=1)[0,0]
            self.mask_cache = grid.MaskGrid(
                path=None, mask=self.mask_cache(self_grid_xyz) & (self_alpha>self.fast_color_thres),
                xyz_min=self.xyz_min, xyz_max=self.xyz_max)

        print('dpvgo: scale_volume_grid finish')

    @torch.no_grad()
    def update_occupancy_cache(self):
        ori_p = self.mask_cache.mask.float().mean().item()
        cache_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.mask_cache.mask.shape[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.mask_cache.mask.shape[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.mask_cache.mask.shape[2]),
        ), -1)
        cache_grid_density = self.density(cache_grid_xyz)[None,None]
        cache_grid_alpha = self.activate_density(cache_grid_density)
        cache_grid_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0,0]
        self.mask_cache.mask &= (cache_grid_alpha > self.fast_color_thres)
        new_p = self.mask_cache.mask.float().mean().item()
        print(f'dpvgo update mask_cache {ori_p:.4f} => {new_p:.4f}')

    def update_occupancy_cache_lt_nviews(self, rays_o_tr, rays_d_tr, imsz, render_kwargs, maskout_lt_nviews):
        print('dpvgo update mask_cache lt_nviews start')
        eps_time = time.time()
        count = torch.zeros_like(self.density.get_dense_grid()).long()
        device = count.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = grid.DenseGrid(1, self.world_size, self.xyz_min, self.xyz_max)
            for rays_o, rays_d in zip(rays_o_.split(8192), rays_d_.split(8192)):
                ray_pts, inner_mask, t = self.sample_ray(
                        ori_rays_o=rays_o.to(device), ori_rays_d=rays_d.to(device),
                        **render_kwargs)
                ones(ray_pts).sum().backward()
            count.data += (ones.grid.grad > 1)
        ori_p = self.mask_cache.mask.float().mean().item()
        self.mask_cache.mask &= (count >= maskout_lt_nviews)[0,0]
        new_p = self.mask_cache.mask.float().mean().item()
        print(f'dpvgo update mask_cache {ori_p:.4f} => {new_p:.4f}')
        eps_time = time.time() - eps_time
        print(f'dpvgo update mask_cache lt_nviews finish (eps time:', eps_time, 'sec)')

    def density_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.density.total_variation_add_grad(w, w, w, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        wx = weight * self.equ_size[1] / 128
        wy = weight * self.equ_size[0] / 128
        self.k0.total_variation_2d_add_grad(wx, wy, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), self.act_shift, interval).reshape(shape)

    def sample_ray(self, ori_rays_o, ori_rays_d, stepsize, is_train=False, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        rays_o = ori_rays_o
        rays_d = ori_rays_d
        N = int(self.world_len / stepsize)
        b1 = torch.linspace(0.1, 1, N//2+1)
        b2 = 1 / torch.linspace(1, 1/256, N//2+1)
        t = torch.cat([
            (b1[1:] + b1[:-1]) * 0.5,
            (b2[1:] + b2[:-1]) * 0.5,
        ])
        ray_pts = rays_o[:,None,:] + rays_d[:,None,:] * t[None,:,None]

        if self.contracted_norm == 'inf':
            norm = ray_pts.abs().amax(dim=-1, keepdim=True)
        elif self.contracted_norm == 'l2':
            norm = ray_pts.norm(dim=-1, keepdim=True)
        else:
            raise NotImplementedError

        ray_pts = ray_pts / norm * (1 - 1 / (norm + 1))

        return ray_pts, t

    def forward(self, rays_o, rays_d, viewdirs, rays_mask=None, global_step=None, is_train=False, **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'
        if isinstance(self._fast_color_thres, dict) and global_step in self._fast_color_thres:
            print(f'dpvgo update fast_color_thres {self.fast_color_thres} => {self._fast_color_thres[global_step]}')
            self.fast_color_thres = self._fast_color_thres[global_step]

        ret_dict = {}
        N = len(rays_o)

        # sample points on rays
        ray_pts, t = self.sample_ray(
                ori_rays_o=rays_o, ori_rays_d=rays_d, is_train=global_step is not None, **render_kwargs)
        n_max = len(t)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio
        ray_id, step_id = create_full_step_id(ray_pts.shape[:2])

        # skip known free space
        mask = self.mask_cache(ray_pts)
        ray_pts = ray_pts[mask]
        t = t[None].expand(N,n_max)[mask]
        ray_id = ray_id[mask.flatten()]
        step_id = step_id[mask.flatten()]

        # query for alpha w/ post-activation
        density = self.density(ray_pts)
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            ray_pts = ray_pts[mask]
            t = t[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]

        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            ray_pts = ray_pts[mask]
            t = t[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]
            weights = weights[mask]

        # query for xy of deformation field
        if self.xyz_enc_type == 'pe':
            ray_pts_emb = self.embedding_xyz(ray_pts, global_step)
        elif self.xyz_enc_type == 'hash':
            ray_pts_norm = (ray_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
            ray_pts_emb = self.embedding_xyz(ray_pts_norm, global_step)
        else:
            raise NotImplementedError

        if self.viewdirs_enc_type == 'pe':
            viewdirs_emb = self.embedding_viewdirs(viewdirs[ray_id], global_step)
        elif self.viewdirs_enc_type == 'hash':
            raise NotImplementedError
        else:
            raise NotImplementedError

        deformation_input = torch.cat([ray_pts_emb, viewdirs_emb], dim=-1)
        dudv = self.deformation_field(deformation_input)

        # query for color
        rgb = self.k0(ray_pts, dudv)

        if global_step is not None and global_step % 500 == 0:
            if dudv is not None:
                print('dudv:', dudv.amin(dim=0), dudv.amax(dim=0))

        # Ray marching            
        rgb_marched = segment_coo(
                src=(weights.unsqueeze(-1) * rgb),
                index=ray_id,
                out=torch.zeros([N, 3]),
                reduce='sum')
        if render_kwargs.get('rand_bkgd', False) and is_train:
            rgb_marched += (alphainv_last.unsqueeze(-1) * torch.rand_like(rgb_marched))
        else:
            rgb_marched += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'])
        
        s = 1 - 1/(1+t)  # [0, inf] => [0, 1]
        ret_dict.update({
            'alphainv_last': alphainv_last,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_density': density,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'ray_id': ray_id,
            'step_id': step_id,
            'n_max': n_max,
            't': t,
            's': s,
            'dudv': dudv,
        })

        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth = segment_coo(
                        src=(weights * s),
                        index=ray_id,
                        out=torch.zeros([N]),
                        reduce='sum')
            ret_dict.update({'depth': depth})

        return ret_dict


class DistortionLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, s, n_max, ray_id):
        n_rays = ray_id.max()+1
        interval = 1/n_max
        w_prefix, w_total, ws_prefix, ws_total = ub360_utils_cuda.segment_cumsum(w, s, ray_id)
        loss_uni = (1/3) * interval * w.pow(2)
        loss_bi = 2 * w * (s * w_prefix - ws_prefix)
        ctx.save_for_backward(w, s, w_prefix, w_total, ws_prefix, ws_total, ray_id)
        ctx.interval = interval
        return (loss_bi.sum() + loss_uni.sum()) / n_rays

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        w, s, w_prefix, w_total, ws_prefix, ws_total, ray_id = ctx.saved_tensors
        interval = ctx.interval
        grad_uni = (1/3) * interval * 2 * w
        w_suffix = w_total[ray_id] - (w_prefix + w)
        ws_suffix = ws_total[ray_id] - (ws_prefix + w*s)
        grad_bi = 2 * (s * (w_prefix - w_suffix) + (ws_suffix - ws_prefix))
        grad = grad_back * (grad_bi + grad_uni)
        return grad, None, None, None

distortion_loss = DistortionLoss.apply

