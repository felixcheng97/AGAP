import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange

import math
import mmcv
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils, dvgo, dmpigo, dpvgo, dmsigo
from lib.load_data import load_data

from torch_efficient_distloss import flatten_eff_distloss

from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime


def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_image", action='store_true')
    parser.add_argument("--render_depth", action='store_true')
    parser.add_argument("--render_video_flipy", action='store_true')
    parser.add_argument("--render_video_rot90", default=0, type=int)
    parser.add_argument("--render_video_factor", type=float, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--dump_images", action='store_true')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--edit", type=str, default='', help='filename of edited k0_xxx.png')
    parser.add_argument("--render_panorama", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')
    return parser


@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, dump_images=False,
                      render_factor=0, render_video_flipy=False, render_video_rot90=0,
                      eval_ssim=False, render_panorama=False, log_metrics=False, dump_depths=False):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    rgbs = []
    depths = []
    bgmaps = []
    psnrs = []
    ssims = []

    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        K = Ks[i]
        c2w = torch.Tensor(c2w)
        if not render_panorama:
            rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                    H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        else:
            rays_o, rays_d, viewdirs = dvgo.get_ray_of_a_panorama(
                H, W, c2w
            )       

        keys = ['rgb_marched', 'depth', 'alphainv_last']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()
        bgmap = render_result['alphainv_last'].cpu().numpy()

        rgbs.append(rgb)
        depths.append(depth)
        bgmaps.append(bgmap)
        if i==0:
            print('Testing', rgb.shape)

        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))

    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if log_metrics:
            f = open(os.path.join(savedir, 'metrics.txt'), "w")
            f.write('PSNR: {:.6f}'.format(np.mean(psnrs)))
            if eval_ssim: f.write('SSIM: {:.6f}'.format(np.mean(ssims)))
            f.close()

    if render_video_flipy:
        for i in range(len(rgbs)):
            rgbs[i] = np.flip(rgbs[i], axis=0)
            depths[i] = np.flip(depths[i], axis=0)
            bgmaps[i] = np.flip(bgmaps[i], axis=0)

    if render_video_rot90 != 0:
        for i in range(len(rgbs)):
            rgbs[i] = np.rot90(rgbs[i], k=render_video_rot90, axes=(0,1))
            depths[i] = np.rot90(depths[i], k=render_video_rot90, axes=(0,1))
            bgmaps[i] = np.rot90(bgmaps[i], k=render_video_rot90, axes=(0,1))

    if savedir is not None and dump_images:
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    if savedir is not None and dump_depths:
        for i in trange(len(depths)):
            filename = os.path.join(savedir, '{:03d}.npy'.format(i))
            np.save(filename, depths[i])
            depth_vis = depths[i] * (1-bgmaps[i]) + bgmaps[i]
            dmin, dmax = np.percentile(depth_vis[bgmaps[i] < 0.1], q=[5, 95])
            depth_vis = plt.get_cmap('rainbow')(1 - np.clip((depth_vis - dmin) / (dmax - dmin), 0, 1)).squeeze()[..., :3]
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            Image.fromarray(np.uint8(depth_vis * 255)).save(filename)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    bgmaps = np.array(bgmaps)

    return rgbs, depths, bgmaps


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data)

    # remove useless field
    kept_keys = {
            'hwf', 'HW', 'Ks', 'Ks_render', 'near', 'far', 'near_clip',
            'i_train', 'i_val', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'images', 'masks', 'xyz_min', 'xyz_max'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    if data_dict['masks'] is not None:
        data_dict['masks'] = torch.Tensor(data_dict['masks'])
    return data_dict


def _compute_bbox_by_cam_frustrm_bounded(cfg, HW, Ks, poses, i_train, near, far):
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        if cfg.data.ndc:
            pts_nf = torch.stack([rays_o+rays_d*near, rays_o+rays_d*far])
        else:
            pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    return xyz_min, xyz_max


def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    if cfg.data.panorama:
        xyz_min, xyz_max = -torch.tensor([far, far, far]).float(), torch.tensor([far, far, far]).float()
    else:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_bounded(
                cfg, HW, Ks, poses, i_train, near, far)
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max


def create_new_model(cfg, cfg_model, cfg_train, xyz_min, xyz_max, uv_min, uv_max, stage, coarse_ckpt_path):
    model_kwargs = copy.deepcopy(cfg_model)
    num_voxels = model_kwargs.pop('num_voxels')
    if len(cfg_train.pg_scale):
        num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))

    if cfg.fine_model_and_render.image_size:
        image_size = model_kwargs.pop('image_size')
        if len(cfg_train.pg_image_scale):
            image_size = (image_size[0] // (2**len(cfg_train.pg_image_scale)), image_size[1] // (2**len(cfg_train.pg_image_scale)))
        model_kwargs['image_size'] = image_size

    if cfg.fine_model_and_render.equ_size:
        equ_size = model_kwargs.pop('equ_size')
        if len(cfg_train.pg_equ_scale):
            equ_size = (equ_size[0] // (2**len(cfg_train.pg_equ_scale)), equ_size[1] // (2**len(cfg_train.pg_equ_scale)))
        model_kwargs['equ_size'] = equ_size        

    if cfg.fine_model_and_render.msi_size:
        msi_size = model_kwargs.pop('msi_size')
        if len(cfg_train.pg_msi_scale):
            msi_size = (msi_size[0] // (2**len(cfg_train.pg_msi_scale)), msi_size[1] // (2**len(cfg_train.pg_msi_scale)))
        model_kwargs['msi_size'] = msi_size    

    # if cfg.data.ndc:
    if cfg.fine_model_and_render.model_type == 'DirectMPIGO':
        print(f'scene_rep_reconstruction ({stage}): \033[96muse multiplane images\033[0m')
        model = dmpigo.DirectMPIGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            **model_kwargs)
    elif cfg.fine_model_and_render.model_type == 'DirectPanoramaVoxGO':
        model = dpvgo.DirectPanoramaVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            **model_kwargs)
    elif cfg.fine_model_and_render.model_type == 'DirectMSIGO':
        model = dmsigo.DirectMSIGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            uv_min=uv_min, uv_max=uv_max, 
            num_voxels=num_voxels,
            **model_kwargs)
    else:
        # print(f'scene_rep_reconstruction ({stage}): \033[96muse dense voxel grid\033[0m')
        # model = dvgo.DirectVoxGO(
        #     xyz_min=xyz_min, xyz_max=xyz_max,
        #     num_voxels=num_voxels,
        #     mask_cache_path=coarse_ckpt_path,
        #     **model_kwargs)
        raise NotImplementedError
    model = model.to(device)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    return model, optimizer


def load_existed_model(args, cfg, cfg_train, reload_ckpt_path):
    if cfg.fine_model_and_render.model_type == 'DirectMPIGO':
        model_class = dmpigo.DirectMPIGO
    elif cfg.fine_model_and_render.model_type == 'DirectPanoramaVoxGO':
        model_class = dpvgo.DirectPanoramaVoxGO
    elif cfg.fine_model_and_render.model_type == 'DirectMSIGO':
        model_class = dmsigo.DirectMSIGO
    else:
        # model_class = dvgo.DirectVoxGO
        raise NotImplementedError
    model = utils.load_model(model_class, reload_ckpt_path).to(device)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    model, optimizer, start = utils.load_checkpoint(
            model, optimizer, reload_ckpt_path, args.no_reload_optimizer)
    return model, optimizer, start


def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, uv_min, uv_max, data_dict, stage, coarse_ckpt_path=None):
    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
    HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images, masks = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images', 'masks'
        ]
    ]

    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last.tar')
    if args.no_reload:
        reload_ckpt_path = None
    elif args.ft_path:
        reload_ckpt_path = args.ft_path
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None

    # update image_size according to xyz_min and xyz_max
    if len(cfg.fine_model_and_render.image_size) == 1:
        H = cfg.fine_model_and_render.image_size[0]
        ratio = ((xyz_max - xyz_min)[0] / (xyz_max - xyz_min)[1]).item() * HW[0][1] / HW[0][0]
        W = int(round((H * ratio) / 64) * 64)
        cfg.fine_model_and_render.image_size = (H, W)
        print('update image_size to', cfg.fine_model_and_render.image_size)

    if len(cfg.fine_model_and_render.msi_size) == 1:
        H = cfg.fine_model_and_render.msi_size[0]
        ratio = ((uv_max - uv_min)[0] / (uv_max - uv_min)[1]).item()
        W = int(round((H * ratio) / 64) * 64)
        cfg.fine_model_and_render.msi_size = (H, W)
        print('update image_size to', cfg.fine_model_and_render.msi_size)

    # init model and optimizer
    if reload_ckpt_path is None:
        print(f'scene_rep_reconstruction ({stage}): train from scratch')
        model, optimizer = create_new_model(cfg, cfg_model, cfg_train, xyz_min, xyz_max, uv_min, uv_max, stage, coarse_ckpt_path)
        start = 0
        if cfg_model.maskout_near_cam_vox:
            model.maskout_near_cam_vox(poses[i_train,:3,3], near)
    else:
        print(f'scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}')
        model, optimizer, start = load_existed_model(args, cfg, cfg_train, reload_ckpt_path)

    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'rand_bkgd': cfg.data.rand_bkgd,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }

    # init batch rays sampler
    def gather_training_rays():
        if data_dict['irregular_shape']:
            rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
        else:
            rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)
        if masks is not None:
            mask_tr_ori = masks[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)
        else:
            mask_tr_ori = [None for _ in i_train]

        if cfg_train.ray_sampler == 'flatten':
            rgb_tr, mask_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori,
                mask_tr_ori=mask_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        elif cfg_train.ray_sampler == 'panorama_uniform':
            mask_tr = None
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_panorama(
                rgb_tr=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train],
            )
        elif cfg_train.ray_sampler == 'random':
            rgb_tr, mask_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays(
                rgb_tr=rgb_tr_ori,
                mask_tr=mask_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        else:
            raise NotImplementedError
            
        return rgb_tr, mask_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz

    rgb_tr, mask_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = gather_training_rays()

    index_generator = dvgo.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
    batch_index_sampler = lambda: next(index_generator)

    # view-count-based learning rate
    if cfg_train.pervoxel_lr:
        def per_voxel_init():
            cnt = model.voxel_count_views(
                    rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz, near=near, far=far,
                    stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
                    irregular_shape=data_dict['irregular_shape'])
            optimizer.set_pervoxel_lr(cnt)
            model.mask_cache.mask[cnt.squeeze() <= 2] = False
        per_voxel_init()

    if cfg_train.maskout_lt_nviews > 0:
        model.update_occupancy_cache_lt_nviews(
                rays_o_tr, rays_d_tr, imsz, render_kwargs, cfg_train.maskout_lt_nviews)

    # GOGO
    ###############
    images_dir = os.path.join(cfg.basedir, cfg.expname, 'images')
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    ###############

    torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1
    for global_step in trange(1+start, 1+cfg_train.N_iters):

        # renew occupancy grid
        if model.mask_cache is not None and (global_step + 500) % 1000 == 0:
            model.update_occupancy_cache()

        # progress scaling checkpoint
        if global_step in cfg_train.pg_image_scale:
            n_rest_image_scales = len(cfg_train.pg_image_scale)-cfg_train.pg_image_scale.index(global_step)-1
            cur_image_size = (cfg_model.image_size[0] // (2**n_rest_image_scales), cfg_model.image_size[1] // (2**n_rest_image_scales))
            upsample = global_step > cfg_train.pg_upsample_after
            if isinstance(model, dmpigo.DirectMPIGO):
                model.scale_image_grid(cur_image_size, upsample)
            else:
                raise NotImplementedError

        if global_step in cfg_train.pg_equ_scale:
            n_rest_equ_scales = len(cfg_train.pg_equ_scale)-cfg_train.pg_equ_scale.index(global_step)-1
            cur_equ_size = (cfg_model.equ_size[0] // (2**n_rest_equ_scales), cfg_model.equ_size[1] // (2**n_rest_equ_scales))
            upsample = global_step > cfg_train.pg_upsample_after
            if isinstance(model, (dpvgo.DirectPanoramaVoxGO)):
                model.scale_equ_grid(cur_equ_size, upsample)
            else:
                raise NotImplementedError
            
        if global_step in cfg_train.pg_msi_scale:
            n_rest_msi_scales = len(cfg_train.pg_msi_scale)-cfg_train.pg_msi_scale.index(global_step)-1
            cur_msi_size = (cfg_model.msi_size[0] // (2**n_rest_msi_scales), cfg_model.msi_size[1] // (2**n_rest_msi_scales))
            upsample = global_step > cfg_train.pg_upsample_after
            if isinstance(model, (dmsigo.DirectMSIGO)):
                model.scale_msi_grid(cur_msi_size, upsample)
            else:
                raise NotImplementedError

        if global_step in cfg_train.pg_scale:
            n_rest_scales = len(cfg_train.pg_scale)-cfg_train.pg_scale.index(global_step)-1
            cur_voxels = int(cfg_model.num_voxels / (2**n_rest_scales))
            if isinstance(model, (dpvgo.DirectPanoramaVoxGO, dmsigo.DirectMSIGO)):
                model.scale_volume_grid(cur_voxels)
            elif isinstance(model, dmpigo.DirectMPIGO):
                model.scale_volume_grid(cur_voxels, model.mpi_depth)
            else:
                raise NotImplementedError
            model.act_shift -= cfg_train.decay_after_scale
            
        if global_step in cfg_train.pg_image_scale or global_step in cfg_train.pg_equ_scale or global_step in cfg_train.pg_scale or global_step in cfg_train.pg_msi_scale:
            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
            torch.cuda.empty_cache()

        # random sample rays
        if cfg_train.ray_sampler in ['flatten']:
            sel_i = batch_index_sampler()
            target = rgb_tr[sel_i]
            rays_o = rays_o_tr[sel_i]
            rays_d = rays_d_tr[sel_i]
            viewdirs = viewdirs_tr[sel_i]
            rays_mask = mask_tr[sel_i] if mask_tr is not None else None
        elif cfg_train.ray_sampler == 'random' or cfg_train.ray_sampler == 'panorama_uniform':
            sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand])
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
            rays_mask = mask_tr[sel_b, sel_r, sel_c] if mask_tr is not None else None
        else:
            raise NotImplementedError

        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)
            if rays_mask is not None:
                rays_mask = rays_mask.to(device)

        # volume rendering
        render_result = model(
            rays_o, rays_d, viewdirs, rays_mask,
            global_step=global_step, is_train=True,
            **render_kwargs)

        # gradient descent step
        optimizer.zero_grad(set_to_none=True)
        loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'], target)
        psnr = utils.mse2psnr(loss.detach())
        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_last'].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss
        if cfg_train.weight_nearclip > 0:
            near_thres = data_dict['near_clip'] / model.scene_radius[0].item()
            near_mask = (render_result['t'] < near_thres)
            density = render_result['raw_density'][near_mask]
            if len(density):
                nearclip_loss = (density - density.detach()).sum()
                loss += cfg_train.weight_nearclip * nearclip_loss
        if cfg_train.weight_distortion > 0:
            n_max = render_result['n_max']
            s = render_result['s']
            w = render_result['weights']
            ray_id = render_result['ray_id']
            loss_distortion = flatten_eff_distloss(w, s, 1/n_max, ray_id)
            loss += cfg_train.weight_distortion * loss_distortion
        if cfg_train.weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
            loss += cfg_train.weight_rgbper * rgbper_loss
        if cfg_train.weight_dudv > 0 and render_result.get('dudv', None) is not None:
            dudv_loss = (render_result['dudv']).pow(2).sum() / len(rays_o)
            loss += cfg_train.weight_dudv * dudv_loss
        loss.backward()

        if global_step<cfg_train.tv_before and global_step>cfg_train.tv_after and global_step%cfg_train.tv_every==0:
            if cfg_train.weight_tv_density>0:
                model.density_total_variation_add_grad(
                    cfg_train.weight_tv_density/len(rays_o), global_step<cfg_train.tv_dense_before)
            if cfg_train.weight_tv_k0>0:
                model.k0_total_variation_add_grad(
                    cfg_train.weight_tv_k0/len(rays_o), global_step<cfg_train.tv_dense_before)

        optimizer.step()
        psnr_lst.append(psnr.item())

        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * decay_factor

        # check log & save
        if global_step%args.i_print==0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                       f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                       f'Eps: {eps_time_str}')
            psnr_lst = []
            ###############
            if cfg.fine_model_and_render.image_size:
                if hasattr(model, 'k0'):
                    Image.fromarray(model.get_k0_grid_rgb()).resize(cfg.fine_model_and_render.image_size[::-1]).save(os.path.join(images_dir, 'k0_{:05d}.png'.format(global_step)))
            if cfg.fine_model_and_render.equ_size:
                if hasattr(model, 'k0'):
                    Image.fromarray(model.get_k0_grid_rgb()).resize(cfg.fine_model_and_render.equ_size[::-1]).save(os.path.join(images_dir, 'k0_{:05d}.png'.format(global_step)))
            if cfg.fine_model_and_render.msi_size:
                if hasattr(model, 'k0'):
                    Image.fromarray(model.get_k0_grid_rgb()).resize(cfg.fine_model_and_render.msi_size[::-1]).save(os.path.join(images_dir, 'k0_{:05d}.png'.format(global_step)))
            ###############

        if global_step%args.i_weights==0:
            path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_{global_step:06d}.tar')
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', path)

    if global_step != -1:
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path)
        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)


def train(args, cfg, data_dict):

    # init
    print('train: start')
    eps_time = time.time()

    savedir = os.path.join(cfg.basedir, cfg.expname)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    elif args.no_reload:
        current_time = datetime.now()
        formatted_time = current_time.strftime('%Y-%m-%d-%H-%M')
        os.rename(savedir, f'{savedir}-{formatted_time}')
        os.makedirs(savedir)
    else:
        return
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

    # coarse geometry searching (only works for inward bounded scenes)
    eps_coarse = time.time()
    if data_dict['xyz_min'] is not None and data_dict['xyz_max'] is not None:
        xyz_min_coarse = torch.Tensor(data_dict['xyz_min'])
        xyz_max_coarse = torch.Tensor(data_dict['xyz_max'])
    elif 'xyz_min' in cfg.data and 'xyz_max' in cfg.data:
        xyz_min_coarse = torch.tensor(cfg.data.xyz_min)
        xyz_max_coarse = torch.tensor(cfg.data.xyz_max)
    else:
        xyz_min_coarse, xyz_max_coarse = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)

    utils.plot_camera_poses(savedir, data_dict['poses'].detach().cpu().numpy(), xyz_min_coarse.detach().cpu().numpy(), xyz_max_coarse.detach().cpu().numpy())
    
    if 'uv_min' in cfg.data and 'uv_max' in cfg.data:
        uv_min_coarse, uv_max_coarse = torch.Tensor(cfg.data.uv_min), torch.Tensor(cfg.data.uv_max)
    else:
        uv_min_coarse, uv_max_coarse = None, None

    if cfg.coarse_train.N_iters > 0:
        scene_rep_reconstruction(
                args=args, cfg=cfg,
                cfg_model=cfg.coarse_model_and_render, cfg_train=cfg.coarse_train,
                xyz_min=xyz_min_coarse, xyz_max=xyz_max_coarse,
                uv_min=uv_min_coarse, uv_max=uv_max_coarse,
                data_dict=data_dict, stage='coarse')
        eps_coarse = time.time() - eps_coarse
        eps_time_str = f'{eps_coarse//3600:02.0f}:{eps_coarse//60%60:02.0f}:{eps_coarse%60:02.0f}'
        print('train: coarse geometry searching in', eps_time_str)
        coarse_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'coarse_last.tar')
    else:
        print('train: skip coarse geometry searching')
        coarse_ckpt_path = None

    # fine detail reconstruction
    eps_fine = time.time()
    if cfg.coarse_train.N_iters == 0:
        xyz_min_fine, xyz_max_fine = xyz_min_coarse.clone(), xyz_max_coarse.clone()
        if 'uv_min' in cfg.data and 'uv_max' in cfg.data:
            uv_min_fine, uv_max_fine = uv_min_coarse.clone(), uv_max_coarse.clone()
        else:
            uv_min_fine, uv_max_fine = None, None
    else:
        raise NotImplementedError
    scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
            xyz_min=xyz_min_fine, xyz_max=xyz_max_fine,
            uv_min=uv_min_fine, uv_max=uv_max_fine,
            data_dict=data_dict, stage='fine',
            coarse_ckpt_path=coarse_ckpt_path)
    eps_fine = time.time() - eps_fine
    eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
    print('train: fine detail reconstruction in', eps_time_str)

    eps_time = time.time() - eps_time
    eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
    print('train: finish (eps time', eps_time_str, ')')


if __name__=='__main__':

    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()

    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg)

    # train
    if not args.render_only:
        train(args, cfg, data_dict)

    # load model for rendring
    if args.render_test or args.render_train or args.render_video or args.render_image or args.render_depth:
        if args.ft_path:
            ckpt_path = args.ft_path
        else:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        if cfg.fine_model_and_render.model_type == 'DirectMPIGO':
            model_class = dmpigo.DirectMPIGO
        elif cfg.fine_model_and_render.model_type == 'DirectPanoramaVoxGO':
            model_class = dpvgo.DirectPanoramaVoxGO
        elif cfg.fine_model_and_render.model_type == 'DirectMSIGO':
            model_class = dmsigo.DirectMSIGO
        else:
            # model_class = dvgo.DirectVoxGO
            raise NotImplementedError
        model = utils.load_model(model_class, ckpt_path).to(device)

        # save k images
        try:
            Image.fromarray(model.get_k0_grid_rgb()).save(os.path.join(cfg.basedir, cfg.expname, 'k0.png'))
            print('k0.png is saved.')
        except:
            print('k0.png is not saved.')

        ###############
        edit = args.edit

        if edit != '':
            model.k0.load(os.path.join(cfg.basedir, cfg.expname, 'k0_{:s}.png'.format(edit)))
        ###############

        stepsize = cfg.fine_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_depth': True,
            },
        }

    # render trainset and eval
    if args.render_train:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}')
        if edit:
            testsavedir += '_{:s}'.format(edit)
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs, depths, bgmaps = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_train']],
                HW=data_dict['HW'][data_dict['i_train']],
                Ks=data_dict['Ks'][data_dict['i_train']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
                savedir=testsavedir, dump_images=args.dump_images,
                eval_ssim=args.eval_ssim, render_panorama=cfg.data.panorama,
                log_metrics=True if edit == '' else False,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)

    # render testset and eval
    if args.render_test:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
        if edit:
            testsavedir += '_{:s}'.format(edit)
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs, depths, bgmaps = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_test']],
                HW=data_dict['HW'][data_dict['i_test']],
                Ks=data_dict['Ks'][data_dict['i_test']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                savedir=testsavedir, dump_images=args.dump_images,
                eval_ssim=args.eval_ssim, render_panorama=cfg.data.panorama,
                log_metrics=True if edit == '' else False,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)

    # render video
    if args.render_video:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_{ckpt_name}')
        if edit:
            testsavedir += '_{:s}'.format(edit)
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        if args.render_panorama:
            HW = data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0)
        else:
            HW = np.array([data_dict['hwf'][:2]]).repeat(len(data_dict['render_poses']), 0)
        rgbs, depths, bgmaps = render_viewpoints(
                render_poses=data_dict['render_poses'],
                HW=HW,
                Ks=data_dict['Ks_render'],
                render_factor=args.render_video_factor,
                render_video_flipy=args.render_video_flipy,
                render_video_rot90=args.render_video_rot90,
                savedir=testsavedir, dump_images=args.dump_images,
                render_panorama=args.render_panorama,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        depths_vis = depths * (1-bgmaps) + bgmaps
        dmin, dmax = np.percentile(depths_vis[bgmaps < 0.1], q=[5, 95])
        depth_vis = plt.get_cmap('rainbow')(1 - np.clip((depths_vis - dmin) / (dmax - dmin), 0, 1)).squeeze()[..., :3]
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(depth_vis), fps=30, quality=8)

    if args.render_image:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_image_{ckpt_name}')
        if edit:
            testsavedir += '_{:s}'.format(edit)
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        if args.render_panorama:
            HW = np.array(data_dict['HW'])[:1]
        else:
            HW = np.array([data_dict['hwf'][:2]])
        render_poses = data_dict['poses'][data_dict['i_train']][[0]]
        rgbs, depths, bgmaps = render_viewpoints(
                render_poses=render_poses,
                HW=HW,
                Ks=data_dict['Ks'][data_dict['i_test']][[0]],
                render_factor=args.render_video_factor,
                render_video_flipy=args.render_video_flipy,
                render_video_rot90=args.render_video_rot90,
                savedir=testsavedir, dump_images=True,
                render_panorama=args.render_panorama,
                **render_viewpoints_kwargs)
        np.save(os.path.join(testsavedir, 'depth.npy'), depths[0, ..., 0])
        depths_vis = depths * (1-bgmaps) + bgmaps
        dmin, dmax = np.percentile(depths_vis[bgmaps < 0.1], q=[5, 95])
        depth_vis = plt.get_cmap('rainbow')(1 - np.clip((depths_vis - dmin) / (dmax - dmin), 0, 1)).squeeze()[..., :3]
        Image.fromarray((depth_vis * 255.).astype(np.uint8)).save(os.path.join(testsavedir, 'depth.png'))

    if args.render_depth:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_depth_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs, depths, bgmaps = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_train']],
                HW=data_dict['HW'][data_dict['i_train']],
                Ks=data_dict['Ks'][data_dict['i_train']],
                render_factor=args.render_video_factor,
                render_video_flipy=args.render_video_flipy,
                render_video_rot90=args.render_video_rot90,
                savedir=testsavedir, dump_images=False,
                render_panorama=args.render_panorama, dump_depths=True,
                **render_viewpoints_kwargs)
        # imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        depths_vis = depths * (1-bgmaps) + bgmaps
        dmin, dmax = np.percentile(depths_vis[bgmaps < 0.1], q=[5, 95])
        depth_vis = plt.get_cmap('rainbow')(1 - np.clip((depths_vis - dmin) / (dmax - dmin), 0, 1)).squeeze()[..., :3]
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(depth_vis), fps=30, quality=8)


    print('Done')

