import numpy as np
import os

from .load_llff import load_llff_data
from .load_replica import load_replica_data
from .load_in2n import load_in2n_data
from .load_blender import load_blender_data
from .load_neuvf import load_neuvf_data
from .load_dtu import load_dtu_data


def load_data(args):

    K, depths = None, None
    K_render = None
    near_clip = None
    xyz_min, xyz_max = None, None
    masks = None

    if args.dataset_type == 'in2n':
        images, depths, poses, bds, render_poses, i_test, K, K_render = load_in2n_data(
                args.datadir, args.factor, args.width, args.height,
                recenter=True, bd_factor=args.bd_factor,
                move_back=args.move_back,
                spherify=args.spherify,
                load_depths=args.load_depths,
                selected_frames=args.selected_frames,
                movie_render_kwargs=args.movie_render_kwargs)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        # i_train = np.array([i for i in np.arange(int(images.shape[0])) if
        #                 (i not in i_test and i not in i_val)])
        i_train = np.array([i for i in np.arange(int(images.shape[0]))])

        print('DEFINING BOUNDS')
        if args.ndc:
            near = 0.
            far = 1.
        else:
            # near_clip = max(np.ndarray.min(bds) * .9, 0)
            # _far = max(np.ndarray.max(bds) * 1., 0)
            # near = 0
            # far = inward_nearfar_heuristic(poses[i_train, :3, 3])[1]
            # print('near_clip', near_clip)
            # print('original far', _far)
            raise NotImplementedError
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'replica':
        images, poses, hwf, render_poses, i_split, K = load_replica_data(basedir=args.datadir, movie_render_kwargs=args.movie_render_kwargs)
        print('Loaded replica', images.shape, poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        
        near = 0.
        far = 1.0
        print('NEAR FAR', near, far)
    
    elif args.dataset_type == 'llff':
        images, masks, depths, poses, bds, render_poses, i_test = load_llff_data(
                args.datadir, args.factor, args.width, args.height,
                recenter=True, bd_factor=args.bd_factor,
                move_back=args.move_back,
                spherify=args.spherify,
                load_depths=args.load_depths,
                movie_render_kwargs=args.movie_render_kwargs)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.ndc:
            near = 0.
            far = 1.
        else:
            raise NotImplementedError
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'neuvf':
        images, poses, render_poses, hwf, i_split, K, xyz_min, xyz_max = load_neuvf_data(args.datadir, args.factor, args.x, args.y, args.z)
        print('Loaded NeUVF', images.shape, poses.shape, render_poses.shape, hwf)
        i_train, i_val, i_test = i_split

        near = args.near
        far = args.far
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'dtu':
        images, masks, poses, render_poses, hwf, i_split, K = load_dtu_data(args.datadir, args.width, args.height, args.factor, args.x, args.y, args.z)
        i_train, i_val, i_test = i_split
        
        # import pdb; pdb.set_trace()
        near = args.near
        far = args.far

        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip, args.x, args.y, args.z)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        # near, far = 2., 6.
        near = args.near
        far = args.far

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]*images[...,-1:]

    else:
        raise NotImplementedError(f'Unknown dataset type {args.dataset_type} exiting')

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    HW = np.array([im.shape[:2] for im in images])
    irregular_shape = (images.dtype is np.dtype('object'))

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K

    render_poses = render_poses[...,:4]
    if K_render is None:
        K_render = K
    if len(K_render.shape) == 2:
        Ks_render = K_render[None].repeat(len(render_poses), axis=0)
    else:
        Ks_render = K_render[0:1].repeat(len(render_poses), axis=0)

    data_dict = dict(
        hwf=hwf, HW=HW, Ks=Ks, Ks_render=Ks_render,
        near=near, far=far, near_clip=near_clip,
        i_train=i_train, i_val=i_val, i_test=i_test,
        poses=poses, render_poses=render_poses,
        images=images, masks=masks, depths=depths,
        irregular_shape=irregular_shape,
        xyz_min=xyz_min, xyz_max=xyz_max,
    )
    return data_dict
