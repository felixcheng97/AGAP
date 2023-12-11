import numpy as np
import os

from .load_llff import load_llff_data
from .load_replica import load_replica_data


def load_data(args):

    K, depths = None, None
    K_render = None
    near_clip = None

    if args.dataset_type == 'replica':
        images, poses, hwf, render_poses, i_split, K = load_replica_data(basedir=args.datadir, movie_render_kwargs=args.movie_render_kwargs)
        print('Loaded replica', images.shape, poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        
        near = 0.
        far = 1.0
        print('NEAR FAR', near, far)
    
    elif args.dataset_type == 'llff':
        images, depths, poses, bds, render_poses, i_test = load_llff_data(
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
        Ks_render = K_render

    data_dict = dict(
        hwf=hwf, HW=HW, Ks=Ks, Ks_render=Ks_render,
        near=near, far=far, near_clip=near_clip,
        i_train=i_train, i_val=i_val, i_test=i_test,
        poses=poses, render_poses=render_poses,
        images=images, depths=depths,
        irregular_shape=irregular_shape,
    )
    return data_dict
