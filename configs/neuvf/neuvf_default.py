_base_ = '../default.py'

basedir = './logs'

import numpy as np
data = dict(
    dataset_type='neuvf',
    object_msi=True,
    factor=8,
    uv_min=[-np.pi/2, -np.pi/2],
    uv_max=[ np.pi/2,  np.pi/2],
)

coarse_train = dict(
    N_iters=0,
)

fine_train = dict(
    N_iters=60000,
    N_rand=4096,
    weight_distortion=0,
    pg_scale=[2000,4000,6000,8000],
    pg_msi_scale=[8000,16000],
    pg_upsample_after=8000,
    decay_after_scale=1.0,
    ray_sampler='flatten',
    # ray_sampler='random',
    tv_before=20000,
    tv_dense_before=10000,
    weight_tv_density=1e-5,
    weight_dudv=1e-5,
)

fine_model_and_render = dict(
    model_type='DirectMSIGO',
    k0_type='DenseMSIExplicit',
    num_voxels=320**3,
    num_voxels_base=320**3,
    msi_size=(768,768),
    alpha_init=1e-4,
    stepsize=0.5,
    rgbnet_dim=0,
    rgbnet_depth=3,
    rgbnet_width=128,
    world_bound_scale=1,
    fast_color_thres=1e-4,
    viewdirs_config={
        'enc_type': 'pe',
        'pe': {
            'N_freqs': 4,
            'annealed_step': 0,
            'annealed_begin_step': 4000,
            'identity': True,
        },
    },
)

