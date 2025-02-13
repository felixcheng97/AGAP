_base_ = '../default.py'

basedir = './logs'

import numpy as np
data = dict(
    dataset_type='blender',
    white_bkgd=True,
    object=True,
    uv_min=[-np.pi/2, -np.pi/2],
    uv_max=[ np.pi/2,  np.pi/2],
    near=2.0,
    far=6.0,
)

coarse_train = dict(
    N_iters=0,
)

fine_train = dict(
    N_iters=60000,
    N_rand=4096,
    weight_distortion=0,
    pg_scale=[],
    pg_msi_scale=[],
    pg_upsample_after=8000,
    decay_after_scale=1.0,
    ray_sampler='flatten',
    tv_before=0,
    tv_dense_before=0,
    weight_tv_density=1e-5,
    weight_dudv=1e-5,
)

fine_model_and_render = dict(
    model_type='DirectMSIGO',
    # k0_type='DenseBevExplicit',
    k0_type='DenseMSIExplicit',
    # num_voxels=320**3,
    # num_voxels_base=320**3,
    num_voxels=160**3,
    num_voxels_base=160**3,
    msi_size=(768,768),
    rgbnet_dim=0,
    rgbnet_depth=3,
    rgbnet_width=64,
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

