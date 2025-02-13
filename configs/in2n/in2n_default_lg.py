_base_ = '../default.py'

basedir = './logs'

data = dict(
    dataset_type='in2n',
    ndc=True,
    factor=1,
    rand_bkgd=True,
    move_back=True,
)

coarse_train = dict(
    N_iters=0,
)

fine_train = dict(
    N_iters=60000,
    N_rand=4096,
    weight_distortion=0.01,
    pg_scale=[2000,4000,6000,8000],
    pg_image_scale=[8000,16000],
    pg_upsample_after=8000,
    decay_after_scale=0.1,
    ray_sampler='flatten',
    tv_before=1e9,
    tv_dense_before=10000,
    weight_tv_density=1e-5,
    weight_dudv=1e-5,
)

_mpi_depth = 256
_stepsize = 1.0

fine_model_and_render = dict(
    model_type='DirectMPIGO',
    k0_type='Dense2DExplicit',
    num_voxels=384*384*_mpi_depth,
    image_size=(768,),
    mpi_depth=_mpi_depth,
    stepsize=_stepsize,
    rgbnet_dim=0,
    rgbnet_depth=3,
    rgbnet_width=64,
    world_bound_scale=1,
    fast_color_thres=_stepsize/_mpi_depth/5,
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

