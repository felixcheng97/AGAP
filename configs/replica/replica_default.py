_base_ = '../default.py'

basedir = './logs/replica'

data = dict(
    dataset_type='replica',
    rand_bkgd=True,
    panorama=True,
)

coarse_train = dict(N_iters=0)

fine_train = dict(
    N_iters=60000,
    N_rand=4096,
    weight_distortion=0.01,
    pg_scale=[2000,4000,6000,8000,10000,12000,14000,16000],
    pg_equ_scale=[4000,8000,12000,16000],
    pg_upsample_after=8000,
    decay_after_scale=0.1,
    ray_sampler='panorama_uniform',
    tv_before=20000,
    tv_dense_before=10000,
    weight_tv_density=1e-4,
    weight_dudv=1e-1,
)

fine_model_and_render = dict(
    model_type='DirectPanoramaVoxGO',
    k0_type='DenseEquExplicit',
    num_voxels=320**3,
    num_voxels_base=320**3,
    equ_size=(768,1536),
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