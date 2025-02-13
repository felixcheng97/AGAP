_base_ = './nerf_default_pe.py'

expname = 'nerf_pe/hotdog_pe'

import numpy as np
data = dict(
    datadir='./data/nerf_synthetic/hotdog',
    white_bkgd=False,
    xyz_min=[-2.0, -2.0,  0.85],
    xyz_max=[ 2.0,  2.0,  2.0],
    uv_min=[-np.pi/3, -np.pi/3],
    uv_max=[ np.pi/3,  np.pi/3],
    near=3.0,
    far=5.0,
    x=0.0,
    y=0.0,
    z=-1.0,
)

fine_train = dict(
    N_iters=60000,
    N_rand=4096,
    pg_scale=[2000,4000,6000,8000],
    pg_msi_scale=[4000,8000,12000,16000],
    pg_upsample_after=8000,
    tv_before=20000,
    tv_dense_before=10000,
    weight_tv_k0=1e-5,
    weight_tv_density=1e-4,
    weight_dudv=1e-1,
)