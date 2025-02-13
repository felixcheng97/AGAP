_base_ = './neuvf_default_pe.py'

expname = 'neuvf_pe/mali_pe'

import numpy as np
data = dict(
    datadir='./data/NeUVF_data/mali',
    # near=4.5,
    # far=6.5,
    near=4.5,
    far=5.5,
    x=0.0,
    y=0.15,
    z=-0.4,
    # uv_min=[-np.pi/2 * 4 / 5, -np.pi/2 * 2 / 5],
    # uv_max=[ np.pi/2 * 4 / 5,  np.pi/2 * 2 / 5],
    uv_min=[-np.pi/2, -np.pi/2],
    uv_max=[ np.pi/2,  np.pi/2],
)

fine_train = dict(
    N_iters=60000,
    N_rand=4096,
    pg_scale=[2000,4000,6000,8000],
    pg_msi_scale=[8000,16000],
    pg_upsample_after=8000,
    tv_before=20000,
    tv_dense_before=10000,
    weight_tv_density=1e-2,
    weight_dudv=1e-2,
)