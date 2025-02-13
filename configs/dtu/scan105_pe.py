_base_ = './dtu_default_pe.py'

expname = 'dtu_pe/scan105_pe'

data = dict(
    datadir='./data/DTU_idr/DTU/scan105',
    near=1.0,
    far=5.0,
    x=0.0,
    y=0.0,
    z=-0.5,
)

fine_train = dict(
    N_iters=60000,
    N_rand=4096,
    pg_scale=[2000,4000,6000,8000],
    pg_msi_scale=[8000,16000],
    pg_upsample_after=8000,
    tv_before=20000,
    tv_dense_before=10000,
    weight_tv_density=1e-5,
    weight_dudv=1e-3,
)
