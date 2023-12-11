_base_ = './llff_default_lg_pe.py'

expname = 'llff_pe/leaves_lg_pe'

data = dict(
    datadir='./data/nerf_llff_data/leaves',
)

fine_train = dict(
    N_iters=60000,
    N_rand=4096,
    pg_scale=[2000,4000,6000,8000],
    pg_image_scale=[8000,16000],
    pg_upsample_after=8000,
    tv_before=1e9,
    tv_dense_before=10000,
    weight_tv_density=1e-5,
    weight_dudv=1e-5,
)
