_base_ = './in2n_default_lg_pe.py'

expname = 'in2n_pe/face_lg_pe'

data = dict(
    datadir='./data/in2n_data/face',
    selected_frames=[i for i in range(1, 66) if i not in [6]],
    xyz_min=[-1.2, -1.2, -1.],
    xyz_max=[ 1.2,  1.2,  1.],
    movie_render_kwargs=dict(
        percentile=50,
        z=0.05,
    ),
)

fine_train = dict(
    N_iters=60000,
    N_rand=4096,
    pg_scale=[2000,4000,6000,8000],
    pg_image_scale=[8000,16000],
    pg_upsample_after=8000,
    tv_before=1e9,
    tv_dense_before=10000,
    weight_tv_density=5e-5,
    weight_dudv=1e-5,
)