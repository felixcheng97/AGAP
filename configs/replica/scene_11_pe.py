_base_ = './replica_default_pe.py'

expname = 'replica_pe/dvgo_scene_11_pe'

data = dict(
    datadir='./data/somsi_data/replica/scene_11',
    movie_render_kwargs=dict(
        H=192*4,
        W=256*4,
        FOV=77,
    ),
)

fine_train = dict(
    N_iters=60000,
    N_rand=4096,
    tv_before=20000,
    tv_dense_before=10000,
    weight_tv_density=1e-4,
    weight_dudv=1e-1,
)
