_base_ = './neuvf_default.py'

basedir = './logs'

fine_model_and_render = dict(
    xyz_config={
        'enc_type': 'pe',
        'pe': {
            'N_freqs': 8,
            'annealed_step': 4000,
            'annealed_begin_step': 4000,
            'identity': True,
        },
    },
    deformation_config={
        'deform_type': 'mlp',
        'mlp': {
            'D': 8,
            'W': 128,
            'skips': [4],
        },
    },
)

