_base_ = './replica_default.py'

basedir = './logs'

fine_model_and_render = dict(
    xyz_config={
        'enc_type': 'hash',
        'hash': {
            'encoding_config': {
                'otype': 'HashGrid',
                'n_levels': 16,
                'n_features_per_level': 2,
                'log2_hashmap_size': 19,
                'base_resolution': 16,
                'per_level_scale': 1.38,
            },
            'annealed_step': 0,
            'annealed_begin_step': 4000,
            'identity': True,
        },
    },
    deformation_config={
        'deform_type': 'tcnn',
        'tcnn': {
            'n_blocks': 1,
            'hidden_neurons': 64,
            'network_config': {
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'None',
                'n_neurons': 64,
                'n_hidden_layers': 8,
            },
        },
    },
)