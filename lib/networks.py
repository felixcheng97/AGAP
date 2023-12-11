import torch
from torch import nn
import math
try:
    import tinycudann as tcnn
except:
    print('warning: tinycudann is not imported, so only PE mode is available.')


class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, annealed_step=0, annealed_begin_step=0, logscale=True, identity=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.in_channels = in_channels
        self.N_freqs = N_freqs
        self.annealed_step = annealed_step
        self.annealed_begin_step = annealed_begin_step
        self.logscale = logscale
        self.identity = identity

        self.funcs = [torch.sin, torch.cos]
        if self.logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + (1 if identity else 0))
        
        self.index = torch.linspace(0, N_freqs-1, N_freqs)

    def extra_repr(self):
        return f'in_channels={self.in_channels}, N_freqs={self.N_freqs}, ' \
               f'annealed_step={self.annealed_step}, annealed_begin_step={self.annealed_begin_step}, ' \
               f'logscale={self.logscale}, identity={self.identity}'

    def forward(self, x, step=None):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        if not self.identity and self.N_freqs == 0:
            return x.new_zeros(*x.shape[:-1], 0)

        if step is None:
            step = self.annealed_begin_step + self.annealed_step

        if self.annealed_step == 0:
            alpha = self.N_freqs
        elif step < self.annealed_begin_step:
            alpha = 0
        else:
            alpha = self.N_freqs * (step - self.annealed_begin_step) / float(self.annealed_step)

        if self.identity:
            out = [x]
        else:
            out = []

        for j, freq in enumerate(self.freq_bands):
            w = (1 - torch.cos(math.pi * torch.clamp(alpha - self.index[j], 0., 1.))) / 2
            for func in self.funcs:
                out += [w * func(freq*x)]

        return torch.cat(out, -1)


class ViewdirEncoding(Embedding):
    def __init__(self, *args, **kwargs):
        super(ViewdirEncoding, self).__init__(*args, **kwargs)


class PositionalEncoding(Embedding):
    def __init__(self, *args, **kwargs):
        super(PositionalEncoding, self).__init__(*args, **kwargs)


class HashEncoding(nn.Module):
    def __init__(self, encoding_config, annealed_step=0, annealed_begin_step=0, identity=True):
        super(HashEncoding, self).__init__()
        self.encoding_config = encoding_config
        self.N_freqs = encoding_config['n_levels']
        self.N_features = encoding_config['n_features_per_level']
        self.annealed_step = annealed_step
        self.annealed_begin_step = annealed_begin_step
        self.identity = identity

        n_input_dims = 3
        self.encoder = tcnn.Encoding(n_input_dims=n_input_dims, encoding_config=encoding_config)
        self.out_channels = self.encoder.n_output_dims + (n_input_dims if identity else 0)

        index = torch.linspace(0, self.N_freqs-1, self.N_freqs)
        self.index = index[:, None].expand(self.N_freqs, self.N_features).reshape(-1)

    def extra_repr(self):
        return f'N_freqs={self.N_freqs}, N_features={self.N_features}, ' \
               f'annealed_step={self.annealed_step}, annealed_begin_step={self.annealed_begin_step}, ' \
               f'identity={self.identity}'

    def forward(self, x, step=0):
        if not self.identity and self.N_freqs == 0:
            return x.new_zeros(*x.shape[:-1], 0)

        if step is None:
            step = self.annealed_begin_step + self.annealed_step

        if self.annealed_step == 0:
            alpha = self.N_freqs
        elif step < self.annealed_begin_step:
            alpha = 0
        else:
            alpha = self.N_freqs * (step - self.annealed_begin_step) / float(self.annealed_step)
        w = (1 - torch.cos(math.pi * torch.clamp(alpha - self.index, 0, 1))) / 2

        x_emb = self.encoder(x)
        out = x_emb * w[None]

        if self.identity:
            out = torch.cat([x, out], dim=-1)
        return out


class DeformationMLP(nn.Module):
    def __init__(self, in_channels, D=8, W=128, skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels: number of input channels for xyz and dir
        skips: add skip connection in the Dth layer
        """
        super(DeformationMLP, self).__init__()
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels = in_channels

        # encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels, W)
            else:
                layer = nn.Linear(W, W)
            torch.nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layer = nn.Sequential(layer, nn.ReLU(True))
            # init the models
            setattr(self, f"xyz_dir_encoding_{i+1}", layer)
        out_layer = nn.Linear(W, 2)
        nn.init.zeros_(out_layer.bias)
        nn.init.uniform_(out_layer.weight, -1e-4, 1e-4)
        self.output = nn.Sequential(out_layer)

    def forward(self, x):
        """
        Encodes input xyz and dir as deformation field for points

        Inputs:
            x: (B, self.in_channels)
               the embedded vector of position and direction
        Outputs:
            dudv: deformation field
        """
        input_xyz = x

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_dir_encoding_{i+1}")(xyz_)

        dudv = self.output(xyz_)
        return dudv


class DeformationTCNN(nn.Module):
    def __init__(self, in_channels, n_blocks, hidden_neurons, network_config):
        super(DeformationTCNN, self).__init__()
        self.in_channels = in_channels
        self.n_blocks = n_blocks
        self.hidden_neurons = hidden_neurons
        self.network_config = network_config

        for i in range(n_blocks):
            n_input_dims = in_channels if i == 0 else in_channels+hidden_neurons
            n_output_dims = 2 if i == n_blocks - 1 else hidden_neurons
            layer = tcnn.Network(n_input_dims=n_input_dims, n_output_dims=n_output_dims, network_config=network_config)
            if i == n_blocks - 1:
                layer = nn.Sequential(layer)
            else:
                layer = nn.Sequential(layer, nn.ReLU())
            setattr(self, f"layer_{i+1}", layer)

    def forward(self, x, step=0):
        skip = x

        x_ = x.new_zeros(*x.shape[:-1], 0)
        for i in range(self.n_blocks):
            x_ = torch.cat([x_, skip], dim=-1)
            x_ = getattr(self, f"layer_{i+1}")(x_)

        dudv = x_
        return dudv

