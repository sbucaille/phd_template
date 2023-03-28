from functools import partial

import torch
from torch import nn

from einops.layers.torch import Rearrange, Reduce


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )


def MLPMixer_(S, input_dim, dim, output_dim, depth, expansion_factor=4, dropout=0.):
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

    return nn.Sequential(
        nn.Linear(input_dim, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(S, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, output_dim)
    )


class MLPMixer(nn.Module):
    def __init__(
            self,
            S: int,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            depth: int,
            expansion_factor: int = 4,
            dropout: float = 0.,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            *[nn.Sequential(
                PreNormResidual(hidden_dim, FeedForward(S, expansion_factor, dropout, chan_first)),
                PreNormResidual(hidden_dim, FeedForward(hidden_dim, expansion_factor, dropout, chan_last))
            ) for _ in range(depth)],
            nn.LayerNorm(hidden_dim),
            Reduce('b n c -> b c', 'mean'),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class DeltaBlock(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, corr_levels=4, corr_radius=3, S=8):
        super(DeltaBlock, self).__init__()

        self.input_dim = input_dim

        kitchen_dim = (corr_levels * (2 * corr_radius + 1) ** 2) + input_dim + 64 * 3 + 3

        self.hidden_dim = hidden_dim

        self.S = S

        self.to_delta = MLPMixer(
            S=self.S,
            input_dim=kitchen_dim,
            hidden_dim=512,
            output_dim=self.S * (input_dim + 2),
            depth=12,
        )

    def forward(self, features, correlations, flows):
        B, S, D = flows.shape
        assert (D == 3)
        flow_sincos = self.get_3d_positional_embedding(flows, 64, cat_coordinates=True)
        x = torch.cat([features, correlations, flow_sincos], dim=2)  # B, S, -1
        delta = self.to_delta(x)
        delta = delta.reshape(B, self.S, self.input_dim + 2)
        return delta

    @staticmethod
    def get_3d_positional_embedding(xyz, embedding_size, cat_coordinates=True):
        B, N, D = xyz.shape
        assert D == 3

        x = xyz[:, :, 0:1]
        y = xyz[:, :, 1:2]
        z = xyz[:, :, 2:3]
        division_term = torch.arange(0, embedding_size, 2,
                                     device=xyz.device,
                                     dtype=torch.float32)
        division_term = division_term * (1000.0 / embedding_size)
        division_term = division_term.reshape(1, 1, int(embedding_size / 2))

        positional_embedding_x = torch.zeros(B, N, embedding_size, device=xyz.device, dtype=torch.float32)
        positional_embedding_y = torch.zeros(B, N, embedding_size, device=xyz.device, dtype=torch.float32)
        positional_embedding_z = torch.zeros(B, N, embedding_size, device=xyz.device, dtype=torch.float32)

        positional_embedding_x[:, :, 0::2] = torch.sin(x * division_term)
        positional_embedding_x[:, :, 1::2] = torch.cos(x * division_term)
        positional_embedding_y[:, :, 0::2] = torch.sin(y * division_term)
        positional_embedding_y[:, :, 1::2] = torch.cos(y * division_term)
        positional_embedding_z[:, :, 0::2] = torch.sin(z * division_term)
        positional_embedding_z[:, :, 1::2] = torch.cos(z * division_term)

        positional_embedding = torch.cat([positional_embedding_x,
                                          positional_embedding_y,
                                          positional_embedding_z], dim=2)
        if cat_coordinates:
            positional_embedding = torch.cat([xyz, positional_embedding], dim=2)
        return positional_embedding
