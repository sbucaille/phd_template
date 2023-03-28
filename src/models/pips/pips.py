import numpy as np
import torch
import torch.nn.functional as F

from torch import nn, Tensor

from src.models.pips.delta import DeltaBlock
from src.models.pips.raft_encoder import CorrelationBlock


class Pips(nn.Module):
    def __init__(
            self,
            S: int,
            stride: int,
            hidden_dim: int,
            latent_dim: int,
            corr_levels: int,
            corr_radius: int,
            encoder: nn.Module,
            delta_block: DeltaBlock,
            *args,
            **kwargs
    ):
        super().__init__()
        self.S = S
        self.stride = stride
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius

        self.encoder = encoder
        self.delta_block = delta_block
        # self.norm = nn.GroupNorm(1, self.latent_dim)
        # GroupNorm(1, x) is equivalent to LayerNorm(x)
        self.norm = nn.LayerNorm(self.latent_dim)
        self.current_feature_updater = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU()
        )
        self.visibility_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 1)
        )

    def forward(
            self,
            xys: Tensor,
            rgbs: Tensor,
            initial_coordinates: Tensor = None,
            initial_features: Tensor = None,
            iterations: int = 3,
            return_features_correlations: bool = False
    ):
        B, N, D = xys.shape
        assert D == 2

        B_, S, C, H, W = rgbs.shape
        assert B == B_

        strided_H = H // self.stride
        strided_W = W // self.stride

        # TODO delete maybe
        device = rgbs.device

        rgbs = rgbs.reshape(B * S, C, H, W)

        # Extract features from all images by passing it to the encoder
        features = self.encoder(rgbs)
        features = features.reshape(B, S, self.latent_dim, strided_H, strided_W)

        xys = xys.clone() / float(self.stride)

        if initial_coordinates is None:
            # If no initial coordinates provided, use i
            coordinates = xys.reshape(B, 1, N, 2).repeat(1, S, 1, 1)
        else:
            coordinates = initial_coordinates.clone() / self.stride

        correlation_block = CorrelationBlock(
            features,
            num_levels=self.corr_levels,
            radius=self.corr_radius
        )

        if initial_features is None:
            current_features = self.sample_features_from_coordinates(
                features=features[:, 0],
                X=coordinates[:, 0, :, 0],
                Y=coordinates[:, 0, :, 1]
            )
            current_features = current_features.permute(0, 2, 1)
        else:
            current_features = initial_features

        current_features = current_features.unsqueeze(1).repeat(1, S, 1, 1)  # B, S, N, C
        coordinates_predictions = []
        coordinates_predictions2 = []

        coordinates_predictions2.append(coordinates.detach() * self.stride)
        coordinates_predictions2.append(coordinates.detach() * self.stride)

        features_correlations = []

        for iteration in range(iterations):
            coordinates = coordinates.detach()
            correlation_block.correlate(current_features)
            features_correlation = torch.zeros((B, S, N, strided_H, strided_W), device=device)
            for correlation_level in range(self.corr_levels):
                level_feature_correlation = correlation_block.correlations_pyramid[correlation_level]
                _, _, _, level_H, level_W = level_feature_correlation.shape
                level_feature_correlation = level_feature_correlation.reshape(B * S, N, level_H, level_W)
                level_feature_correlation = F.interpolate(
                    level_feature_correlation,
                    (strided_H, strided_W),
                    mode='bilinear',
                    align_corners=True
                )
                features_correlation = features_correlation + level_feature_correlation.reshape(B, S, N,
                                                                                                strided_H,
                                                                                                strided_W)
            features_correlations.append(features_correlation)
            correlations = correlation_block.sample(coordinates)
            LRR = correlations.shape[3]

            correlations = correlations.permute(0, 2, 1, 3).reshape(B * N, S, LRR)
            flows = (coordinates - coordinates[:, 0:1]).permute(0, 2, 1, 3).reshape(B * N, S, 2)
            times = torch.linspace(0, S, S, device=device).reshape(1, S, 1).repeat(B * N, 1, 1)
            flows = torch.cat([flows, times], dim=2)

            current_features = current_features.permute(0, 2, 1, 3).reshape(B * N, S, self.latent_dim)
            delta = self.delta_block(current_features, correlations, flows)
            delta_coordinates = delta[:, :, :2]
            delta_features = delta[:, :, 2:]

            current_features = current_features.reshape(B * N * S, self.latent_dim)
            delta_features = delta_features.reshape(B * N * S, self.latent_dim)
            delta_features = self.norm(delta_features)
            current_features = self.current_feature_updater(self.norm(delta_features)) + current_features
            current_features = current_features.reshape(B, N, S, self.latent_dim).permute(0, 2, 1, 3)

            coordinates = coordinates + delta_coordinates.reshape(B, N, S, 2).permute(0, 2, 1, 3)

            coordinates_predictions.append(coordinates * self.stride)
            coordinates_predictions2.append(coordinates * self.stride)

        visibility_probability = self.visibility_predictor(current_features.reshape(B * S * N, self.latent_dim))
        visibility_probability = visibility_probability.reshape(B, S, N)

        coordinates_predictions2.append(coordinates * self.stride)
        coordinates_predictions2.append(coordinates * self.stride)

        features_correlations = torch.stack(features_correlations, dim=2)
        if return_features_correlations:
            return coordinates_predictions, coordinates_predictions2, visibility_probability, features_correlations
        else:
            return coordinates_predictions, coordinates_predictions2, visibility_probability

    @staticmethod
    def sample_features_from_coordinates(features, X, Y):
        # equivalent to utils.samp.bilinear_sample2d in original repository
        # features represent the image
        # X and Y are the coordinates to extract the features from
        B, C, H, W = features.shape
        N = X.shape[1]

        max_X = W - 1
        max_Y = H - 1

        X0 = torch.floor(X).int()
        X1 = X0 + 1
        Y0 = torch.floor(Y).int()
        Y1 = Y0 + 1

        X0_clip = torch.clamp(X0, 0, max_X)
        X1_clip = torch.clamp(X1, 0, max_X)
        Y0_clip = torch.clamp(Y0, 0, max_Y)
        Y1_clip = torch.clamp(Y1, 0, max_Y)
        dim1 = W * H
        dim2 = W

        base = torch.arange(0, B, dtype=torch.int64, device=features.device) * dim1
        base = torch.reshape(base, [B, 1]).repeat([1, N])

        base_Y0 = base + Y0_clip * dim2
        base_Y1 = base + Y1_clip * dim2

        index_Y0_X0 = base_Y0 + X0_clip
        index_Y0_X1 = base_Y0 + X1_clip
        index_Y1_X0 = base_Y1 + X0_clip
        index_Y1_X1 = base_Y1 + X1_clip

        features_flat = features.permute(0, 2, 3, 1).reshape(B * H * W, C)
        features_Y0_X0 = features_flat[index_Y0_X0]
        features_Y0_X1 = features_flat[index_Y0_X1]
        features_Y1_X0 = features_flat[index_Y1_X0]
        features_Y1_X1 = features_flat[index_Y1_X1]

        W_Y0_X0 = ((X1 - X) * (Y1 - Y)).unsqueeze(2)
        W_Y0_X1 = ((X - X0) * (Y1 - Y)).unsqueeze(2)
        W_Y1_X0 = ((X1 - X) * (Y - Y0)).unsqueeze(2)
        W_Y1_X1 = ((X - X0) * (Y - Y0)).unsqueeze(2)

        output = W_Y0_X0 * features_Y0_X0 + W_Y0_X1 * features_Y0_X1 + \
                 W_Y1_X0 * features_Y1_X0 + W_Y1_X1 * features_Y1_X1
        output = output.view(B, -1, C)
        output = output.permute(0, 2, 1)
        return output  # B, C, N

    def load_state_dict_from_original(self, weights_path):
        state_dict = torch.load(weights_path)['model_state_dict']
        state_dict = {k.replace('fnet', 'encoder'): v for k, v in state_dict.items()}
        state_dict = {k.replace('ffeat_updater', 'current_feature_updater'): v for k, v in state_dict.items()}
        state_dict = {k.replace('vis_predictor', 'visibility_predictor'): v for k, v in state_dict.items()}
        self.load_state_dict(state_dict, strict=False)

    @staticmethod
    def generate_inputs(input_info, device: torch.device):
        B, S, N, H, W = input_info.batch_size, input_info.num_frames, input_info.num_points, input_info.height, input_info.width

        # Create B batch of S random images of size H by W
        rgbs = torch.rand(
            input_info.batch_size,
            input_info.num_frames,
            3,
            input_info.height,
            input_info.width,
            device=device
        )

        # Create uniform grid of N points coordinates
        N_ = np.sqrt(N).round().astype(np.int32)
        grid_x = torch.linspace(0.0, N_ - 1, N_, device=device).reshape([1, 1, N_]).repeat([B, N_, 1])
        grid_x = N_ / 2 + grid_x.reshape(B, -1) / float(N_ - 1) * (W - N_)

        grid_y = torch.linspace(0.0, N_ - 1, N_, device=device).reshape([1, N_, 1]).repeat([B, 1, N_])
        grid_y = N_ / 2 + grid_y.reshape(B, -1) / float(N_ - 1) * (H - N_)

        xys = torch.stack([grid_x, grid_y], dim=-1)
        return ["xyz", "rgbs"], (xys, rgbs)
