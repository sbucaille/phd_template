import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, normalization_function='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride, padding_mode='zeros')
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, padding_mode='zeros')
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if normalization_function == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif normalization_function == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif normalization_function == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif normalization_function == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class RAFTEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            stride: int,
            normalization_function: str,
            dropout: float,
            in_planes: int,
            shallow: bool,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.stride = stride
        self.normalization_function = normalization_function
        self.dropout = dropout
        self.in_planes = in_planes
        self.shallow = shallow

        if self.normalization_function == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=self.in_planes)
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=output_dim * 2)
        elif self.normalization_function == 'batch':
            self.norm1 = nn.BatchNorm2d(self.in_planes)
            self.norm2 = nn.BatchNorm2d(self.output_dim * 2)
        elif self.normalization_function == 'instance':
            self.norm1 = nn.InstanceNorm2d(self.in_planes)
            self.norm2 = nn.InstanceNorm2d(self.output_dim * 2)
        elif self.normalization_function == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(input_dim, self.in_planes, kernel_size=7, stride=2, padding=3, padding_mode='zeros')
        self.relu1 = nn.ReLU(inplace=True)

        if self.shallow:
            self.layer1 = self._make_layer(64, stride=1)
            self.layer2 = self._make_layer(96, stride=2)
            self.layer3 = self._make_layer(128, stride=2)
            self.conv2 = nn.Conv2d(128 + 96 + 64, output_dim, kernel_size=1)
        else:
            self.layer1 = self._make_layer(64, stride=1)
            self.layer2 = self._make_layer(96, stride=2)
            self.layer3 = self._make_layer(128, stride=2)
            self.layer4 = self._make_layer(128, stride=2)

            self.conv2 = nn.Conv2d(128 + 128 + 96 + 64, output_dim * 2, kernel_size=3, padding=1, padding_mode='zeros')
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = nn.Conv2d(output_dim * 2, output_dim, kernel_size=1)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(
            self.in_planes,
            dim,
            self.normalization_function,
            stride=stride
        )
        layer2 = ResidualBlock(
            dim,
            dim,
            self.normalization_function,
            stride=1
        )
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, H, W = x.shape

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        if self.shallow:
            a = self.layer1(x)
            b = self.layer2(a)
            c = self.layer3(b)
            a = F.interpolate(a, (H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
            b = F.interpolate(b, (H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
            c = F.interpolate(c, (H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
            x = self.conv2(torch.cat([a, b, c], dim=1))
        else:
            a = self.layer1(x)
            b = self.layer2(a)
            c = self.layer3(b)
            d = self.layer4(c)
            a = F.interpolate(a, (H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
            b = F.interpolate(b, (H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
            c = F.interpolate(c, (H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
            d = F.interpolate(d, (H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
            x = self.conv2(torch.cat([a, b, c, d], dim=1))
            x = self.norm2(x)
            x = self.relu2(x)
            x = self.conv3(x)

        return x


class CorrelationBlock(nn.Module):
    def __init__(
            self,
            features,
            num_levels=4,
            radius=4
    ):
        B, self.S, C, H, W = features.shape
        self.C, self.H, self.W = C, H, W
        self.num_levels = num_levels
        self.radius = radius
        self.features_pyramid = []

        self.features_pyramid.append(features)
        for i in range(self.num_levels - 1):
            features = features.reshape(B * self.S, C, H, W)
            features = F.avg_pool2d(features, 2, stride=2)
            _, _, H, W = features.shape
            features = features.reshape(B, self.S, C, H, W)
            self.features_pyramid.append(features)

    def correlate(self, targets):
        B, S, N, C = targets.shape
        assert C == self.C
        assert S == self.S

        features_1 = targets
        self.correlations_pyramid = []
        for features in self.features_pyramid:
            _, _, _, H, W = features.shape
            features_2 = features.view(B, S, C, H * W)
            correlations = torch.matmul(features_1, features_2)
            correlations = correlations.view(B, S, N, H, W)
            correlations = correlations / torch.sqrt(torch.tensor(C).float())
            self.correlations_pyramid.append(correlations)

    def sample(self, coordinates):
        B, S, N, D = coordinates.shape
        assert D == 2

        out_pyramid = []
        for i in range(self.num_levels):
            level_correlations = self.correlations_pyramid[i]
            _, _, _, H, W = level_correlations.shape
            dx = torch.linspace(-self.radius, self.radius, 2 * self.radius + 1)
            dy = torch.linspace(-self.radius, self.radius, 2 * self.radius + 1)
            delta = torch.stack(
                torch.meshgrid(dy, dx, indexing='ij'),
                axis=-1
            ).to(coordinates.device)
            level_centroid = coordinates.reshape(B * S * N, 1, 1, 2) / 2 ** i
            level_delta = delta.view(1, 2 * self.radius + 1, 2 * self.radius + 1, 2)
            level_coordinates = level_centroid + level_delta

            level_correlations = self.sample_correlations_from_coordinates(
                level_correlations.reshape(B * S * N, 1, H, W),
                level_coordinates
            )
            level_correlations = level_correlations.view(B, S, N, -1)
            out_pyramid.append(level_correlations)
        out = torch.cat(out_pyramid, dim=-1)
        return out.contiguous()

    @staticmethod
    def sample_correlations_from_coordinates(correlations, coordinates):
        H, W = correlations.shape[-2:]
        x_grid, y_grid = coordinates.split([1, 1], dim=-1)
        x_grid = 2 * x_grid / (W - 1) - 1
        y_grid = 2 * y_grid / (H - 1) - 1

        grid = torch.cat([x_grid, y_grid], dim=-1)
        correlations = F.grid_sample(correlations, grid, align_corners=True)
        return correlations
