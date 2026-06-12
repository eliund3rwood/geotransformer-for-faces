r"""KPConv FPN with per-cloud GroupNorm.

The library GroupNorm reshapes (N, C) -> (1, C, N), so its statistics pool over
the ENTIRE stacked point set: in the original single-pair model, ref and src
features couple through the norm stats. With shared-ref batching that coupling
would make every sample depend on its batchmates, so here normalization is
applied per cloud (per segment of the stack, via `lengths`). Parameter names
match the library blocks exactly, so original checkpoints still load (with a
small, expected distribution shift from the changed norm semantics).
"""
import torch
import torch.nn as nn

from geotransformer.modules.kpconv import KPConv, LastUnaryBlock, nearest_upsample
from geotransformer.modules.kpconv.functional import maxpool


class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels):
        super(GroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.norm = nn.GroupNorm(self.num_groups, self.num_channels)

    def _normalize(self, seg):
        # Same statistics as nn.GroupNorm on (1, C, n) — mean/var over the
        # (C/G x n) values of each group — but computed manually so degenerate
        # single-point clouds (1 value per group) don't raise. There the var is 0
        # and the output reduces to the bias, which is the eps-limit of GroupNorm.
        n = seg.shape[0]
        v = seg.view(n, self.num_groups, self.num_channels // self.num_groups)
        mean = v.mean(dim=(0, 2), keepdim=True)
        var = v.var(dim=(0, 2), unbiased=False, keepdim=True)
        v = (v - mean) / torch.sqrt(var + self.norm.eps)
        return v.view(n, self.num_channels) * self.norm.weight + self.norm.bias

    def forward(self, x, lengths=None):
        # x: (N, C); lengths: per-cloud sizes summing to N (None = single cloud)
        if lengths is None:
            return self._normalize(x)
        outputs = []
        start = 0
        for n in lengths.tolist():
            outputs.append(self._normalize(x[start:start + n]))
            start += n
        return torch.cat(outputs, dim=0)


class UnaryBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group_norm, has_relu=True, bias=True):
        super(UnaryBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_norm = group_norm
        self.mlp = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = GroupNorm(group_norm, out_channels)
        if has_relu:
            self.leaky_relu = nn.LeakyReLU(0.1)
        else:
            self.leaky_relu = None

    def forward(self, x, lengths=None):
        x = self.mlp(x)
        x = self.norm(x, lengths)
        if self.leaky_relu is not None:
            x = self.leaky_relu(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, radius, sigma, group_norm,
                 negative_slope=0.1, bias=True):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.KPConv = KPConv(in_channels, out_channels, kernel_size, radius, sigma, bias=bias)
        self.norm = GroupNorm(group_norm, out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, s_feats, q_points, s_points, neighbor_indices, q_lengths=None):
        x = self.KPConv(s_feats, q_points, s_points, neighbor_indices)
        x = self.norm(x, q_lengths)
        x = self.leaky_relu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, radius, sigma, group_norm,
                 strided=False, bias=True):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strided = strided

        mid_channels = out_channels // 4

        if in_channels != mid_channels:
            self.unary1 = UnaryBlock(in_channels, mid_channels, group_norm, bias=bias)
        else:
            self.unary1 = nn.Identity()

        self.KPConv = KPConv(mid_channels, mid_channels, kernel_size, radius, sigma, bias=bias)
        self.norm_conv = GroupNorm(group_norm, mid_channels)

        self.unary2 = UnaryBlock(mid_channels, out_channels, group_norm, has_relu=False, bias=bias)

        if in_channels != out_channels:
            self.unary_shortcut = UnaryBlock(in_channels, out_channels, group_norm, has_relu=False, bias=bias)
        else:
            self.unary_shortcut = nn.Identity()

        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, s_feats, q_points, s_points, neighbor_indices, s_lengths=None, q_lengths=None):
        if isinstance(self.unary1, nn.Identity):
            x = self.unary1(s_feats)
        else:
            x = self.unary1(s_feats, s_lengths)

        x = self.KPConv(x, q_points, s_points, neighbor_indices)
        x = self.norm_conv(x, q_lengths)
        x = self.leaky_relu(x)

        x = self.unary2(x, q_lengths)

        if self.strided:
            shortcut = maxpool(s_feats, neighbor_indices)
        else:
            shortcut = s_feats
        if isinstance(self.unary_shortcut, nn.Identity):
            shortcut = self.unary_shortcut(shortcut)
        else:
            shortcut = self.unary_shortcut(shortcut, q_lengths)

        x = x + shortcut
        x = self.leaky_relu(x)

        return x


class KPConvFPN(nn.Module):
    def __init__(self, input_dim, output_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm):
        super(KPConvFPN, self).__init__()

        self.encoder1_1 = ConvBlock(input_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm)
        self.encoder1_2 = ResidualBlock(init_dim, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm)

        self.encoder2_1 = ResidualBlock(
            init_dim * 2, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm, strided=True
        )
        self.encoder2_2 = ResidualBlock(
            init_dim * 2, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )
        self.encoder2_3 = ResidualBlock(
            init_dim * 4, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )

        self.encoder3_1 = ResidualBlock(
            init_dim * 4, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm, strided=True
        )
        self.encoder3_2 = ResidualBlock(
            init_dim * 4, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )
        self.encoder3_3 = ResidualBlock(
            init_dim * 8, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )

        self.encoder4_1 = ResidualBlock(
            init_dim * 8, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm, strided=True
        )
        self.encoder4_2 = ResidualBlock(
            init_dim * 8, init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8, group_norm
        )
        self.encoder4_3 = ResidualBlock(
            init_dim * 16, init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8, group_norm
        )

        self.decoder3 = UnaryBlock(init_dim * 24, init_dim * 8, group_norm)
        self.decoder2 = LastUnaryBlock(init_dim * 12, output_dim)

    def forward(self, feats, data_dict):
        feats_list = []

        points_list = data_dict['points']
        neighbors_list = data_dict['neighbors']
        subsampling_list = data_dict['subsampling']
        upsampling_list = data_dict['upsampling']
        lengths_list = data_dict['lengths']

        feats_s1 = feats
        feats_s1 = self.encoder1_1(feats_s1, points_list[0], points_list[0], neighbors_list[0],
                                   q_lengths=lengths_list[0])
        feats_s1 = self.encoder1_2(feats_s1, points_list[0], points_list[0], neighbors_list[0],
                                   s_lengths=lengths_list[0], q_lengths=lengths_list[0])

        feats_s2 = self.encoder2_1(feats_s1, points_list[1], points_list[0], subsampling_list[0],
                                   s_lengths=lengths_list[0], q_lengths=lengths_list[1])
        feats_s2 = self.encoder2_2(feats_s2, points_list[1], points_list[1], neighbors_list[1],
                                   s_lengths=lengths_list[1], q_lengths=lengths_list[1])
        feats_s2 = self.encoder2_3(feats_s2, points_list[1], points_list[1], neighbors_list[1],
                                   s_lengths=lengths_list[1], q_lengths=lengths_list[1])

        feats_s3 = self.encoder3_1(feats_s2, points_list[2], points_list[1], subsampling_list[1],
                                   s_lengths=lengths_list[1], q_lengths=lengths_list[2])
        feats_s3 = self.encoder3_2(feats_s3, points_list[2], points_list[2], neighbors_list[2],
                                   s_lengths=lengths_list[2], q_lengths=lengths_list[2])
        feats_s3 = self.encoder3_3(feats_s3, points_list[2], points_list[2], neighbors_list[2],
                                   s_lengths=lengths_list[2], q_lengths=lengths_list[2])

        feats_s4 = self.encoder4_1(feats_s3, points_list[3], points_list[2], subsampling_list[2],
                                   s_lengths=lengths_list[2], q_lengths=lengths_list[3])
        feats_s4 = self.encoder4_2(feats_s4, points_list[3], points_list[3], neighbors_list[3],
                                   s_lengths=lengths_list[3], q_lengths=lengths_list[3])
        feats_s4 = self.encoder4_3(feats_s4, points_list[3], points_list[3], neighbors_list[3],
                                   s_lengths=lengths_list[3], q_lengths=lengths_list[3])

        latent_s4 = feats_s4
        feats_list.append(feats_s4)

        latent_s3 = nearest_upsample(latent_s4, upsampling_list[2])
        latent_s3 = torch.cat([latent_s3, feats_s3], dim=1)
        latent_s3 = self.decoder3(latent_s3, lengths_list[2])
        feats_list.append(latent_s3)

        latent_s2 = nearest_upsample(latent_s3, upsampling_list[1])
        latent_s2 = torch.cat([latent_s2, feats_s2], dim=1)
        latent_s2 = self.decoder2(latent_s2)
        feats_list.append(latent_s2)

        feats_list.reverse()

        return feats_list
