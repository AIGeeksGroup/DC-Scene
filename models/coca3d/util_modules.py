import torch
import torch.nn as nn
import torch.nn.functional as F
from third_party.pointnet2 import pointnet2_utils
class EmbeddingEncoder(nn.Module):
    def __init__(self, encoder_channel, in_channels=512):
        super().__init__()
        self.in_channels = in_channels
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(in_channels, 128, 1),  # Modified from hardcoded 512
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        # Maintain architecture with concatenated global+local features (256+256=512)
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),  # This 512 is from concatenation, not input features
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N C
            -----------------
            feature_global : B G C
        '''
        bs, g, n, c = point_groups.shape

        if torch.isnan(point_groups).any():
            point_groups = torch.nan_to_num(point_groups, nan=0.0)

        # Verify input channel compatibility
        assert c == self.in_channels, f"Input channels {c} != expected {self.in_channels}"

        # Reshape for processing
        point_groups = point_groups.reshape(bs * g, n, self.in_channels).transpose(2, 1)

        # Forward pass with dynamic feature dimensions
        feature = self.first_conv(point_groups)  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1

        # Concatenation creates fixed 512 dims regardless of input feature dims
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG encoder_channel n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG encoder_channel

        return feature_global.reshape(bs, g, self.encoder_channel)

class SparseGroup(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group, self.group_size = num_group, group_size

    def forward(self, xyz, feature=None):
        if feature is None or xyz.numel() == 0:
            raise ValueError("Sparse grouping needs nonempty coordinates and features")
        neighborhoods, centers, indices, originals, features = [], [], [], [], []
        for batch_id in xyz[:, 0].unique(sorted=True):
            mask = xyz[:, 0] == batch_id
            points = xyz[mask, 1:4].to(feature.device).float().unsqueeze(0)
            values = feature[mask.to(feature.device)]
            n = points.shape[1]
            if n == 0:
                raise ValueError("Empty sparse scene")
            with torch.no_grad():
                idx = pointnet2_utils.furthest_point_sample(points.contiguous(), min(n, self.num_group)).long()
                if idx.shape[1] < self.num_group:
                    idx = torch.cat([idx, idx[:, -1:].expand(1, self.num_group - idx.shape[1])], 1)
                center = points[:, idx[0]]
                near = torch.cdist(center, points).topk(min(n, self.group_size), largest=False).indices
                if near.shape[-1] < self.group_size:
                    near = torch.cat([near, near[..., -1:].expand(1, self.num_group, self.group_size - near.shape[-1])], -1)
            centers.append(center)
            indices.append(idx)
            neighborhoods.append(points[0][near])
            originals.append(points)
            features.append(values[near])
        return torch.cat(neighborhoods), torch.cat(centers), torch.cat(indices), originals, torch.cat(features)


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(
                nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        # Coordinates/features are [B,N,C]; output is [B,D,N].
        k = min(3, xyz2.shape[1])
        distance, indices = torch.cdist(xyz1.float(), xyz2.float()).topk(k, largest=False)
        reciprocal = distance.clamp_min(1e-8).reciprocal()
        weight = reciprocal / reciprocal.sum(-1, keepdim=True)
        batches = torch.arange(points2.shape[0], device=points2.device)[:, None, None]
        interpolated = (points2[batches, indices] * weight[..., None]).sum(2)
        values = torch.cat([points1, interpolated], -1) if points1 is not None else interpolated
        values = values.transpose(1, 2)
        for conv, norm in zip(self.mlp_convs, self.mlp_bns):
            values = F.relu(norm(conv(values)))
        return values
