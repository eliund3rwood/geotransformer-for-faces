import torch
import torch.nn as nn
from pytorch3d.ops import sample_farthest_points, knn_points


def generate_reference_geometry(z_delta, pca_basis, pca_mean, patch_indices):
    """Reconstruct a full reference point cloud from per-patch PCA coefficients.

    Args:
        z_delta        : [num_patches, num_components]
        pca_basis      : [num_patches, num_components, k_neighbors*3]
        pca_mean       : [num_patches, k_neighbors*3]
        patch_indices  : [num_patches, k_neighbors]  (global vertex indices)
    """
    num_patches = patch_indices.shape[0]
    k_neighbors = patch_indices.shape[1]

    delta = torch.matmul(z_delta.unsqueeze(1), pca_basis).squeeze(1)
    reconstructed_patches_flat = pca_mean + delta
    reconstructed_points = reconstructed_patches_flat.view(num_patches, k_neighbors, 3)

    flat_indices = patch_indices.view(-1).long()
    flat_points = reconstructed_points.contiguous().view(-1, 3)
    num_global_verts = flat_indices.max().item() + 1

    # Last-write-wins overlap resolution (higher patch index wins)
    patch_scores = torch.arange(num_patches, device=z_delta.device, dtype=torch.long)
    flat_scores = patch_scores.unsqueeze(1).expand(-1, k_neighbors).reshape(-1)

    max_scores = torch.full((num_global_verts,), -1, device=z_delta.device, dtype=torch.long)
    max_scores.scatter_reduce_(0, flat_indices, flat_scores, reduce='amax', include_self=False)

    is_max_score = flat_scores == max_scores[flat_indices]
    valid_flat_positions = torch.arange(flat_indices.size(0), device=z_delta.device)[is_max_score]
    valid_global_indices = flat_indices[is_max_score]

    best_idx_per_global = torch.zeros(num_global_verts, dtype=torch.long, device=z_delta.device)
    best_idx_per_global.scatter_(0, valid_global_indices, valid_flat_positions)

    ref_points = torch.zeros((num_global_verts, 3), device=z_delta.device, dtype=z_delta.dtype)
    has_points = torch.zeros(num_global_verts, dtype=torch.bool, device=z_delta.device)
    has_points[valid_global_indices] = True
    ref_points[has_points] = flat_points[best_idx_per_global[has_points]]

    return ref_points


class PointNetPPEncoder(nn.Module):
    def __init__(self, feature_dim=1024, num_sampled_points=128, k_neighbors=32):
        super().__init__()
        self.num_sampled_points = num_sampled_points
        self.k_neighbors = k_neighbors

        # Shared MLP applied to relative coords of each neighbor w.r.t. its centroid
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            nn.ReLU(),
        )

    def forward(self, coords):
        # coords: [B, N, 3]
        B, N, _ = coords.shape

        # FPS: pick num_sampled_points representative centroids
        centroids, _ = sample_farthest_points(coords, K=self.num_sampled_points)
        # centroids: [B, S, 3]

        # kNN: for each centroid find k nearest neighbors in the full cloud
        knn_idx = knn_points(centroids, coords, K=self.k_neighbors).idx  # [B, S, k]

        S, k = self.num_sampled_points, self.k_neighbors

        # Gather neighbor coordinates
        idx_flat = knn_idx.reshape(B, S * k)
        neighbors = torch.gather(coords, 1, idx_flat.unsqueeze(-1).expand(-1, -1, 3))
        neighbors = neighbors.reshape(B, S, k, 3)

        # Relative coordinates w.r.t. each centroid — removes global position bias
        rel_coords = neighbors - centroids.unsqueeze(2)  # [B, S, k, 3]

        # Shared MLP (Linear applies to last dim, works on any leading dims)
        feats = self.mlp(rel_coords)  # [B, S, k, feature_dim]

        # PointNet-style max pool over k neighbors
        feats, _ = feats.max(dim=2)  # [B, S, feature_dim]

        return feats


class CrossAttentionRegressor(nn.Module):
    def __init__(
        self,
        feature_dim=128,
        num_patches=32,
        num_coeffs=100,
        nhead=4,
        num_layers=2,
        encoder_type='pointnetpp',
        num_sampled_points=128,
        k_neighbors=32,
        dropout=0.1,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.feature_dim = feature_dim

        # Learnable query tokens — one per patch
        self.patch_tokens = nn.Parameter(torch.randn(1, num_patches, feature_dim))

        # Cross-Attention Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=feature_dim * 2,
            batch_first=True,
            norm_first=True,
            dropout=dropout,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        if encoder_type == 'pointnetpp':
            self.point_encoder = PointNetPPEncoder(
                feature_dim=feature_dim,
                num_sampled_points=num_sampled_points,
                k_neighbors=k_neighbors,
            )
        else:
            self.point_encoder = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, 256),
                nn.ReLU(),
                nn.Linear(256, feature_dim),
            )

        # Final proj: feature_dim -> 100 PCA coeffs + 1 scale
        self.output_proj = nn.Linear(feature_dim, num_coeffs + 1)

        with torch.no_grad():
            nn.init.constant_(self.output_proj.bias[0], 0.0)

    def forward(self, src_coords_padded, src_padding_mask):
        src_feats_encoded = self.point_encoder(src_coords_padded)

        B = src_coords_padded.shape[0]
        tokens = self.patch_tokens.expand(B, -1, -1)

        updated_tokens = self.transformer_decoder(
            tgt=tokens,
            memory=src_feats_encoded,
        )

        raw_output = self.output_proj(updated_tokens)  # [B, num_patches, num_coeffs+1]

        scale_logits = raw_output[:, :, 0].mean(dim=1)
        pred_scale = 0.4 + 1.2 * torch.sigmoid(scale_logits)

        coeffs = raw_output[:, :, 1:]  # [B, num_patches, num_coeffs]

        return coeffs, pred_scale
