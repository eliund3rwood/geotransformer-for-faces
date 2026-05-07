"""
Inspect CrossAttentionRegressor: tensor shapes and weight stats at each step.

Two encoder variants are compared side-by-side:
  - PointNetPP  : FPS -> kNN -> shared MLP -> max-pool  (current branch)
  - SimpleMLP   : per-point MLP only                    (updated_scaleinv branch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch3d.ops import sample_farthest_points, knn_points
    HAS_P3D = True
except ImportError:
    HAS_P3D = False
    print("[warn] pytorch3d not available — PointNetPP encoder will be skipped")

# ---------------------------------------------------------------------------
# Model definitions (self-contained, no import from model.py)
# ---------------------------------------------------------------------------

class PointNetPPEncoder(nn.Module):
    def __init__(self, feature_dim=256, num_sampled_points=128, k_neighbors=32):
        super().__init__()
        self.num_sampled_points = num_sampled_points
        self.k_neighbors = k_neighbors
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            nn.ReLU(),
        )

    def forward(self, coords, hooks=None):
        B, N, _ = coords.shape
        _h(hooks, "encoder_input", coords)

        centroids, _ = sample_farthest_points(coords, K=self.num_sampled_points)
        _h(hooks, "fps_centroids", centroids)

        knn_result = knn_points(centroids, coords, K=self.k_neighbors)
        knn_idx = knn_result.idx
        _h(hooks, "knn_idx", knn_idx)

        S, k = self.num_sampled_points, self.k_neighbors
        idx_flat = knn_idx.reshape(B, S * k)
        neighbors = torch.gather(coords, 1, idx_flat.unsqueeze(-1).expand(-1, -1, 3))
        neighbors = neighbors.reshape(B, S, k, 3)
        _h(hooks, "neighbors", neighbors)

        rel_coords = neighbors - centroids.unsqueeze(2)
        _h(hooks, "rel_coords", rel_coords)

        feats = self.mlp(rel_coords)
        _h(hooks, "mlp_out_pre_pool", feats)

        feats, _ = feats.max(dim=2)
        _h(hooks, "encoder_output", feats)

        return feats


class SimpleMLP(nn.Module):
    """Per-point MLP from updated_scaleinv branch."""
    def __init__(self, feature_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

    def forward(self, coords, hooks=None):
        _h(hooks, "encoder_input", coords)
        feats = self.mlp(coords)
        _h(hooks, "encoder_output", feats)
        return feats


class CrossAttentionRegressor(nn.Module):
    def __init__(
        self,
        feature_dim=256,
        num_patches=32,
        num_coeffs=100,
        nhead=4,
        num_layers=2,
        encoder="pointnetpp",        # "pointnetpp" | "simple_mlp"
        num_sampled_points=128,
        k_neighbors=32,
        dropout=0.1,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.feature_dim = feature_dim
        self.encoder_type = encoder

        self.patch_tokens = nn.Parameter(torch.randn(1, num_patches, feature_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=feature_dim * 2,
            batch_first=True,
            norm_first=True,
            dropout=dropout,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        if encoder == "pointnetpp":
            assert HAS_P3D, "pytorch3d required for PointNetPP encoder"
            self.point_encoder = PointNetPPEncoder(
                feature_dim=feature_dim,
                num_sampled_points=num_sampled_points,
                k_neighbors=k_neighbors,
            )
        else:
            self.point_encoder = SimpleMLP(feature_dim=feature_dim)

        self.output_proj = nn.Linear(feature_dim, num_coeffs + 1)
        with torch.no_grad():
            nn.init.constant_(self.output_proj.bias[0], 0.0)

    def forward(self, src_coords, src_padding_mask=None, hooks=None):
        _h(hooks, "regressor_input", src_coords)

        src_feats = self.point_encoder(src_coords, hooks=hooks)
        _h(hooks, "encoded_memory", src_feats)

        B = src_coords.shape[0]
        tokens = self.patch_tokens.expand(B, -1, -1)
        _h(hooks, "patch_tokens", tokens)

        updated_tokens = self.transformer_decoder(
            tgt=tokens,
            memory=src_feats,
            memory_key_padding_mask=src_padding_mask if (
                self.encoder_type == "simple_mlp" and src_padding_mask is not None
            ) else None,
        )
        _h(hooks, "decoder_output", updated_tokens)

        raw_output = self.output_proj(updated_tokens)
        _h(hooks, "proj_output", raw_output)

        scale_logits = raw_output[:, :, 0].mean(dim=1)
        pred_scale = 0.4 + 1.2 * torch.sigmoid(scale_logits)
        _h(hooks, "pred_scale", pred_scale)

        coeffs = raw_output[:, :, 1:]
        _h(hooks, "pred_coeffs", coeffs)

        return coeffs, pred_scale


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _h(hooks, name, tensor):
    """Store tensor in hooks dict if provided."""
    if hooks is not None:
        hooks[name] = tensor.detach()


def _stats(t):
    t = t.float()
    return f"min={t.min():.4f}  max={t.max():.4f}  mean={t.mean():.4f}  std={t.std():.4f}"


def print_tensor_info(name, tensor, indent=2):
    pad = " " * indent
    print(f"{pad}{name:30s}  shape={list(tensor.shape)}  {_stats(tensor)}")


def print_weight_info(name, param, indent=4):
    pad = " " * indent
    print(f"{pad}{name:50s}  shape={list(param.shape)}  {_stats(param.data)}")


def inspect_weights(model, label):
    print(f"\n{'='*70}")
    print(f"  WEIGHTS — {label}")
    print(f"{'='*70}")
    for name, param in model.named_parameters():
        print_weight_info(name, param)


def inspect_forward(model, coords, label):
    print(f"\n{'='*70}")
    print(f"  FORWARD PASS SHAPES — {label}")
    print(f"{'='*70}")
    hooks = {}
    padding_mask = torch.zeros((coords.shape[0], coords.shape[1]), dtype=torch.bool, device=coords.device)
    with torch.no_grad():
        coeffs, scale = model(coords, src_padding_mask=padding_mask, hooks=hooks)

    for step_name, tensor in hooks.items():
        print_tensor_info(step_name, tensor)

    print(f"\n  Final outputs:")
    print_tensor_info("coeffs", coeffs)
    print_tensor_info("pred_scale", scale)
    return coeffs, scale


def param_count(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    B, N = 1, 1024  # batch=1, 1024 input points
    coords = torch.randn(B, N, 3, device=device)

    configs = []

    if HAS_P3D:
        configs.append(dict(
            label="CrossAttentionRegressor  [PointNetPP encoder]",
            encoder="pointnetpp",
            feature_dim=256,
            num_patches=32,
            num_coeffs=100,
            nhead=4,
            num_layers=2,
            num_sampled_points=128,
            k_neighbors=32,
            dropout=0.1,
        ))

    configs.append(dict(
        label="CrossAttentionRegressor  [SimpleMLP encoder  — updated_scaleinv]",
        encoder="simple_mlp",
        feature_dim=256,
        num_patches=32,
        num_coeffs=100,
        nhead=4,
        num_layers=2,
        num_sampled_points=128,
        k_neighbors=32,
        dropout=0.1,
    ))

    for cfg in configs:
        label = cfg.pop("label")
        model = CrossAttentionRegressor(**cfg).to(device).eval()

        print(f"\n{'#'*70}")
        print(f"  {label}")
        print(f"  Total params: {param_count(model):,}")
        print(f"{'#'*70}")

        inspect_weights(model, label)
        inspect_forward(model, coords, label)


if __name__ == "__main__":
    main()
