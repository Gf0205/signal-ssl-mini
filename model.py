import argparse
import random

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class IQPatchEmbed(nn.Module):
    """Convert IQ sequence [B, 2, L] into patch tokens [B, P, D]."""

    def __init__(self, seq_len: int = 128, patch_size: int = 8, hidden_dim: int = 64) -> None:
        super().__init__()
        assert seq_len > 0
        assert patch_size > 0
        assert seq_len % patch_size == 0, "seq_len must be divisible by patch_size"

        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.patch_dim = 2 * patch_size
        self.proj = nn.Linear(self.patch_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3, f"Expected x [B, 2, L], got {tuple(x.shape)}"
        assert x.shape[1] == 2, f"Expected IQ channel dim=2, got {tuple(x.shape)}"
        assert x.shape[2] == self.seq_len, f"Expected seq_len={self.seq_len}, got {x.shape[2]}"

        batch_size = x.shape[0]

        # [B, 2, L] -> [B, 2, P, patch_size]
        x = x.reshape(batch_size, 2, self.num_patches, self.patch_size)

        # [B, 2, P, patch_size] -> [B, P, 2, patch_size]
        x = x.permute(0, 2, 1, 3)

        # [B, P, 2, patch_size] -> [B, P, 2 * patch_size]
        x = x.reshape(batch_size, self.num_patches, self.patch_dim)

        # [B, P, patch_dim] -> [B, P, hidden_dim]
        tokens = self.proj(x)
        return tokens


class TinyIQTransformerBackbone(nn.Module):
    """Tiny Transformer encoder backbone for IQ signals."""

    def __init__(
        self,
        seq_len: int = 128,
        patch_size: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.patch_embed = IQPatchEmbed(
            seq_len=seq_len,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
        )
        self.num_patches = self.patch_embed.num_patches
        self.hidden_dim = hidden_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, return_tokens: bool = False) -> torch.Tensor:
        patch_tokens = self.patch_embed(x)
        batch_size = patch_tokens.shape[0]

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_token, patch_tokens], dim=1)
        tokens = tokens + self.pos_embed

        encoded = self.encoder(tokens)
        encoded = self.norm(encoded)

        if return_tokens:
            return encoded
        return encoded[:, 0]


class TinyIQClassifier(nn.Module):
    """Backbone plus a classification head for AMC."""

    def __init__(
        self,
        seq_len: int = 128,
        patch_size: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        num_classes: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = TinyIQTransformerBackbone(
            seq_len=seq_len,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    model = TinyIQClassifier(
        seq_len=args.seq_len,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
    )
    model.eval()

    x = torch.randn(args.batch_size, 2, args.seq_len)

    with torch.no_grad():
        patch_tokens = model.backbone.patch_embed(x)
        encoded_tokens = model.backbone(x, return_tokens=True)
        features = model.backbone(x)
        logits = model(x)

    print("=== Tiny IQ Transformer shape check ===")
    print(f"Input x shape: {tuple(x.shape)}")
    print(f"Patch tokens shape: {tuple(patch_tokens.shape)}")
    print(f"Encoded tokens shape: {tuple(encoded_tokens.shape)}")
    print(f"CLS feature shape: {tuple(features.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    expected_patches = args.seq_len // args.patch_size
    assert patch_tokens.shape == (args.batch_size, expected_patches, args.hidden_dim)
    assert encoded_tokens.shape == (args.batch_size, expected_patches + 1, args.hidden_dim)
    assert features.shape == (args.batch_size, args.hidden_dim)
    assert logits.shape == (args.batch_size, args.num_classes)
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"

    print("\nPASS: model forward sanity checks completed.")


if __name__ == "__main__":
    main()
