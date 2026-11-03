"""Temporal failure model (DESIGN.md §6).

frames (B, 4, 3, H, W), slot 0 newest -> per-type presence logits (B, 4)
and per-type patch heatmap logits (B, 4, gh, gw) for the newest frame.

A pretrained DINOv2-S/14 (with registers) encodes every frame; its first
`frozen_blocks` blocks run without gradients. Fusion blocks then alternate
attention across time at each patch (the cameras are fixed, so a patch
location is a stable place in the scene) and across space within a frame.
Missing history slots are masked out of temporal attention, never seen as
black frames.
"""

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

N_TYPES = 4
N_SLOTS = 4


class Mlp(nn.Sequential):
    def __init__(self, dim, hidden, out=None):
        super().__init__(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, out or dim))


class Attention(nn.Module):
    """Multi-head attention; queries and keys/values may differ."""

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, ctx=None, mask=None):
        ctx = x if ctx is None else ctx
        B, N, D = x.shape
        h = self.heads
        q = self.q(x).view(B, N, h, D // h).transpose(1, 2)
        k, v = self.kv(ctx).view(B, ctx.shape[1], 2, h, D // h).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(x.transpose(1, 2).reshape(B, N, D))


class FusionBlock(nn.Module):
    """Newest-frame tokens attend to their own patch across all slots
    (time), then to each other (space), then an MLP. History tokens are
    context only, so cost and memory scale with one frame, not four."""

    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.n1, self.nc, self.time = nn.LayerNorm(dim), nn.LayerNorm(dim), Attention(dim, heads)
        self.n2, self.space = nn.LayerNorm(dim), Attention(dim, heads)
        self.n3, self.mlp = nn.LayerNorm(dim), Mlp(dim, dim * mlp_ratio)
        # Start as identity so the pretrained features pass through unchanged.
        for lin in (self.time.proj, self.space.proj, self.mlp[2]):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x, hist, time_mask):
        # x (B, P, D) newest; hist (B, S-1, P, D); time_mask (B*P, 1, 1, S) bool.
        B, P, D = x.shape
        q = x.reshape(B * P, 1, D)
        ctx = torch.cat([q, hist.permute(0, 2, 1, 3).reshape(B * P, -1, D)], 1)
        q = q + self.time(self.n1(q), self.nc(ctx), time_mask)
        x = q.view(B, P, D)
        x = x + self.space(self.n2(x))
        return x + self.mlp(self.n3(x))


class TemporalFailureNet(nn.Module):
    def __init__(self, encoder="vit_small_patch14_reg4_dinov2.lvd142m", frozen_blocks=6,
                 fusion_blocks=2, pretrained=True, grad_checkpoint=False, history_grad=False):
        super().__init__()
        # Without history_grad, history frames pass through the (shared,
        # fine-tuned) encoder without gradients: the fusion blocks still learn
        # from them fully, but encoder backward and activation memory cover
        # the newest frame only (~4x less).
        self.history_grad = history_grad
        self.encoder = timm.create_model(encoder, pretrained=pretrained, num_classes=0,
                                         dynamic_img_size=True)
        self.frozen_blocks = frozen_blocks
        self.grad_checkpoint = grad_checkpoint
        enc = self.encoder
        for p in enc.patch_embed.parameters():
            p.requires_grad = frozen_blocks == 0
        for name in ("cls_token", "reg_token", "pos_embed"):
            if getattr(enc, name, None) is not None:
                getattr(enc, name).requires_grad = frozen_blocks == 0
        for blk in enc.blocks[:frozen_blocks]:
            for p in blk.parameters():
                p.requires_grad = False
        D = enc.embed_dim
        heads = max(1, D // 64)
        self.prefix = enc.num_prefix_tokens
        self.slot_embed = nn.Parameter(torch.zeros(N_SLOTS, D))
        self.gap_embed = Mlp(1, 64, D)
        self.telem_embed = Mlp(5, 128, D)
        self.fusion = nn.ModuleList(FusionBlock(D, heads) for _ in range(fusion_blocks))
        self.norm = nn.LayerNorm(D)
        self.heat = nn.Linear(D, N_TYPES)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, D))
        self.pool = nn.MultiheadAttention(D, heads, batch_first=True)
        self.cls_head = Mlp(2 * D, D, N_TYPES)
        nn.init.normal_(self.slot_embed, std=0.02)
        nn.init.normal_(self.pool_query, std=0.02)
        # Failures are rare: start presence and heatmaps near p = 0.01.
        bias = -math.log(99.0)
        nn.init.constant_(self.heat.bias, bias)
        nn.init.constant_(self.cls_head[2].bias, bias)

    def encode(self, x):
        """(N, 3, H, W) -> prefix tokens (N, prefix, D), patch tokens (N, P, D)."""
        enc = self.encoder
        x = enc.patch_embed(x)
        x = enc._pos_embed(x)
        x = enc.norm_pre(x)
        if self.frozen_blocks:
            with torch.no_grad():
                for blk in enc.blocks[:self.frozen_blocks]:
                    x = blk(x)
            x = x.detach()
        for blk in enc.blocks[self.frozen_blocks:]:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = enc.norm(x)
        return x[:, :self.prefix], x[:, self.prefix:]

    def forward(self, frames, valid, gaps, telem, has_telem):
        B, S, C, H, W = frames.shape
        gh, gw = H // 14, W // 14
        if self.history_grad or not self.training:
            prefix, tokens = self.encode(frames.flatten(0, 1))
        else:
            p0, t0 = self.encode(frames[:, 0])
            with torch.no_grad():
                ph, th = self.encode(frames[:, 1:].flatten(0, 1))
            prefix = torch.cat([p0[:, None], ph.view(B, S - 1, *ph.shape[1:])], 1).flatten(0, 1)
            tokens = torch.cat([t0[:, None], th.view(B, S - 1, *th.shape[1:])], 1).flatten(0, 1)
        D = tokens.shape[-1]
        x = tokens.view(B, S, gh * gw, D)
        cond = (self.slot_embed[None]
                + self.gap_embed(torch.log1p(gaps)[..., None] / 6.0)
                + self.telem_embed(torch.cat([telem, has_telem[:, None]], -1))[:, None])
        x = x + cond[:, :, None]
        # Newest attends only to valid slots (slot 0 is always valid).
        allow = valid.repeat_interleave(gh * gw, 0)[:, None, None, :]
        newest, hist = x[:, 0], x[:, 1:]
        for blk in self.fusion:
            newest = blk(newest, hist, allow)
        # Heads in fp32: bf16 logits saturate and tie, which breaks ranking
        # metrics and threshold calibration at very low false-alarm rates.
        with torch.autocast("cuda", enabled=False):
            newest = self.norm(newest.float())  # (B, P, D)
            heat = self.heat(newest).transpose(1, 2).reshape(B, N_TYPES, gh, gw)
            pooled, _ = self.pool(self.pool_query.expand(B, -1, -1), newest, newest,
                                  need_weights=False)
            cls = prefix.view(B, S, self.prefix, D)[:, 0, 0].float()
            logits = self.cls_head(torch.cat([pooled[:, 0], cls], -1))
        return logits, heat

    def param_groups(self, lr, encoder_lr_scale=0.1, layer_decay=0.8, weight_decay=0.05):
        """Encoder blocks get lr * scale * decay^(depth from top); new
        modules get lr. No weight decay on norms, biases and embeddings."""
        groups = {}
        n_blocks = len(self.encoder.blocks)
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("encoder."):
                if name.startswith("encoder.blocks."):
                    depth = n_blocks - int(name.split(".")[2])
                elif name.startswith("encoder.norm."):
                    depth = 0  # final norm, directly under the heads
                else:
                    depth = n_blocks + 1
                scale = encoder_lr_scale * layer_decay ** (depth - 1)
            else:
                scale = 1.0
            no_wd = p.ndim < 2 or "embed" in name or "token" in name or "query" in name
            key = (scale, no_wd)
            g = groups.setdefault(key, {"params": [], "lr": lr * scale, "lr_scale": scale,
                                        "weight_decay": 0.0 if no_wd else weight_decay})
            g["params"].append(p)
        return list(groups.values())
