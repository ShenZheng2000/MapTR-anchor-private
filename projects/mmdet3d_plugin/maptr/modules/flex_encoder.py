# projects/mmdet3d_plugin/maptr/modules/flex_encoder.py
#
# Geometry-agnostic Flex scene encoder, drop-in for the LSSTransform slot
# in MapTRPerceptionTransformer. Produces K unordered 1D scene tokens.
#
# Hand-rolled pre-norm transformer to stay compatible with PyTorch 1.9.x
# (nn.TransformerEncoderLayer gained norm_first / batch_first only in 1.11).

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from mmcv.runner import BaseModule
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE


class _PreNormLayer(nn.Module):
    """Single pre-norm (norm -> attn/ffn -> residual) transformer encoder block.
    Equivalent to nn.TransformerEncoderLayer(..., norm_first=True, batch_first=True)
    but works on PyTorch >= 1.9."""

    def __init__(self, embed_dims, num_heads, ffn_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        # batch_first=True supported since PyTorch 1.9
        self.attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dims, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dims)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # pre-norm self-attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + self.drop(h)
        # pre-norm FFN with GELU
        h = self.norm2(x)
        h = self.linear2(self.drop(F.gelu(self.linear1(h))))
        x = x + self.drop(h)
        return x


class _PreNormEncoder(nn.Module):
    def __init__(self, layer, num_layers, use_checkpoint=True):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(num_layers)])
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        for layer in self.layers:
            if self.use_checkpoint and x.requires_grad:
                x = checkpoint(layer, x)
            else:
                x = layer(x)
        return x


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class FlexSceneEncoder(BaseModule):
    """Flex: jointly encode all camera image tokens into K learnable
    scene tokens via full self-attention. No 3D priors, no BEV grid,
    no camera pose, no depth.

    Output: dict(bev=(B, K, C), depth=None)
        The 'bev' key name is kept only so the surrounding
        MapTRPerceptionTransformer code does not need renaming.
        It is NOT a BEV grid -- it is K unordered scene tokens.
    """

    def __init__(self,
                 num_scene_tokens=900,
                 embed_dims=256,
                 num_layers=8,
                 num_heads=8,
                 num_cams=6,
                 num_timesteps=1,
                 ffn_ratio=4,
                 dropout=0.1,
                 feat_down_sample_indice=-1,
                 pool_stride=1,
                 max_spatial_h=50,
                 max_spatial_w=100,
                 **kwargs):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.num_scene_tokens = num_scene_tokens
        self.num_cams = num_cams
        self.num_timesteps = num_timesteps
        self.feat_down_sample_indice = feat_down_sample_indice
        self.pool_stride = pool_stride

        # K learnable scene tokens (the queries / latent array)
        self.scene_tokens = nn.Parameter(
            torch.randn(num_scene_tokens, embed_dims) * 0.02)
        # learnable per-camera embedding, indexed by camera id
        self.cams_embeds = nn.Parameter(
            torch.randn(num_cams, embed_dims) * 0.02)
        # learnable per-timestep embedding (PE_t^time in the paper, Sec. III-B)
        self.times_embeds = nn.Parameter(
            torch.randn(num_timesteps, embed_dims) * 0.02)
        # factorized 2D spatial positional embedding (ViT-style, but for ResNet tokens)
        # split embed_dims across H and W axes then concat: (H,C/2) x (W,C/2) -> (H*W,C)
        self.spatial_embeds_h = nn.Parameter(
            torch.randn(max_spatial_h, embed_dims // 2) * 0.02)
        self.spatial_embeds_w = nn.Parameter(
            torch.randn(max_spatial_w, embed_dims // 2) * 0.02)

        layer = _PreNormLayer(
            embed_dims=embed_dims,
            num_heads=num_heads,
            ffn_dim=embed_dims * ffn_ratio,
            dropout=dropout,
        )
        self.encoder = _PreNormEncoder(layer, num_layers)
        # pre-norm stacks don't normalize after the last layer; without this
        # the residual stream grows unbounded and produces NaN after ~tens of steps
        self.out_norm = nn.LayerNorm(embed_dims)

    def forward(self, images, img_metas=None, **kwargs):
        # images: (B, N_cam*T, C, H, W) -- N_cam cameras, T timesteps stacked
        B, N, C, H, W = images.shape
        assert C == self.embed_dims, \
            f'feature channel {C} != embed_dims {self.embed_dims}'
        assert N % self.num_cams == 0, \
            f'N={N} must be divisible by num_cams={self.num_cams}'
        T = N // self.num_cams
        assert T <= self.num_timesteps, \
            f'derived T={T} exceeds num_timesteps={self.num_timesteps}'

        # optional spatial pooling to reduce image token count
        if self.pool_stride > 1:
            images = images.view(B * N, C, H, W)
            images = F.avg_pool2d(images, self.pool_stride)
            _, _, H, W = images.shape
            images = images.view(B, N, C, H, W)

        # flatten each image to tokens: (B, N, H*W, C)
        x = images.flatten(3).permute(0, 1, 3, 2).contiguous()
        # build factorized 2D spatial PE: concat(h_embed, w_embed) -> (H*W, C)
        h_emb = self.spatial_embeds_h[:H, :]                 # (H, C/2)
        w_emb = self.spatial_embeds_w[:W, :]                 # (W, C/2)
        # outer product over H and W positions
        spatial_pe = torch.cat([
            h_emb[:, None, :].expand(H, W, -1),              # (H, W, C/2)
            w_emb[None, :, :].expand(H, W, -1),              # (H, W, C/2)
        ], dim=-1).reshape(H * W, C)                         # (H*W, C)

        # reshape to (B, T, N_cam, H*W, C) to apply embeddings per axis
        x = x.view(B, T, self.num_cams, H * W, C)
        # add learnable camera embedding (PE_c^cam, broadcast over T and H*W)
        x = x + self.cams_embeds[None, None, :, None, :].to(x.dtype)
        # add learnable timestep embedding (PE_t^time, broadcast over N_cam and H*W)
        x = x + self.times_embeds[None, :T, None, None, :].to(x.dtype)
        # add 2D spatial PE (broadcast over B, T, N_cam)
        x = x + spatial_pe[None, None, None, :, :].to(x.dtype)
        # flatten back to one image-token sequence
        x = x.reshape(B, N * H * W, C)                       # (B, L_img, C)

        # prepend the K scene tokens  (Eq. 2: [S^(0); X])
        s = self.scene_tokens[None].expand(B, -1, -1)        # (B, K, C)
        seq = torch.cat([s, x], dim=1)                       # (B, K+L_img, C)

        # full joint self-attention over scene + image tokens
        seq = self.encoder(seq)

        # discard image tokens, keep only updated scene tokens  (S^(L))
        scene = seq[:, :self.num_scene_tokens, :].contiguous()  # (B, K, C)
        scene = self.out_norm(scene)

        return dict(bev=scene, depth=None)
