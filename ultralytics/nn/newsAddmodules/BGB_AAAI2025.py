from __future__ import annotations

import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_
from torch.nn.init import _calculate_fan_in_and_fan_out

__all__ = ["BGBFusion"]


def get_same_padding(kernel_size: int | tuple[int, ...]) -> int | tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.network_depth = network_depth

        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1, groups=in_features),
            nn.Conv2d(in_features, hidden_features, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_features, out_features, 1),
            nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1, groups=out_features),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


class CAB(nn.Module):
    def __init__(self, dim, num_heads=8, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        _b, _c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


# Intensity Enhancement Layer
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.dwconv1 = nn.Conv2d(
            hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias
        )
        self.dwconv2 = nn.Conv2d(
            hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="withBias"):
        super().__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# Lightweight Cross Attention
class IAM(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        self.gdfn = IEL(dim)  # IEL and CDL have same structure
        self.norm = LayerNorm(dim, LayerNorm_type="withBisa")
        self.ffn = CAB(dim, num_heads, bias)

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = self.gdfn(self.norm(x))
        return x


# Interaction Attention Module
class IAMB(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

        self.proj = nn.Conv2d(self.dim, self.dim, kernel_size=1)

        self.rgb_cab = IAM(self.dim)
        self.ycbcr_cab = IAM(self.dim)

    def forward(self, in_feats):
        out_rgb = self.rgb_cab(in_feats[0], in_feats[1])
        out_ycbcr = self.ycbcr_cab(in_feats[1], in_feats[0])

        x = self.proj(out_ycbcr + out_rgb)
        return x, out_rgb, out_ycbcr


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class MDHTAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, out_dim, kernel_size=1, bias=bias)
        self.act = nn.Tanh()

    def forward(self, x):
        _b, _c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.act(attn)

        out = attn @ v

        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, out_dim, kernel_size=1, bias=bias)
        self.act = nn.Tanh()

    def forward(self, x):
        _b, _c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q, k = F.normalize(q, dim=-2), F.normalize(k, dim=-2)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        out = attn.softmax(dim=-1) @ v

        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


# Phase integration module
class PIM(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.processmag = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0), nn.LeakyReLU(0.1, inplace=True), nn.Conv2d(channel, channel, 1, 1, 0)
        )

        self.processpha = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0), nn.LeakyReLU(0.1, inplace=True), nn.Conv2d(channel, channel, 1, 1, 0)
        )

    def forward(self, rgb_x, ycbcr_x):
        rgb_fft = torch.fft.rfft2(rgb_x, norm="backward")
        ycbcr_fft = torch.fft.rfft2(ycbcr_x, norm="backward")
        rgb_amp = torch.abs(rgb_fft)
        rgb_phase = torch.angle(rgb_fft)

        ycbcr_amp = torch.abs(ycbcr_fft)
        ycbcr_phase = torch.angle(ycbcr_fft)

        rgb_amp = self.processmag(rgb_amp)
        rgb_phase = self.processmag(rgb_phase)

        ycbcr_amp = self.processmag(ycbcr_amp)
        ycbcr_phase = self.processmag(ycbcr_phase)

        mix_phase = rgb_phase + ycbcr_phase

        out_rgb = torch.fft.irfft2(rgb_amp * torch.exp(1j * mix_phase), norm="backward")
        out_ycbcr = torch.fft.irfft2(ycbcr_amp * torch.exp(1j * mix_phase), norm="backward")
        return out_rgb, out_ycbcr


class BGBFusion(nn.Module):  # BI_color_Guidance_Bridge
    def __init__(self, dim):
        super().__init__()

        self.detail_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.smooth_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.iam = IAMB(dim)
        self.detail_attention = SpatialAttention(dim, dim)

        self.color_attention = MDHTAttention(dim, dim)

        self.pim = PIM(dim)

        self.pixel_shuffle = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.rgb_feat_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim // 2), nn.Softmax(dim=1)
        )
        self.ycbcr_feat_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim // 2), nn.Softmax(dim=1)
        )

    def forward(self, in_feats):
        down_feat0 = self.smooth_pool(in_feats[0])
        down_feat1 = self.detail_pool(in_feats[1])
        pre_down_list = [down_feat0, down_feat1]
        if pre_down_list is not None:
            down_feat0 = self.color_attention(down_feat0 * self.rgb_feat_conv(pre_down_list[0]))
            down_feat1 = self.detail_attention(down_feat1 * self.ycbcr_feat_conv(pre_down_list[1]))
        else:
            down_feat0, down_feat1 = self.pim(down_feat0, down_feat1)

        feat0 = self.pixel_shuffle(down_feat0)
        feat1 = self.pixel_shuffle(down_feat1)
        inp_fusion_out, rgb_feat, ycbcr_feat = self.iam([feat0, feat1])

        return inp_fusion_out + rgb_feat + ycbcr_feat
