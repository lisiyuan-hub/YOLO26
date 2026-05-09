import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ultralytics.nn.modules import C3, C2f

# 使用相对导入，确保 irpe.py 在同一目录下时能被找到
try:
    from .irpe import build_rpe, get_rpe_config
except ImportError:
    from ultralytics.nn.newsAddmodules.irpe import build_rpe, get_rpe_config

# --- 彻底移除 MMDetection 依赖，改用纯 Torch 逻辑 ---
has_mmdet = False
# 如果后续代码中有用到 _load_checkpoint，直接改为 torch.load
# 如果用到 BACKBONES，对于 YOLO 任务来说可以直接忽略
# -----------------------------------------------

__all__ = ["PATConv", "PATConvC3k2"]


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(x + 3.0) / 6.0


def _make_divisible(v, divisor, min_value=None):
    """This function is taken from the original tf repo. It ensures that all layers have a channel number that is
    divisible by 4 It can be seen
    here: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcite(nn.Module):
    def __init__(
        self, in_chs, se_ratio=0.25, reduced_base_chs=None, act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_
    ):
        super().__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class RPEAttention(nn.Module):
    """Attention with image relative position encoding."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, rpe_config=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # image relative position encoding
        self.rpe_q, self.rpe_k, self.rpe_v = build_rpe(rpe_config, head_dim=head_dim, num_heads=num_heads)

    def forward(self, x):
        B, C, h, w = x.shape
        x = x.view(B, C, h * w).transpose(1, 2)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q *= self.scale

        attn = q @ k.transpose(-2, -1)

        # image relative position on keys
        if self.rpe_k is not None:
            # attn += self.rpe_k(q)
            attn += self.rpe_k(q, h, w)
        # image relative position on queries
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale).transpose(2, 3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v

        # image relative position on values
        if self.rpe_v is not None:
            out += self.rpe_v(attn)

        x = out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(1, 2).view(B, C, h, w)
        return x


class SRM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.cfc1 = nn.Conv2d(channel, channel, kernel_size=(1, 2), bias=False)
        # self.cfc2 = nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channel)
        self.sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        b, c, _h, _w = x.shape
        # style pooling
        mean = x.reshape(b, c, -1).mean(-1).view(b, c, 1, 1)
        # 修改 PATConv_AAAI2026.py 中的 SRM 类
        std = x.reshape(b, c, -1).std(-1).view(b, c, 1, 1) + 1e-3
        # max_value = torch.max(x.reshape(b, c, -1), -1)[0].view(b,c,1,1)
        u = torch.cat([mean, std], dim=-1)
        # style integration
        z = self.cfc1(u)
        # z = self.act(z)
        # z = self.cfc2(z)
        # z = self.bn(z)
        g = self.sigmoid(z)
        g = g.reshape(b, c, 1, 1)
        return x * g.expand_as(x)


class PATConv(nn.Module):
    def __init__(
        self, dim, n_div=4, forward_type="split_cat", use_attn=True, channel_type="se", patnet_t0=True
    ):  #'se' if i_stage <= 2 else 'self',
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim = dim
        self.n_div = n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.use_attn = use_attn
        self.channel_type = channel_type

        if use_attn:
            if channel_type == "self":
                self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
                rpe_config = get_rpe_config(
                    ratio=20,
                    method="euc",
                    mode="bias",
                    shared_head=False,
                    skip=0,
                    rpe_on="k",
                )
                if patnet_t0:
                    num_heads = 4
                else:
                    num_heads = 6
                self.attn = RPEAttention(
                    self.dim_untouched, num_heads=num_heads, attn_drop=0.1, proj_drop=0.1, rpe_config=rpe_config
                )
                self.norm = timm.layers.LayerNorm2d(self.dim_untouched)
                # self.norm = timm.layers.LayerNorm2d(self.dim)
                self.forward = self.forward_atten
            elif channel_type == "se":
                self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
                self.attn = SRM(self.dim_untouched)
                self.norm = nn.BatchNorm2d(self.dim_untouched)
                self.forward = self.forward_atten
        else:
            if forward_type == "slicing":
                self.forward = self.forward_slicing
            elif forward_type == "split_cat":
                self.forward = self.forward_split_cat
            else:
                raise NotImplementedError

    def forward_atten(self, x: Tensor) -> Tensor:
        if self.channel_type:
            # print(self.channel_type)
            if self.channel_type == "se":
                x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
                x1 = self.partial_conv3(x1)
                # x = self.partial_conv3(x)
                x2 = self.attn(x2)
                x2 = self.norm(x2)
                x = torch.cat((x1, x2), 1)
                # x = self.attn(x)
            else:
                x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
                x1 = self.partial_conv3(x1)
                x2 = self.norm(x2)
                x2 = self.attn(x2)
                x = torch.cat((x1, x2), 1)
        return x

    def forward_slicing(self, x: Tensor) -> Tensor:
        x1 = x.clone()  # !!! Keep the original input intact for the residual connection later
        x1[:, : self.dim_conv3, :, :] = self.partial_conv3(x1[:, : self.dim_conv3, :, :])
        return x1

    def forward_split_cat(self, x: Tensor) -> Tensor:
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


# YOLOV13改进
class DSConv(nn.Module):
    """The Basic Depthwise Separable Convolution."""

    def __init__(self, c_in, c_out, k=3, s=1, p=None, d=1, bias=False):
        super().__init__()
        if p is None:
            p = (d * (k - 1)) // 2
        self.dw = nn.Conv2d(c_in, c_in, kernel_size=k, stride=s, padding=p, dilation=d, groups=c_in, bias=bias)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1, 0, bias=bias)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return self.act(self.bn(x))


class DSBottleneck_PATConv(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5, k1=3, k2=5, d2=1):
        super().__init__()
        c_ = int(c2 * e)
        # self.cv1 = DSConv(c1, c_, k1, s=1, p=None, d=1)
        # self.cv2 = DSConv(c_, c2, k2, s=1, p=None, d=d2)
        self.cv1 = PATConv(c_)  # 在这里可以替换一种DSConv，也可以都替换。自己可以做一下消融实验
        self.cv2 = PATConv(c_)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class PATConvC3(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k1=3, k2=5, d2=1):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)

        self.m = nn.Sequential(
            *(DSBottleneck_PATConv(c_, c_, shortcut=shortcut, e=1.0, k1=k1, k2=k2, d2=d2) for _ in range(n))
        )


# class DSC3k2(C2f):
class PATConvC3k2(C2f):
    def __init__(self, c1, c2, n=1, dsc3k=False, e=0.5, g=1, shortcut=True, k1=3, k2=7, d2=1):
        super().__init__(c1, c2, n, shortcut, g, e)
        if dsc3k:
            self.m = nn.ModuleList(
                PATConvC3(self.c, self.c, n=2, shortcut=shortcut, g=g, e=1.0, k1=k1, k2=k2, d2=d2) for _ in range(n)
            )
        else:
            self.m = nn.ModuleList(
                DSBottleneck_PATConv(self.c, self.c, shortcut=shortcut, e=1.0, k1=k1, k2=k2, d2=d2) for _ in range(n)
            )


# 输入 N C H W,  输出 N C H W
if __name__ == "__main__":
    # 初始化 PATConv 模块并设定通道维度
    PATConv_module = PATConv(32)
    # 创建一个随机输入张量，假设批量大小为1，通道数为32，图像尺寸为64x64
    input = torch.randn(1, 32, 64, 64)
    # 将输入张量传入 PATConv 模块
    output = PATConv_module(input)
    # 输出结果的形状
    print("PATConv_输入张量的形状：", input.shape)
    print("PATConv_输出张量的形状：", output.shape)
