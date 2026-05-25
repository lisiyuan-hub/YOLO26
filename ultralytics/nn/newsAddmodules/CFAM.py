import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_tf_
from timm.layers.drop import DropPath
from timm.models import named_apply

from ultralytics.nn.modules import C2PSA

from ..modules import C2f
from ..modules.block import C3k, PSABlock

__all__ = ["C2PSA_CFAM", "CFAM", "CFAMblock"]


class Nonlocal(nn.Module):
    """Builds Non-local Neural Networks as a generic family of building blocks for capturing long-range dependencies.
    Non-local Network computes the response at a position as a weighted sum of the features at all positions. This
    building block can be plugged into many computer vision architectures. More details in the
    paper: https://arxiv.org/pdf/1711.07971.pdf.
    """

    def __init__(
        self,
        dim_inner,
        pool_size=None,
        instantiation="softmax",
        zero_init_final_conv=False,
        zero_init_final_norm=True,
        norm_eps=1e-5,
        norm_momentum=0.1,
        norm_module=nn.BatchNorm2d,
    ):
        """
        Args:
            dim (int): number of dimension for the input.
            dim_inner (int): number of dimension inside of the Non-local block.
            pool_size (list): the kernel size of spatial temporal pooling, temporal pool kernel size, spatial pool
                kernel size, spatial pool kernel size in order. By default pool_size is None, then there would be no
                pooling used.
            instantiation (string): supports two different instantiation method:
            "dot_product": normalizing correlation matrix with L2.
            "softmax": normalizing correlation matrix with Softmax.
            zero_init_final_conv (bool): If true, zero initializing the final convolution of the Non-local block.
            zero_init_final_norm (bool): If true, zero initializing the final batch norm of the Non-local block.
            norm_module (nn.Module): nn.Module for the normalization layer. The default is nn.BatchNorm3d.
        """
        super().__init__()
        # self.dim = dim
        self.dim_inner = dim_inner
        self.pool_size = pool_size
        self.instantiation = instantiation
        self.use_pool = False if pool_size is None else any(size > 1 for size in pool_size)
        self.norm_eps = norm_eps
        self.norm_momentum = norm_momentum
        self._construct_nonlocal(zero_init_final_conv, zero_init_final_norm, norm_module)

    def _construct_nonlocal(self, zero_init_final_conv, zero_init_final_norm, norm_module):
        # Three convolution heads: theta, phi, and g.
        self.conv_theta = nn.Conv2d(self.dim_inner, self.dim_inner, kernel_size=1, stride=1, padding=0)
        self.conv_phi = nn.Conv2d(self.dim_inner, self.dim_inner, kernel_size=1, stride=1, padding=0)
        self.conv_g = nn.Conv2d(self.dim_inner, self.dim_inner, kernel_size=1, stride=1, padding=0)

        # Final convolution output.
        self.conv_out = nn.Conv2d(self.dim_inner, self.dim_inner, kernel_size=1, stride=1, padding=0)
        # Zero initializing the final convolution output.
        self.conv_out.zero_init = zero_init_final_conv

        # TODO: change the name to `norm`
        self.bn = norm_module(
            num_features=self.dim_inner,
            eps=self.norm_eps,
            momentum=self.norm_momentum,
        )
        # Zero initializing the final bn.
        self.bn.transform_final_bn = zero_init_final_norm

        self.w = nn.Parameter(torch.tensor(0.5))  # Initial weight

        # Optional to add the spatial-temporal pooling.
        if self.use_pool:
            self.pool = nn.MaxPool2d(
                kernel_size=self.pool_size,
                stride=self.pool_size,
                padding=[0, 0],
            )

    def forward(self, x):
        x_identity = x.clone()
        N, _C, H, W = x.size()

        theta = self.conv_theta(x)

        # print(theta.shape)
        # Perform temporal-spatial pooling to reduce the computation.
        if self.use_pool:
            x = self.pool(x)

        phi = self.conv_phi(x)
        g = self.conv_g(x)

        theta = theta.view(N, self.dim_inner, -1)
        phi = phi.view(N, self.dim_inner, -1)
        g = g.view(N, self.dim_inner, -1)

        # (N, C, HxW) * (N, C, HxW) => (N, HxW, HxW).
        theta_phi = torch.einsum("nch,ncp->nhp", (theta, phi))
        # print(theta_phi.shape)
        # For original Non-local paper, there are two main ways to normalize
        # the affinity tensor:
        #   1) Softmax normalization (norm on exp).
        #   2) dot_product normalization.
        if self.instantiation == "softmax":
            # Normalizing the affinity tensor theta_phi before softmax.
            theta_phi = theta_phi * (self.dim_inner**-0.5)
            theta_phi = nn.functional.softmax(theta_phi, dim=2)
        elif self.instantiation == "dot_product":
            spatial_dim = theta_phi.shape[2]
            theta_phi = theta_phi / spatial_dim
        else:
            raise NotImplementedError(f"Unknown norm type {self.instantiation}")

        # (N, HW, HW) * (N, C, HW) => (N, C, HW)
        theta_phi_g = torch.einsum("nhg,ncg->nch", (theta_phi, g))

        # print(theta_phi_g.shape)
        # (N, C, HxW) => (N, C, H, W).
        theta_phi_g = theta_phi_g.view(N, self.dim_inner, H, W)

        p = self.conv_out(theta_phi_g)
        p = self.bn(p)

        z = (1 - self.w) * x_identity + self.w * p
        return z


# Other types of layers can go here (e.g., nn.Linear, etc.)
def _init_weights(module, name, scheme=""):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == "normal":
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == "trunc_normal":
            trunc_normal_tf_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == "xavier_normal":
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == "kaiming_normal":
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


class SepConvBN(nn.Module):
    def __init__(self, in_channels, filters, kernel_size=3, stride=1, rate=1, depth_activation=False, epsilon=1e-3):
        super().__init__()

        # Calculate padding
        # if stride == 1:
        #     self.padding = kernel_size // 2
        # else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        self.padding = (kernel_size_effective - 1) // 2

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=rate,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(in_channels, eps=epsilon)

        self.pointwise = nn.Conv2d(in_channels, filters, kernel_size=1, stride=1, padding=0, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(filters, eps=epsilon)
        self.depth_activation = depth_activation
        self.init_weights("normal")

    def init_weights(self, scheme=""):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        if not self.depth_activation:
            x = F.relu(x, inplace=True)

        x = self.depthwise(x)
        x = self.depthwise_bn(x)

        if self.depth_activation:
            x = F.relu(x, inplace=True)

        x = self.pointwise(x)
        x = self.pointwise_bn(x)

        if self.depth_activation:
            x = F.relu(x, inplace=True)

        return x


def build_act_layer(act_type):
    """Build activation layer."""
    if act_type is None:
        return nn.Identity()
    assert act_type in ["GELU", "ReLU", "SiLU"]
    if act_type == "SiLU":
        return nn.SiLU()
    elif act_type == "ReLU":
        return nn.ReLU()
    else:
        return nn.GELU()


def build_norm_layer(norm_type, embed_dims):
    """Build normalization layer."""
    assert norm_type in ["BN", "GN", "LN2d", "SyncBN"]
    if norm_type == "GN":
        return nn.GroupNorm(embed_dims, embed_dims, eps=1e-5)
    if norm_type == "LN2d":
        return LayerNorm2d(embed_dims, eps=1e-6)
    if norm_type == "SyncBN":
        return nn.SyncBatchNorm(embed_dims, eps=1e-5)
    else:
        return nn.BatchNorm2d(embed_dims, eps=1e-5)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class LayerNorm2d(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first. The ordering of the
    dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height, width, channels)
    while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        assert self.data_format in ["channels_last", "channels_first"]
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class SRM(nn.Module):
    def __init__(self):
        super().__init__()
        self.pwc = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        self.dwc = nn.Conv2d(3, 1, kernel_size=3, padding=1, bias=False)
        self.act = nn.GELU()
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        x_max = x.max(1, keepdim=True)[0]  # b, 1, h, w
        x_mean = x.mean(1, keepdim=True)  # b, 1, h, w
        x_std = x.std(1, keepdim=True)  # b, 1, h, w
        u = torch.cat([x_max, x_mean, x_std], dim=1)  # b, 3, h, w
        f = self.act(self.pwc(u) + self.dwc(u))
        f = self.bn(f)
        g = torch.sigmoid(f)
        return x * g.expand_as(x)


class Mlp(nn.Module):
    """An implementation of FFN with Channel Aggregation.

    Args:
        embed_dims (int): The feature dimension. Same as `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        kernel_size (int): The depth-wise conv kernel size as the depth-wise convolution. Defaults to 3.
        act_type (str): The type of activation. Defaults to 'GELU'.
        ffn_drop (float, optional): Probability of an element to be zeroed in FFN. Default 0.0.
    """

    def __init__(self, embed_dims, feedforward_channels, kernel_size=3, act_type="GELU", ffn_drop=0.0):
        super().__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels

        self.fc1 = nn.Conv2d(in_channels=embed_dims, out_channels=self.feedforward_channels, kernel_size=1)
        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels,
        )
        self.act = build_act_layer(act_type)
        self.fc2 = nn.Conv2d(in_channels=feedforward_channels, out_channels=embed_dims, kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.srm = SRM()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.srm(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiOrderDWConv(nn.Module):
    """Multi-order Features with Dilated DWConv Kernel.

    Args:
        embed_dims (int): Number of input channels.
        dw_dilation (list): Dilations of three DWConv layers.
        channel_split (list): The raletive ratio of three split channels.
    """

    def __init__(
        self,
        embed_dims,
        channel_split=[1, 3, 4, 2],
        rates=[6, 12, 18],
        flag_useAllChannels=False,
    ):
        super().__init__()

        channel_split = [5, 5, 5, 1]

        self.useAllChannels = flag_useAllChannels
        if flag_useAllChannels:
            channel_indices = [(0, embed_dims)] * len(channel_split)
        else:
            split_ratio = [i / sum(channel_split) for i in channel_split]
            channel_indices = [(0, int(split_ratio[0] * embed_dims))]
            for cr in split_ratio[1:]:
                nci = int(cr * embed_dims)
                assert nci > 0, "Ops. Channel split ratio is not correct"
                channel_indices.append((channel_indices[-1][1], channel_indices[-1][1] + nci))
        self.channel_indices = channel_indices

        assert len(rates) + 1 == len(channel_split) == 4

        self.dlps = nn.ModuleList()
        for rate, cids in zip(rates, channel_indices):
            kernel_size_effective = 3 + (3 - 1) * (rate - 1)
            (kernel_size_effective - 1) // 2
            self.dlps.append(
                SepConvBN(
                    in_channels=cids[1] - cids[0],
                    filters=cids[1] - cids[0],
                    kernel_size=3,
                    stride=1,
                    rate=rate,
                    depth_activation=True,
                    epsilon=1e-5,
                )
            )

        ipd = channel_indices[-1][1] - channel_indices[-1][0]
        # image pooling
        self.dlps.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Conv2d(ipd, ipd, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(ipd, eps=1e-5),
                nn.LeakyReLU(inplace=True),
                nn.UpsamplingBilinear2d(scale_factor=7),
            )
        )

        self.embed_dims = embed_dims
        # a channel convolution
        self.PW_conv = nn.Conv2d(  # point-wise convolution
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1
        )

    def forward(self, x):
        dls_res = []
        for dlp, csps in zip(self.dlps, self.channel_indices):
            y = dlp(x[:, csps[0] : csps[1], ...])
            if y.shape[2] != x.shape[2] or y.shape[3] != x.shape[3]:
                y = F.interpolate(y, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
            dls_res.append(y)

        if self.useAllChannels:
            x = torch.sum(dls_res)
        else:
            x = torch.cat(dls_res, dim=1)

        x = self.PW_conv(x)
        return x


class CCU(nn.Module):
    def __init__(self, channel, hidden_scale=3):
        super().__init__()
        self.fc1 = nn.Conv1d(channel, hidden_scale * channel, kernel_size=3, groups=channel, bias=False, padding=0)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(hidden_scale * channel, channel, kernel_size=1, groups=channel, bias=False, padding=0)
        self.bn = nn.BatchNorm1d(channel)

    def forward(self, x):
        b, c, _h, _w = x.shape
        x_max = torch.max(x.view(x.size(0), x.size(1), -1), dim=2)[0]
        x_mean = torch.mean(x, dim=(2, 3))
        x_std = torch.std(x, dim=(2, 3), unbiased=False)

        u = torch.stack([x_max, x_mean, x_std], dim=-1)
        # style integration
        z = self.fc2(self.act(self.fc1(u))).view(b, c)
        if z.shape[0] > 1:
            z = self.bn(z)
        g = torch.sigmoid(z)
        g = g.reshape(b, c, 1, 1)
        return x * g.expand_as(x)


class MCA(nn.Module):
    """Spatial Block with Multi-scale Contextual Aggregation.

    Args:
        embed_dims (int): Number of input channels.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of split channels.
        attn_act_type (str): The activation type for Spatial Block. Defaults to 'SiLU'.
    """

    def __init__(self, embed_dims, attn_channel_split=[1, 3, 4], attn_act_type="SiLU", rates=[2, 3, 4]):
        super().__init__()
        self.attn_force_fp32 = True  # whether to force the attention to use fp32
        self.embed_dims = embed_dims
        self.gate = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(embed_dims=embed_dims, rates=rates, channel_split=attn_channel_split)
        self.proj_2 = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        # activation for gating and value
        self.act_gate = build_act_layer(attn_act_type)

        # denoising module (NLB)
        self.denoising_module = Nonlocal(embed_dims)

        self.ccu = CCU(embed_dims)  # style-based recalibration

    def forward(self, x):
        shortcut = x.clone()
        x = self.ccu(x)
        g = self.gate(x)
        v = self.value(x)
        x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        x = x + shortcut
        x = self.denoising_module(x)
        return x


class CFAMblock(nn.Module):
    """A block of CFAM.

    Args:
        embed_dims (int): Number of input channels.
        ffn_ratio (float): The expansion ratio of feedforward network hidden layer channels. Defaults to 4.
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        act_type (str): The activation type for projections and FFNs. Defaults to 'GELU'.
        norm_cfg (str): The type of normalization layer. Defaults to 'BN'.
        init_value (float): Init value for Layer Scale. Defaults to 1e-5.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of split channels.
        attn_act_type (str): The activation type for the gating branch. Defaults to 'SiLU'.
    """

    def __init__(
        self,
        embed_dims,
        ffn_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        act_type="GELU",
        norm_type="BN",
        init_value=1e-5,
        attn_channel_split=[1, 3, 4],
        attn_act_type="SiLU",
        mca_rates=[6, 12, 18],
        writer=None,
    ):
        super().__init__()
        self.writer = writer
        self.out_channels = embed_dims

        self.norm1 = build_norm_layer(norm_type, embed_dims)

        # spatial attention
        self.mca = MCA(
            embed_dims,
            attn_channel_split=attn_channel_split,
            attn_act_type=attn_act_type,
            rates=mca_rates,
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = build_norm_layer(norm_type, embed_dims)

        # channel MLP
        mlp_hidden_dim = int(embed_dims * ffn_ratio)
        self.mlp = Mlp(embed_dims, mlp_hidden_dim, 3, act_type, drop_rate)

        # init layer scale
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)

    def forward(self, x):
        # spatial
        identity = x
        x = self.layer_scale_1 * self.mca(self.norm1(x))
        x = identity + self.drop_path(x)
        # channel
        identity = x
        x = self.layer_scale_2 * self.mlp(self.norm2(x))
        x = identity + self.drop_path(x)
        return x


class CFAMblock_C3k(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(CFAMblock(c_) for _ in range(n)))


class CFAM(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                CFAMblock(self.c),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else CFAMblock_C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else CFAMblock(self.c)
            for _ in range(n)
        )


class PSABlock_CFAM(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.ffn = CFAMblock(c)


class C2PSA_CFAM(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_CFAM(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
