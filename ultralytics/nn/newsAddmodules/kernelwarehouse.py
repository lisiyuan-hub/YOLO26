import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv  # 引用你提供的 conv.py 中的标准 Conv 类

class KernelWarehouseConv(nn.Module):
    """
    适配 YOLOv8 C2f 的 Kernel Warehouse 卷积模块
    """
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super(KernelWarehouseConv, self).__init__()
        
        # 使用 Conv 类处理基础卷积，因为它自带了 BN 和 SiLU 激活函数
        # 这样能直接嵌入到 C2f 的模块列表里
        self.base_conv = Conv(c1, c2, k, s, p, g, act=act)
        
        # 通道注意力：保持原逻辑，但适配 channels
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, max(c2 // 16, 1), 1),
            nn.ReLU(),
            nn.Conv2d(max(c2 // 16, 1), c2, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力：保持原逻辑
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 1. 基础卷积 (含 BN 和 Act)
        x = self.base_conv(x)
        
        # 2. 通道注意力
        ca = self.channel_attention(x)
        x = x * ca
        
        # 3. 空间注意力
        sa = torch.cat([x.mean(dim=1, keepdim=True), x.max(dim=1, keepdim=True)[0]], dim=1)
        sa = self.spatial_attention(sa)
        x = x * sa
        
        return x