import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


# GELAN模块实现（block.py中添加）
class GELAN(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # 动态通道配比
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_ * 2, c2, 1)
        self.m = nn.ModuleList([Conv(c_, c_, 3, 1, g=g) for _ in range(n)])
        self.gap = nn.AdaptiveAvgPool2d(1)  # 全局池化
        
    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x).chunk(2, 1)[0]  # 通道拆分
        for m in self.m:
            y2 = m(y2)
        # 全局增强
        g = self.gap(y1)
        y1 = y1 * g.expand_as(y1)
        return self.cv3(torch.cat([y1, y2], 1))