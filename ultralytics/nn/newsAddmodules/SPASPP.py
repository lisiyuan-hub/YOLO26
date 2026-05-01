import torch
import torch.nn as nn
import torch.nn.functional as F
BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = True
 
__all__ = ['SPASPP']
class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, dilation=1):
        super(ConvX, self).__init__()
        if dilation == 1:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, bias=False)
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, dilation=dilation,
                                  padding=dilation, bias=False)
        #         self.bn = BatchNorm2d(out_planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
 
    def forward(self, x):
        out = self.relu(self.conv(x))
        return out
 
 
class Conv1X1(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=1, stride=1, dilation=1):
        super(Conv1X1, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, bias=False)
        #         self.bn = BatchNorm2d(out_planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
 
    def forward(self, x):
        out = self.relu(self.conv(x))
        return out
# 池化 -> 1*1 卷积 -> 上采样
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
            nn.ReLU())
 
    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # 上采样
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
 
    # 整个 ASPP 架构
class SPASPP(nn.Module):
    def __init__(self, in_planes,out_planes, block_num=3, stride=1, dilation=[6,12,18,24]):
        super(SPASPP, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        inter_planes = in_planes
        self.stride = stride
        self.conv_list.append(ConvX(in_planes, inter_planes, stride=stride, dilation=dilation[0]))
        self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[1]))
        self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[2]))
        self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[3]))
        self.pooling = ASPPPooling(in_planes, inter_planes)
        self.process1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=False ),
#             BatchNorm2d(out_planes, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        self.process2 = nn.Sequential(
            nn.Conv2d(inter_planes*5, out_planes, kernel_size=1, padding=0, bias=False ),
#             BatchNorm2d(out_planes, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        self.process3 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1),
#             BatchNorm2d(out_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
#             nn.Dropout(0.2)
        )
    def forward(self, x):
        out_list = []
        out = x
        out1 = self.process1(x)
        out2 = self.pooling(x)
        # out1 = self.conv_list[0](x)
        for idx in range(4):
            out = self.conv_list[idx](out)
            out_list.append(out)
        out = torch.cat(out_list, dim=1)
        out = torch.cat((out,out2), dim=1)
        return self.process3(self.process2(out) + out1)