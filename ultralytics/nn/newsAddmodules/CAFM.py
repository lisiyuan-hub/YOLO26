import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = ["CAFM"]


# https://github.com/chenx2000/DSMT-main/blob/main/simulation/train/model/DSMT.py#L391
# Cross attention fusion module
class CAFM(nn.Module):
    def __init__(self, channel=112):
        super().__init__()

        self.conv1_spatial = nn.Conv2d(2, 1, 3, stride=1, padding=1, groups=1)
        self.conv2_spatial = nn.Conv2d(1, 1, 3, stride=1, padding=1, groups=1)

        self.avg1 = nn.Conv2d(channel, 64, 1, stride=1, padding=0)
        self.avg2 = nn.Conv2d(channel, 64, 1, stride=1, padding=0)
        self.max1 = nn.Conv2d(channel, 64, 1, stride=1, padding=0)
        self.max2 = nn.Conv2d(channel, 64, 1, stride=1, padding=0)

        self.avg11 = nn.Conv2d(64, channel, 1, stride=1, padding=0)
        self.avg22 = nn.Conv2d(64, channel, 1, stride=1, padding=0)
        self.max11 = nn.Conv2d(64, channel, 1, stride=1, padding=0)
        self.max22 = nn.Conv2d(64, channel, 1, stride=1, padding=0)
        self.channel = channel

        self.fusion = nn.Conv2d(channel * 2, channel, 1, 1, 0)

    def forward(self, DATA):
        f1, f2 = DATA
        b, c, h, w = f1.size()

        f1 = f1.reshape([b, c, -1])
        f2 = f2.reshape([b, c, -1])

        avg_1 = torch.mean(f1, dim=-1, keepdim=True).unsqueeze(-1)
        max_1, _ = torch.max(f1, dim=-1, keepdim=True)
        max_1 = max_1.unsqueeze(-1)

        avg_1 = F.relu(self.avg1(avg_1))
        max_1 = F.relu(self.max1(max_1))
        avg_1 = self.avg11(avg_1).squeeze(-1)
        max_1 = self.max11(max_1).squeeze(-1)
        a1 = avg_1 + max_1

        avg_2 = torch.mean(f2, dim=-1, keepdim=True).unsqueeze(-1)
        max_2, _ = torch.max(f2, dim=-1, keepdim=True)
        max_2 = max_2.unsqueeze(-1)

        avg_2 = F.relu(self.avg2(avg_2))
        max_2 = F.relu(self.max2(max_2))
        avg_2 = self.avg22(avg_2).squeeze(-1)
        max_2 = self.max22(max_2).squeeze(-1)
        a2 = avg_2 + max_2

        cross = torch.matmul(a1, a2.transpose(1, 2))

        a1 = torch.matmul(F.softmax(cross, dim=-1), f1)
        a2 = torch.matmul(F.softmax(cross.transpose(1, 2), dim=-1), f2)

        a1 = a1.reshape([b, c, h, w])
        avg_out = torch.mean(a1, dim=1, keepdim=True)
        max_out, _ = torch.max(a1, dim=1, keepdim=True)
        a1 = torch.cat([avg_out, max_out], dim=1)
        a1 = F.relu(self.conv1_spatial(a1))
        a1 = self.conv2_spatial(a1)
        a1 = a1.reshape([b, 1, -1])
        a1 = F.softmax(a1, dim=-1)

        a2 = a2.reshape([b, c, h, w])
        avg_out = torch.mean(a2, dim=1, keepdim=True)
        max_out, _ = torch.max(a2, dim=1, keepdim=True)
        a2 = torch.cat([avg_out, max_out], dim=1)
        a2 = F.relu(self.conv1_spatial(a2))
        a2 = self.conv2_spatial(a2)
        a2 = a2.reshape([b, 1, -1])
        a2 = F.softmax(a2, dim=-1)

        f1 = f1 * a1 + f1
        f2 = f2 * a2 + f2

        f1 = rearrange(f1, "b n (h d) -> b n h d", h=h, d=w)
        f2 = rearrange(f2, "b n (h d) -> b n h d", h=h, d=w)

        out = self.fusion(torch.cat((f1, f2), dim=1))
        return out
