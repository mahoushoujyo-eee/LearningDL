import math
import torch
import torch.nn as nn
from typing import List, Tuple

# ------------- Swish 激活 -------------
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# ------------- SE 模块 -------------
class SEBlock(nn.Module):
    def __init__(self, in_ch: int, se_ratio: float = 0.25):
        super().__init__()
        hidden = max(1, int(in_ch * se_ratio))
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitate = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1),
            Swish(),
            nn.Conv2d(hidden, in_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.excitate(self.squeeze(x))

# ------------- MBConv -------------
class MBConv(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 kernel: int,
                 stride: int,
                 exp_ratio: int,
                 se_ratio: float,
                 drop_rate: float = 0.):
        super().__init__()
        hidden = int(in_ch * exp_ratio)
        self.use_res = stride == 1 and in_ch == out_ch

        layers = []
        # expansion 1×1
        if exp_ratio != 1:
            layers += [nn.Conv2d(in_ch, hidden, 1, bias=False),
                       nn.BatchNorm2d(hidden),
                       Swish()]
        # depthwise k×k
        layers += [nn.Conv2d(hidden, hidden, kernel, stride,
                             kernel // 2, groups=hidden, bias=False),
                   nn.BatchNorm2d(hidden),
                   Swish()]
        # SE
        if se_ratio > 0:
            layers.append(SEBlock(hidden, se_ratio))
        # projection 1×1
        layers += [nn.Conv2d(hidden, out_ch, 1, bias=False),
                   nn.BatchNorm2d(out_ch)]

        self.conv = nn.Sequential(*layers)
        self.drop = nn.Dropout2d(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        if self.use_res:
            out = self.drop(out) + x
        return out

# ------------- 整体网络 -------------
class EfficientNet(nn.Module):
    cfg = [  # (kernel, stride, expand, in_ch, out_ch, layers, se_ratio)
        [3, 1, 1, 32, 16, 1, 0.25],
        [3, 2, 6, 16, 24, 2, 0.25],
        [5, 2, 6, 24, 40, 2, 0.25],
        [3, 2, 6, 40, 80, 3, 0.25],
        [5, 1, 6, 80, 112, 3, 0.25],
        [5, 2, 6, 112, 192, 4, 0.25],
        [3, 1, 6, 192, 320, 1, 0.25]
    ]

    def __init__(self,
                 width_mult: float = 1.0,
                 depth_mult: float = 1.0,
                 dropout: float = 0.2,
                 num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.round_filters(32, width_mult),
                      3, 2, 1, bias=False),
            nn.BatchNorm2d(self.round_filters(32, width_mult)),
            Swish()
        )

        blocks = []
        for k, s, e, i, o, n, se in self.cfg:
            in_ch = self.round_filters(i, width_mult)
            out_ch = self.round_filters(o, width_mult)
            repeats = max(1, int(math.ceil(n * depth_mult)))
            for j in range(repeats):
                stride = s if j == 0 else 1
                blocks.append(MBConv(in_ch if j == 0 else out_ch,
                                     out_ch, k, stride, e, se,
                                     dropout))
        self.blocks = nn.Sequential(*blocks)

        head_ch = self.round_filters(1280, width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(self.round_filters(320, width_mult),
                      head_ch, 1, 1, bias=False),
            nn.BatchNorm2d(head_ch),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(head_ch, num_classes)
        )
        self._init_weights()

    def round_filters(self, ch, mult):
        """通道数按 width_mult 四舍五入到 8 的倍数"""
        ch *= mult
        new_ch = max(8, int(ch + 8 / 2) // 8 * 8)
        if new_ch < 0.9 * ch:  # 防止过度下取整
            new_ch += 8
        return int(new_ch)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

# ------------- 复合缩放接口 -------------
def efficientnet_b0(num_classes=10):
    return EfficientNet(width_mult=1.0, depth_mult=1.0,
                        dropout=0.2, num_classes=num_classes)

def efficientnet_b1(num_classes=10):
    return EfficientNet(width_mult=1.0, depth_mult=1.1,
                        dropout=0.2, num_classes=num_classes)

# 可以继续写 b2-b7，按论文 α=1.2, β=1.1, γ=1.15 计算即可
