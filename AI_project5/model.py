# model.py (Attention UNet++ 版本)
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *


class AttentionGate(nn.Module):
    """注意力门机制，与原实现保持一致"""

    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_l = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_l(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ResidualConv(nn.Module):
    """带残差连接的卷积块，用于UNet++的密集连接"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class AttentionUNetPlusPlus(nn.Module):
    def __init__(self):
        super(AttentionUNetPlusPlus, self).__init__()
        filters = [HIDDEN_CHANNELS, HIDDEN_CHANNELS * 2, HIDDEN_CHANNELS * 4, HIDDEN_CHANNELS * 8]

        # 编码器部分 (Encoder)
        self.enc0_0 = ResidualConv(INPUT_CHANNELS, filters[0])
        self.enc1_0 = ResidualConv(filters[0], filters[1], stride=2)
        self.enc2_0 = ResidualConv(filters[1], filters[2], stride=2)
        self.enc3_0 = ResidualConv(filters[2], filters[3], stride=2)

        # 密集连接编码路径 (Encoder Dense Connections)
        self.enc0_1 = ResidualConv(filters[0] + filters[1], filters[0])
        self.enc1_1 = ResidualConv(filters[1] + filters[2], filters[1])
        self.enc2_1 = ResidualConv(filters[2] + filters[3], filters[2])

        self.enc0_2 = ResidualConv(filters[0] * 2 + filters[1], filters[0])
        self.enc1_2 = ResidualConv(filters[1] * 2 + filters[2], filters[1])

        self.enc0_3 = ResidualConv(filters[0] * 3 + filters[1], filters[0])

        # 注意力门 (Attention Gates)
        self.att1 = AttentionGate(filters[1], filters[0], filters[0] // 2)
        self.att2 = AttentionGate(filters[2], filters[1], filters[1] // 2)
        self.att3 = AttentionGate(filters[3], filters[2], filters[2] // 2)

        # 解码器部分 (Decoder)
        self.up3_0 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.dec2_0 = ResidualConv(filters[2] * 2, filters[2])

        self.up2_1 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.dec1_1 = ResidualConv(filters[1] * 2, filters[1])

        self.up1_2 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.dec0_2 = ResidualConv(filters[0] * 2, filters[0])

        self.up0_3 = nn.ConvTranspose2d(filters[0], filters[0], 2, stride=1)  # 保持尺寸

        # 最终输出层
        self.out = nn.Conv2d(filters[0], OUTPUT_CHANNELS, 1)

    def forward(self, x):
        # 编码器前向传播
        x0_0 = self.enc0_0(x)
        x1_0 = self.enc1_0(x0_0)
        x2_0 = self.enc2_0(x1_0)
        x3_0 = self.enc3_0(x2_0)

        # 密集连接路径
        x0_1 = self.enc0_1(torch.cat([x0_0, self.att1(g=x1_0, x=x0_0)], dim=1))
        x1_1 = self.enc1_1(torch.cat([x1_0, self.att2(g=x2_0, x=x1_0)], dim=1))
        x2_1 = self.enc2_1(torch.cat([x2_0, self.att3(g=x3_0, x=x2_0)], dim=1))

        x0_2 = self.enc0_2(torch.cat([x0_0, x0_1, self.att1(g=x1_1, x=x0_0)], dim=1))
        x1_2 = self.enc1_2(torch.cat([x1_0, x1_1, self.att2(g=x2_1, x=x1_0)], dim=1))

        x0_3 = self.enc0_3(torch.cat([x0_0, x0_1, x0_2, self.att1(g=x1_2, x=x0_0)], dim=1))

        # 解码器路径
        x2_2 = self.dec2_0(torch.cat([x2_1, self.up3_0(x3_0)], dim=1))
        x1_3 = self.dec1_1(torch.cat([x1_2, self.up2_1(x2_2)], dim=1))
        x0_4 = self.dec0_2(torch.cat([x0_3, self.up1_2(x1_3)], dim=1))

        # 最终输出
        out = self.up0_3(x0_4)
        return self.out(out)


def init_model(pretrained=False):
    model = AttentionUNetPlusPlus().to(DEVICE)
    return model