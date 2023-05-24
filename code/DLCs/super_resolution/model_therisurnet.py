import torch
import torch.nn as nn
import time

# 현재는 4배율만 가능
class pelu(nn.Module):
    def __init__(self):
        super(pelu, self).__init__()

        self.alpha = nn.Parameter(torch.Tensor([1.0]))
        self.alpha.data = torch.clamp(self.alpha.data, 0.1)

        self.beta = nn.Parameter(torch.Tensor([1.0]))
        self.beta.data = torch.clamp(self.beta.data, 0.1)

    def forward(self, x):
        positive = nn.functional.relu(x) * self.alpha / (self.beta + 1e-9)
        negative = self.alpha * (torch.exp((-nn.functional.relu(-x)) / (self.beta + 1e-9)) - 1)

        return negative + positive

class adaptive_global_average_pool_2d(nn.Module):
    def __init__(self):
        super(adaptive_global_average_pool_2d, self).__init__()

    def forward(self, x):
        c = x.size()[1]
        adap2d = torch.mean(x, dim=[2, 3], keepdim=True).view(-1, c, 1, 1)

        return adap2d

class ChannelAttention(nn.Module) :
    def __init__(self, in_channels, f, reduction) :
        super(ChannelAttention, self).__init__()
        self.identity = nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, f // reduction, kernel_size=1)
        self.conv2 = nn.Conv2d(f // reduction, f, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.pelu = pelu().to(self.device)
        self.adaptive_global_average_pool_2d = adaptive_global_average_pool_2d()

    def forward(self, x):
        skip_conn = self.identity(x)

        x = self.adaptive_global_average_pool_2d(x)
        x = self.conv1(x)
        x = self.pelu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)

        y = skip_conn * x

        return y


class ConcatentedBlock(nn.Module):
    def __init__(self, filters):
        super(ConcatentedBlock, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.layer_1 = nn.Sequential(nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=1, padding='same'),
                                     pelu().to(self.device),
                                     nn.Conv2d(filters, filters, kernel_size=(1, 1), stride=1, padding='same'),
                                     pelu().to(self.device),
                                     nn.Conv2d(filters, filters, kernel_size=(1, 1), stride=1, padding='same'),
                                     ChannelAttention(in_channels=filters, f=filters, reduction=4).to(self.device)
        )

        self.layer_2 = nn.Sequential(nn.Conv2d(filters * 2, filters, kernel_size=(3, 3), stride=1, padding='same'),
                                     pelu().to(self.device),
                                     nn.Conv2d(filters, filters, kernel_size=(1, 1), stride=1, padding='same'),
                                     pelu().to(self.device),
                                     nn.Conv2d(filters, filters, kernel_size=(1, 1), stride=1, padding='same'),
                                     ChannelAttention(in_channels=filters, f=filters, reduction=4).to(self.device)
        )

        self.layer_3 = nn.Sequential(nn.Conv2d(filters * 2, filters, kernel_size=(3, 3), stride=1, padding='same'),
                                     pelu().to(self.device),
                                     nn.Conv2d(filters, filters, kernel_size=(1, 1), stride=1, padding='same'),
                                     pelu().to(self.device),
                                     nn.Conv2d(filters, filters, kernel_size=(1, 1), stride=1, padding='same'),
                                     ChannelAttention(in_channels=filters, f=filters, reduction=4).to(self.device)
        )

    def forward(self, x):
        y1 = self.layer_1(x)
        x1 = torch.cat([y1, x], dim=1)

        y2 = self.layer_2(x1)
        x2 = torch.cat([y2, x], dim=1)

        y3 = self.layer_3(x2)
        return y3


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, filters, strides=1):
        super(ResidualBlock, self).__init__()

        self.layer_init = nn.Conv2d(in_channels, filters, kernel_size=1, stride=1, padding='same')

        self.concatented_block_1 = ConcatentedBlock(filters)
        self.concatented_block_2 = ConcatentedBlock(filters)
        self.concatented_block_3 = ConcatentedBlock(filters)

        self.layer_last = nn.Conv2d(filters * 3, filters, kernel_size=1, stride=strides, padding='same', bias=False)

    def forward(self, x):

        x1 = self.layer_init(x)

        x2_1 = self.concatented_block_1(x1)
        x2_2 = self.concatented_block_2(x1)
        x2_3 = self.concatented_block_3(x1)

        x3 = self.layer_last(torch.cat([x2_1, x2_2, x2_3], dim=1))

        y = x1 + x3

        return y

class Upsample2xBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, strides):
        super(Upsample2xBlock, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv = nn.Conv2d(in_channels, filters, kernel_size=kernel_size, stride=strides, padding='same')
        self.pelu = pelu().to(self.device)

    def forward(self, x):
        x = self.conv(x)
        x = self.pelu(x)
        x = nn.functional.pixel_shuffle(x, 2)
        x = self.pelu(x)

        return x

class TherISuRNet(nn.Module):
    def __init__(self):
        super(TherISuRNet, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pelu = pelu().to(self.device)

        self.identity = nn.Identity()

        self.GRL_conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding='same')
        self.GRL_conv2 = nn.Conv2d(64, 16, kernel_size=1, stride=1, padding='same')
        self.GRL_conv3 = nn.Conv2d(16, 3, kernel_size=1, stride=1, padding='same')

        self.LFE_conv = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding="same")

        self.HFE1_ResBlock = nn.ModuleList([
            ResidualBlock(in_channels=64, kernel_size=1, filters=64, strides=1).to(self.device),
            ResidualBlock(in_channels=128, kernel_size=1, filters=64, strides=1).to(self.device),
            ResidualBlock(in_channels=128, kernel_size=1, filters=64, strides=1).to(self.device),
            ResidualBlock(in_channels=128, kernel_size=1, filters=64, strides=1).to(self.device)
        ])
        self.HFE1_conv1 = nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same', bias=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same', bias=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same', bias=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same', bias=False)
        ])
        self.HFE1_conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding='same', bias=False)

        self.HFE2_ResBlock = nn.ModuleList([
            ResidualBlock(in_channels=64, kernel_size=1, filters=32, strides=1).to(self.device),
            ResidualBlock(in_channels=96, kernel_size=1, filters=32, strides=1).to(self.device)
        ])
        self.HFE2_conv1 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding='same', bias=False)
        self.HFE2_conv2 = nn.ModuleList([
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding='same', bias=False),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding='same', bias=False)
        ])
        self.HFE2_conv3 = nn.Conv2d(96, 64, kernel_size=3, stride=1, padding='same', bias=False)

        self.Upsample1 = Upsample2xBlock(in_channels=64, kernel_size=3, filters=64, strides=1).to(self.device)
        self.Upsample2 = Upsample2xBlock(in_channels=64, kernel_size=3, filters=64, strides=1).to(self.device)

        self.Recon_conv1 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding='same')
        self.Recon_conv2 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding='same')



    def forward(self, x):
        # Global Residual Learning
        size = x.size()
        h = size[2]
        w = size[3]

        x_GRL = nn.functional.interpolate(x, size=(h * 4, w * 4), align_corners=False, mode='bicubic')
        x_GRL = self.GRL_conv1(x_GRL)
        x_GRL = self.pelu(x_GRL)
        x_GRL = self.GRL_conv2(x_GRL)
        x_GRL = self.pelu(x_GRL)
        x_GRL = self.GRL_conv3(x_GRL)
        x_GRL = self.pelu(x_GRL)

        # Low-frequency Feature Extraction (LFE)
        x = self.LFE_conv(x)
        x = self.pelu(x)

        # High-frequency Feaure Extraction 1 (HFE1)
        skip1 = self.identity(x)
        for i in range(4):
            x = self.HFE1_ResBlock[i](x)
            x = self.HFE1_conv1[i](x)
            x = self.pelu(x)
            x = torch.cat([x, skip1], dim=1)
        x = self.HFE1_conv2(x)
        x = self.pelu(x)
        x = x + skip1

        x = self.Upsample1(x)

        # High-frequency Feaure Extraction 2 (HFE2)
        x = self.HFE2_conv1(x)
        skip2 = self.identity(x)
        for i in range(2):
            x = self.HFE2_ResBlock[i](x)
            x = self.HFE2_conv2[i](x)
            x = self.pelu(x)
            x = torch.cat([x, skip2], dim=1)
        x = self.HFE2_conv3(x)
        x = self.pelu(x)
        x = x + skip2

        x = self.Upsample2(x)

        # Reconstruction
        x = self.Recon_conv1(x)
        x = self.pelu(x)
        x = self.Recon_conv2(x)
        x = self.pelu(x)

        y = x + x_GRL

        return y
