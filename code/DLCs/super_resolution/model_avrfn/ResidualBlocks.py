import torch
import torch.nn as nn
import torch.nn.functional as F
from AttentionLayers import *
from InceptionModules import *
from FeatureExtractors import *

# Residual Block
class res_block(nn.Module) :
    def __init__(self, filters) :
        super(res_block, self).__init__()
        self.identity = nn.Identity()
        self.conv2D_1 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same'),
            nn.ReLU()
        )
        self.conv2D_2 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same')

    def forward(self, x) :
        skip_conn = self.identity(x)
        x = self.conv2D_1(x)
        x = self.conv2D_2(x)
        x = x + skip_conn
        return x

# Residual Channel Attention Block
class residual_channel_attention_block(nn.Module) :
    """
    Residual Channel Attention Block
    """
    def __init__(self, filters) :
        super(residual_channel_attention_block, self).__init__()
        self.identity = nn.Identity()
        self.conv2D_1 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same'),
            nn.ReLU()
        )
        self.conv2D_2 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same')
        self.channel_attention = Channel_Attention(filters)

    def forward(self, x) :
        skip_conn = self.identity(x)
        x = self.conv2D_1(x)
        x = self.conv2D_2(x)
        x = self.channel_attention(x)
        x = x + skip_conn
        return x

# Residual Channel Attention Block with Dilated Convolutions
class rcab_dd_comp(nn.Module) :
    def __init__(self, filters) :
        super(rcab_dd_comp, self).__init__()
        self.identity = nn.Identity()
        self.conv2D_1 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same', dilation = 2),
            nn.ReLU()
        )
        self.conv2D_2 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same', dilation = 2)
        self.channel_attention = Channel_Attention(filters)

    def forward(self, x) :
        skip_conn = self.identity(x)
        x = self.conv2D_1(x)
        x = self.conv2D_2(x)
        x = self.channel_attention(x)
        x = x + skip_conn
        return x

# Residual Channel Attention Block with Dilated Convolutions and Batch Normalization
class rcab_dilated(nn.Module) :
    def __init__(self, filters, d = 1) :
        super(rcab_dilated, self).__init__()
        self.identity = nn.Identity()
        self.conv2D_1 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same', dilation = d),
            nn.ReLU()
        )
        self.conv2D_2 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same')
        self.bn1 = nn.BatchNorm2d(filters)
        self.bn2 = nn.BatchNorm2d(filters)
        self.channel_attention = Channel_Attention(filters)

    def forward(self, x) :
        skip_conn = self.identity(x)
        x = self.conv2D_1(x)
        x = self.bn1(x)
        x = self.conv2D_2(x)
        x = self.bn2(x)
        x = self.channel_attention(x)
        x = x + skip_conn
        return x

# Channel Attention Block (No Residual)
class rcab_features_summed(nn.Module) :
    def __init__(self, filters) :
        super(rcab_features_summed, self).__init__()
        self.conv2D_1 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same', dilation=d),
            nn.ReLU()
        )
        self.conv2D_2 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same')
        self.bn1 = nn.BatchNorm2d(filters)
        self.bn2 = nn.BatchNorm2d(filters)
        self.channel_attention = Channel_Attention(filters)

    def forward(self, x) :
        x = self.conv2D_1(x)
        x = self.bn1(x)
        x = self.conv2D_2(x)
        x = self.bn2(x)
        x = self.channel_attention(x)

        return x

# Residual Channel Attention Block
class rcab_no_res(nn.Module) :
    def __init__(self, filters, name=None) :
        super(rcab_no_res, self).__init__()
        self.conv2D_1 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same', dilation=d),
            nn.ReLU()
        )
        self.conv2D_2 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same')
        self.bn1 = nn.BatchNorm2d(filters)
        self.bn2 = nn.BatchNorm2d(filters)
        self.channel_attention = Channel_Attention(filters)

    def forward(self, x) :
        x = self.conv2D_1(x)
        x = self.bn1(x)
        x = self.conv2D_2(x)
        x = self.bn2(x)
        x = self.channel_attention(x)

        return x

# Residual Channel Attention Block With Scale Attention
class RCAB_with_Scale_Attention(nn.Module) :
    def __init__(self, filters, name=None) :
        self.identity = nn.Identity()
        super(RCAB_with_Scale_Attention, self).__init__()
        self.conv2D_1 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same', dilation=d),
            nn.ReLU()
        )
        self.conv2D_2 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same')
        self.bn1 = nn.BatchNorm2d(filters)
        self.bn2 = nn.BatchNorm2d(filters)
        self.channel_attention = Scale_Attention(filters)

    def forward(self, x) :
        skip_conn = self.identity(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.channel_attention(x)
        x = x + skip_conn

        return x

# RCAB with Inception, Different Kernel Sizes
class RCAB_with_inception(nn.Module):
    def __init__(self, filters, name=None):
        super(RCAB_with_inception, self).__init__()
        self.identity = nn.Identity()
        self.conv2D_1 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same'),
            nn.ReLU()
        )
        self.conv2D_2 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same')
        self.incept = inception_module(filters)
        self.channel_attention = Channel_Attention(filters)

    def forward(self, x):
        skip_conn = self.identity(x)
        x = self.conv2D_1(x)
        x = self.incept(x)
        x = self.conv2D_2(x)
        x = self.channel_attention(x)
        x = x + skip_conn

        return x

# RCAB with Inception, Different dilation rates
class RCAB_incept_dilated(nn.Module):
    def __init__(self, filters, name=None):
        super(RCAB_incept_dilated, self).__init__()
        self.indentity = nn.Identity()
        self.incept = incept_dilated(filters)
        self.conv2D_1 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same'),
            nn.ReLU()
        )
        self.conv2D_2 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same')
        self.channel_attention = Channel_Attention(filters)

    def forward(self, x):
        skip_conn = self.identity(x)
        x = self.conv2D_1(x)
        x = self.incept(x)
        x = self.conv2D_2(x)
        x = self.channel_attention(x)
        x = x + skip_conn

        return x

# RCAB with Inception, Different dilation rates
class RCAB_dense_dilated(nn.Module):
    def __init__(self, filters, d = 1):
        super(RCAB_dense_dilated, self).__init__()
        self.identity = nn.Identity()
        self.conv2D_1 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same'),
            nn.ReLU()
        )
        self.conv2D_2 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same', dilation = d),
            nn.ReLU()
        )
        self.conv2D_3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same', dilation=d),
            nn.ReLU()
        )
        self.conv2D_4 = nn.Conv2d(filters * 3, filters, kernel_size=(3, 3), padding='same')
        self.channel_attention = Channel_Attention(filters)

    def forward(self, x):
        skip_conn = self.identity(x)
        a = self.conv2D_1(x)
        b = self.conv2D_2(x)
        c = self.conv2D_3(x)
        attent = torch.cat([a, b, c], dim = 1)
        d = self.conv2D_4(attent)
        x = self.channel_attention(d)
        x = x + skip_conn

        return x

# RCAB with Inception, Different dilation rates
class RCAB_dense_dilated_SOCA(nn.Module):
    def __init__(self, filters, d = 1, input_shape = (48, 48)):
        super(RCAB_dense_dilated_SOCA, self).__init__()
        self.identity = nn.Identity()
        self.conv2D_1 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same'),
            nn.ReLU()
        )
        self.conv2D_2 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same', dilation = 2),
            nn.ReLU()
        )
        self.conv2D_4 = nn.Conv2d(filters * 2, filters, kernel_size=(3, 3), padding='same')
        self.channel_attention = SOCA(filters, input_shape = input_shape)

    def foward(self, x) :
        skip_conn = self.identity(x)
        a = self.conv2D_1(x)
        b = self.conv2D_2(x)
        attent = torch.cat([a, b], dim = 1)
        d = self.conv2D_4(attent)
        x = self.channel_attention(d)
        x = x + skip_conn

        return x

# RCAB with Inception, Different dilation rates
class rcab_dd(nn.Module) :
    def __init__(self, filters, d = 1, input_shape = (48, 48)) :
        super(rcab_dd, self).__init__()
        self.identity = nn.Identity()
        self.conv2D_1 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same'),
            nn.ReLU()
        )
        self.conv2D_2 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same', dilation=2),
            nn.ReLU()
        )
        self.conv2D_4 = nn.Conv2d(filters * 2, filters, kernel_size=(3, 3), padding='same')

    def foward(self, x):
        skip_conn = self.identity(x)
        a = self.conv2D_1(x)
        b = self.conv2D_2(x)
        attent = torch.cat([a, b], dim=1)
        d = self.conv2D_4(attent)
        x = d + skip_conn

        return x

# RCAB with Inception, Different dilation rates, SOCA, smoothed dilations
class RCAB_DD_Smooth_SOCA(nn.Module) :
    def __init__(self, filters, d = 1):
        super(RCAB_DD_Smooth_SOCA, self).__init__()
        self.identity = nn.Identity()
        self.conv2D_1 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same', dilation = d),
            nn.ReLU()
        )
        self.conv2D_2 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(5, 5), padding='same', dilation = d),
            nn.ReLU()
        )
        self.conv2D_3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(7, 7), padding='same', dilation=d),
            nn.ReLU()
        )
        self.conv2D_4 = nn.Conv2d(filters * 3, filters, kernel_size=(3, 3), padding='same')
        self.channel_attention = Channel_Attention(filters)

    def forward(self, x):
        skip_conn = self.identity(x)
        x = self.conv2D_1(x)
        x = self.conv2D_2(x)
        x = self.conv2D_3(x)
        x = self.conv2D_4(x)
        x = self.channel_attention(x)
        x = x + skip_conn

        return x

# RCAB with Inception, Different dilation rates, SOCA, smoothed dilations
class edsr_smooth_block(nn.Module) :
    def __init__(self, filters, d = 1):
        super(edsr_smooth_block, self).__init__()
        self.identity = nn.Identity()
        self.conv2D_1 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same', dilation = 2),
            nn.ReLU()
        )
        self.conv2D_4 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same')
        self.channel_attention = Channel_Attention(filters)

    def forward(self, x):
        skip_conn = self.identity(x)
        x = self.conv2D_1(x)
        x = self.conv2D_4(x)
        x = self.channel_attention(x)
        x = x + skip_conn

        return x

# RCAB with Inception, Different dilation rates, SOCA, smoothed dilations
class RCAB_DD_Smooth_SOCA_incr(nn.Module) :
    def __init__(self, filters, d=1, input_shape=(48, 48)) :
        super(RCAB_DD_Smooth_SOCA_incr, self).__init__()
        self.identity = nn.Identity()
        self.conv2D_1 = SmoothDilated(filters, filters, k=(3, 3), padding='same', activation = 'relu', dilation_factor = 2)
        self.conv2D_2 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(5, 5), padding='same', dilation = d),
            nn.ReLU()
        )
        self.conv2D_3 = SmoothDilated(filters, filters, k=(3, 3), padding='same', activation='relu', dilation_factor=3)
        self.conv2D_4 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same', dilation=d),
            nn.ReLU()
        )
        self.conv2D_5 = nn.Conv2d(filters * 3, filters, kernel_size=(3, 3), padding='same')
        self.channel_attention = SOCA(filters, input_shape = input_shape)

    def foward(self, x) :
        skip_conn = self.identity(x)

        a = self.conv2D_1(x)
        b = self.conv2D_2(a)
        c = self.conv2D_3(b)

        attent = torch.cat([a, b, c], dim = 1)
        x = self.conv2D_4(attent)
        attent2 = torch.cat([x, c], dim = 1)
        x = self.channel_attention(attent2)
        x = x + skip_conn

        return x

# RCAB with Inception, Different dilation rates
class RCAB_dense_dilated_SOCA_stable(nn.Module):
    def __init__(self, filters, name=None, d=1, input_shape=(48, 48)):
        super(RCAB_dense_dilated_SOCA_stable, self).__init__()
        self.identity = nn.Identity()
        self.conv2D_1 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same'),
            nn.ReLU()
        )
        self.conv2D_4 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same')
        self.channel_attention = SOCA(filters, input_shape = input_shape)

    def foward(self, x):
        skip_conn = self.identity(x)
        x = self.conv2D_1(x)
        x = self.conv2D_4(x)
        x = self.channel_attention(x)
        x = x + skip_conn

        return x

# RCAB with Inception, Different dilation rates, Geometric Averaging
class RCAB_dense_dilated_SOCAG(nn.Module):
    def __init__(self, filters, d=1, input_shape=(48, 48)):
        super(RCAB_dense_dilated_SOCAG, self).__init__()
        self.identity = nn.Identity()
        self.deformedMap1 = ConvOffset2D(filters * 2)
        self.deformedMap2 = ConvOffset2D(filters * 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(filters,filters, kernel_size=(3, 3), padding='same'),
            nn.ReLU()
        )
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding = "same")
        self.channel_attention = SOCA(filters, input_shape=input_shape)

    def foward(self, x):
        skip_conn = self.identity(x)

        a = self.deformedMap1(x)
        a = self.conv1(a)
        a = self.deformedMap2(a)
        a = self.conv2(a)
        x = self.channel_attention(a)

        x = x + skip_conn

        return x

# Multi Scale Residual Block
class msrn_block(nn.Module):
    def __init__(self, filters):
        super(msrn_block, self).__init__()
        self.identity = nn.Identity()

        self.conv_a1 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(5, 5), padding='same'),
            nn.ReLU()
        )
        self.conv_a2 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same'),
            nn.ReLU()
        )
        self.conv_b1 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(5, 5), padding='same'),
            nn.ReLU()
        )
        self.conv_b2 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same'),
            nn.ReLU()
        )
        self.conv_c = nn.Conv2d(filters, filters, kernel_size=(1, 1), padding='same')

    def foward(self, x):
        skip_conn = self.identity(x)

        a1 = self.conv_a1(x)
        a2 = self.conv_a2(x)
        merge1 = torch.cat([a1, a2], dim=1)

        b1 = self.conv_b1(merge1)
        b2 = self.conv_b2(merge1)
        merge2 = torch.cat([b1, b2], dim=1)

        c = self.conv_c(merge2)

        return c + skip_conn

# Residual Channel Attention Block with SOCA
class rcab_soca(nn.Module):
    def __init__(self, filters, name=None):
        super(rcab_soca, self).__init__()
        self.identity = nn.Identity()
        self.conv2D_1 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same'),
            nn.ReLU()
        )
        self.conv2D_2 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same')
        self.channel_attention = Channel_Attention(filters)

    def call(self, x):
        skip_conn = self.identity(x)
        x = self.conv2D_1(x)
        x = self.conv2D_2(x)
        x = self.channel_attention(x)
        x = x + skip_conn

        return x

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.tensor(rgb_mean) / std
        for param in self.parameters():
            param.requires_grad = False
