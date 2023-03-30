import torch
import torch.nn as nn
from ResidualBlocks import *

# Residual Group
class residual_group(nn.Module):
    def __init__(self, filters, n_blocks=5):
        super(residual_group, self).__init__()
        self.blocks = [residual_channel_attention_block(filters) for i in range(n_blocks)]
        self.identity = nn.Identity()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=(3,3), padding='same')

    def forward(self, x):
        skip_conn = self.identity(x)
        for block in self.blocks :
            x = block(x)
        x = self.conv1(x)
        x = x + skip_conn

        return x

# Residual Group, Dilated Convolutions
class RG_dd(nn.Module):
    def __init__(self, filters, n_blocks=5):
        super(RG_dd, self).__init__()
        self.blocks = [rcab_dd(filters) for i in range(n_blocks)]
        self.identity = nn.Identity()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=(3,3), padding='same')

    def forward(self, x):
        skip_conn = self.identity(x)
        for block in self.blocks :
            x = block(x)
        x = self.conv1(x)
        x = x + skip_conn
        return x

# Residual Group with Dilation Rates = 2
class RG_dd_comp(nn.Module):
    def __init__(self, filters, n_blocks=5):
        super(RG_dd_comp, self).__init__()
        self.identity = nn.Identity()
        self.blocks = nn.ModuleList([rcab_dd_comp(filters) for i in range(n_blocks)])
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=(3,3), padding='same')

    def forward(self, x):
        skip_conn = self.identity(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv1(x)
        x = x + skip_conn
        return x

# Residual Group with Smoothed Convolutions
class smooth_group(nn.Module):
    def __init__(self, filters, n_blocks=5):
        super(smooth_group, self).__init__()
        self.identity = nn.Identity()
        self.blocks = nn.ModuleList([edsr_smooth_block(filters) for i in range(n_blocks)])
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same')

    def forward(self, x):
        skip_conn = self.identity(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv1(x)
        x = x + skip_conn
        return x


# Residual Group with Dialated Convolutions
class RG_dilated(nn.Module):
    def __init__(self, filters, n_blocks=5, d=5):
        super(RG_dilated, self).__init__()
        self.identity = nn.Identity()
        self.blocks = nn.ModuleList([rcab_dilated(filters, d=d) for i in range(n_blocks)])
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same', dilation=d)

    def forward(self, x):
        skip_conn = self.identity(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv1(x)
        x = x + skip_conn
        return x


# Residual Group with Summed Features in Residual Block
class RG_features_summed(nn.Module):
    def __init__(self, filters, n_blocks=5):
        super(RG_features_summed, self).__init__()
        self.blocks = nn.ModuleList([rcab_features_summed(filters) for i in range(n_blocks)])
        self.conv1 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same'),
            nn.ReLU()
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.conv1(x)
        return x

# Residual Group with Scale Attention
class residual_group_with_Scale_Attention(nn.Module):
    def __init__(self, filters, n_blocks=5):
        super(residual_group_with_Scale_Attention, self).__init__()
        self.identity = nn.Identity()
        self.blocks = nn.ModuleList([RCAB_with_Scale_Attention(filters) for i in range(n_blocks)])
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same')

    def forward(self, x):
        skip_conn = self.identity(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv1(x)
        x = x + skip_conn
        return x

# Residual Group with Dense Connections, Channel Attention
class dense_res_group(nn.Module):
    def __init__(self, filters, n_blocks=5):
        super(dense_res_group, self).__init__()
        self.identity = nn.Identity()
        self.blocks = nn.ModuleList([RCAB_with_Scale_Attention(filters) for i in range(n_blocks)])
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same')

    def forward(self, x):
        skip_conn = self.identity(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv1(x)
        x = x + skip_conn
        return x

# Residual Group With Inception
class residual_group_with_inception(nn.Module):
    def __init__(self, filters, n_blocks=5):
        super(residual_group_with_inception, self).__init__()
        self.identity = nn.Identity()
        self.blocks = nn.ModuleList([RCAB_with_inception(filters) for i in range(n_blocks)])
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same')

    def forward(self, x):
        skip_conn = self.identity(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv1(x)
        x = x + skip_conn
        return x

# Residual Group With Inception, SOCA
class Dense_RG_SOCA(nn.Module):
    def __init__(self, filters, n_blocks=5, input_shape=(48,48)):
        super(Dense_RG_SOCA, self).__init__()
        self.identity = nn.Identity()
        self.blocks = nn.ModuleList([RCAB_dense_dilated_SOCA(filters, input_shape=input_shape) for i in range(n_blocks)])
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same')

    def forward(self, x):
        skip_conn = self.identity(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv1(x)
        x = x + skip_conn
        return x

# Residual Group With SOCA, Smoothed Convolutions
class Dense_RG_SOCA_smooth(nn.Module):
    def __init__(self, filters, n_blocks=5, input_shape=(48, 48)):
        super(Dense_RG_SOCA_smooth, self).__init__()
        self.identity = nn.Identity()
        self.blocks = nn.ModuleList([RCAB_DD_Smooth_SOCA(filters, input_shape=input_shape) for i in range(n_blocks)])
        self.conv1 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same'),
            nn.ReLU()
        )

    def forward(self, x):
        skip_conn = self.identity(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv1(x)
        x = x + skip_conn
        return x

# Residual Group With multi-level aggregation, SOCA, Smoothed Convolutions
class Dense_RG_SOCA_smooth_incr(nn.Module):
    def __init__(self, filters, n_blocks=5, input_shape=(48, 48)):
        super(Dense_RG_SOCA_smooth_incr, self).__init__()
        self.identity = nn.Identity()
        self.blocks = nn.ModuleList([RCAB_DD_Smooth_SOCA_incr(filters, input_shape=input_shape) for i in range(n_blocks)])
        self.conv1 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same'),
            nn.ReLU()
        )

    def forward(self, x):
        skip_conn = self.identity(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv1(x)
        x = x + skip_conn
        return x

# Residual Group with SOCA, 1 by 1 kernel sizes
class Dense_RG_SOCA_stable(nn.Module):
    def __init__(self, filters, n_blocks=5, input_shape=(48, 48)):
        super(Dense_RG_SOCA_stable, self).__init__()
        self.identity = nn.Identity()
        self.blocks = nn.ModuleList([RCAB_dense_dilated_SOCA_stable(filters, input_shape=input_shape) for i in range(n_blocks)])
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same')

    def forward(self, x):
        skip_conn = self.identity(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv1(x)
        x = x + skip_conn
        return x

# Residual Group With Inception, geometric merging
class Dense_RG_SOCAG(nn.Module):
    def __init__(self, filters, n_blocks=5, input_shape=(48, 48)):
        super(Dense_RG_SOCAG, self).__init__()
        self.identity = nn.Identity(x)
        self.blocks = nn.ModuleList([RCAB_dense_dilated_SOCAG(filters, input_shape=input_shape) for i in range(n_blocks)])
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same')

    def forward(self, x):
        skip_conn = self.identity(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv1(x)
        x = x + skip_conn
        return x

# Residual Group with Dilated Convolutions
class Dense_RG_dilated(nn.Module):
    def __init__(self, filters, n_blocks=5):
        super(Dense_RG_dilated, self).__init__()
        self.identity = nn.Identity()
        self.blocks = nn.ModuleList([RCAB_dense_dilated(filters) for i in range(n_blocks)])
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same')

    def forward(self, x):
        skip_conn = self.identity(x)
        for block in self.blocks :
            x = block(x)
        x = self.conv1(x)
        x = x + skip_conn
        return x

# Residual Group With Inception modules that apply different dilation rates
class res_group_incept_dilated(nn.Module):
    def __init__(self, filters, n_blocks=5):
        super(res_group_incept_dilated, self).__init__()
        self.identity = nn.Identity()
        self.blocks = nn.ModuleList([RCAB_incept_dilated(filters) for i in range(n_blocks)])
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same')

    def forward(self, x):
        skip_conn = self.identity(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv1(x)
        x = x + skip_conn
        return x

# Residual Group With Inception with SOCA
class RG_SOCA(nn.Module):
    def __init__(self, filters, n_blocks=5):
        super(RG_SOCA, self).__init__()
        self.identity = nn.Identity()
        self.blocks = nn.ModuleList([rcab_soca(filters) for i in range(n_blocks)])
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same')

    def forward(self, x):
        skip_conn = self.identity(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv1(x)
        x = x + skip_conn
        return x
