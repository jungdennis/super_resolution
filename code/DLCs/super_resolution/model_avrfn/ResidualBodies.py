import torch
import torch.nn as nn
import numpy as np
from ResidualGroups import *

BLOCKS = 8
GROUPS = 4

class MDSR(nn.Module):
    def __init__(self, in_channels, f=64, k=(3,3), n_blocks=BLOCKS):
        super(MDSR, self).__init__()
        self.head = nn.Conv2d(in_channels, f, kernel_size=k, padding='same')
        self.rgb_mean = nn.Parameter(torch.tensor([0.4488, 0.4371, 0.4040]).reshape(1, 3, 1, 1), requires_grad=False)
        self.pre_process = nn.ModuleList([res_block(f) for i in range(2)])
        self.body = nn.ModuleList([res_block(f) for i in range(n_blocks)])

    def forward(self, x):
        x = x - self.rgb_mean
        x = self.head(x)
        for i in range(len(self.pre_process)):
            x = self.pre_process[i](x)
        res = self.body[0](x)
        for i in range(1, len(self.body)):
            res = self.body[i](res)

        return res

class MDSR_smooth(nn.Module):
    def __init__(self, in_channels, f=64, k=(3,3), n_blocks=BLOCKS):
        super(MDSR_smooth, self).__init__()
        self.head = nn.Conv2d(in_channels, f, kernel_size=k, padding='same')
        self.rgb_mean = nn.Parameter(torch.tensor([0.4488, 0.4371, 0.4040]).reshape(1, 3, 1, 1), requires_grad=False)
        self.pre_process = nn.ModuleList([edsr_smooth_block(f) for i in range(2)])
        self.body = nn.ModuleList([edsr_smooth_block(f) for i in range(n_blocks)])

    def forward(self, x):
        x = x - self.rgb_mean
        x = self.head(x)
        for i in range(len(self.pre_process)):
            x = self.pre_process[i](x)
        res = self.body[0](x)
        for i in range(1, len(self.body)):
            res = self.body[i](res)

        return res

class MSRN(nn.Module):
    def __init__(self, in_channels, f=64, k=(3,3), n_blocks=BLOCKS):
        super(MSRN, self).__init__()
        self.head = nn.Conv2d(in_channels, f, kernel_size=k, padding='same')
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(255, rgb_mean, rgb_std)
        self.add_mean = MeanShift(255, rgb_mean, rgb_std, 1)
        self.body = nn.ModuleList([msrn_block(f) for i in range(n_blocks)])

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = x
        for i in range(len(self.body)):
            res = self.body[i](res)
        res += x
        x = self.add_mean(x)

        return x

class EDSR(nn.Module):
    def __init__(self, in_channels, f=64, k=(3, 3), n_blocks=BLOCKS):
        super(EDSR, self).__init__()
        self.head = nn.Conv2d(in_channels, f, kernel_size=k, padding='same')
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(255, rgb_mean, rgb_std)
        self.add_mean = MeanShift(255, rgb_mean, rgb_std, 1)
        self.body = nn.ModuleList([res_block(f) for i in range(n_blocks)])

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = x
        for i in range(len(self.body)):
            res = self.body[i](res)
        res += x
        x = self.add_mean(x)

        return x

# Residual Channel Attention Block with Smoothed Convolutions
class rcan_smooth_body(nn.Module):
    def __init__(self, in_channels, f=64, k=(3, 3), n_groups=GROUPS, n_blocks=BLOCKS):
        super(rcan_smooth_body, self).__init__()
        self.head = nn.Conv2d(in_channels, f, kernel_size=k, padding='same')
        self.body = nn.Conv2d(f, f, kernel_size=k, padding='same')
        self.residual_groups = nn.ModuleList([smooth_group(f, n_blocks=n_blocks) for i in range(n_groups)])

    def forward(self, x):
        head = self.head(x)
        x = head
        for group in self.residual_groups:
            x = group(x)
        x = self.body(x) + head

        return x

# Residual Channel Attention Block
class residual_channel_attention_network(nn.Module):
    def __init__(self, in_channels, f=64, k=(3, 3), n_groups=GROUPS, n_blocks=BLOCKS):
        super(residual_channel_attention_network, self).__init__()
        self.head = nn.Conv2d(in_channels, f, kernel_size=k, padding='same')
        self.body = nn.Conv2d(f, f, kernel_size=k, padding='same')
        self.residual_groups = nn.ModuleList([residual_group(f, n_blocks=n_blocks) for i in range(n_groups)])

    def forward(self, x):
        head = self.head(x)
        x = head
        for group in self.residual_groups:
            x = group(x)
        x = self.body(x) + head

        return x

# Residual Channel Attention Block with SOCA
class RCAN_Dense_SOCA(nn.Module):
    def __init__(self, in_channels, f=64, k=(3, 3), n_groups=GROUPS, n_blocks=BLOCKS, input_shape=(48, 48)):
        super(RCAN_Dense_SOCA, self).__init__()
        self.head = nn.Conv2d(in_channels, f, kernel_size=k, padding='same')
        self.body = nn.Conv2d(f, f, kernel_size=k, padding='same')
        self.residual_groups = nn.ModuleList([Dense_RG_SOCA(f, n_blocks=n_blocks, input_shape=input_shape) for i in range(n_groups)])

    def forward(self, x):
        head = self.head(x)
        x = head
        for group in self.residual_groups:
            x = group(x)
        x = self.body(x) + head

        return x

# Residual Channel Attention Block with SOCA, Smoothed Convolutions
class RCAN_Dense_SOCA_smooth(nn.Module):
    def __init__(self, in_channels, f=64, k=(3, 3), n_groups=GROUPS, n_blocks=BLOCKS, input_shape=(48, 48)):
        super(RCAN_Dense_SOCA_smooth, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, f, kernel_size=k, padding='same'),
            nn.ReLU()
        )
        self.body = nn.Sequential(
            nn.Conv2d(f, f, kernel_size=k, padding='same'),
            nn.ReLU()
        )
        self.residual_groups = nn.ModuleList([Dense_RG_SOCA_smooth(f, n_blocks=n_blocks, input_shape=input_shape) for i in range(n_groups)])

    def forward(self, x):
        head = self.head(x)
        x = head
        for group in self.residual_groups:
            x = group(x)
        x = self.body(x) + head

        return x

# Residual Channel Attention Block with SOCA, Different Dilation Rates, Smoothed convolutions
class RCAN_Dense_SOCA_smooth_res(nn.Module):
    def __init__(self, in_channels, f=64, k=(3, 3), n_blocks=BLOCKS, input_shape=(48, 48)):
        super(RCAN_Dense_SOCA_smooth_res, self).__init__()
        self.head = nn.Conv2d(in_channels, f, kernel_size=k, padding='same')
        self.body = nn.Conv2d(f, f, kernel_size=k, padding='same')
        self.residual_blocks = nn.ModuleList([RCAB_DD_Smooth_SOCA(f, input_shape=input_shape) for i in range(n_blocks)])

    def forward(self, x):
        head = self.head(x)
        x = head
        for block in self.residual_blocks:
            x = block(x)
        x = self.body(x) + head

        return x

# Residual Channel Attention Block with SOCA, Smoothed Dilated Convolutions, Two-level Aggregation
class RCAN_Dense_SOCA_smooth_incr(nn.Module):
    def __init__(self, in_channels, f=64, k=(3, 3), n_groups=GROUPS, n_blocks=BLOCKS, input_shape=(48, 48)):
        super(RCAN_Dense_SOCA_smooth_incr, self).__init__()
        self.head = nn.Conv2d(in_channels, f, kernel_size=k, padding='same')
        self.body = nn.Conv2d(f, f, kernel_size=k, padding='same')
        self.residual_groups = nn.ModuleList(
            [Dense_RG_SOCA_smooth_incr(f, n_blocks=n_blocks, input_shape=input_shape) for i in range(n_groups)])

    def forward(self, x):
        head = self.head(x)
        x = head
        for group in self.residual_groups:
            x = group(x)
        x = self.body(x) + head
        return x


# Residual Channel Attention Block, with SOCA, dilated convolutions, 1 by 1 kernels
class RCAN_Dense_SOCA_stable(nn.Module):
    def __init__(self, in_channels, f=64, k=(3, 3), n_groups=GROUPS, n_blocks=BLOCKS, input_shape=(48, 48)):
        super(RCAN_Dense_SOCA_stable, self).__init__()
        self.head = nn.Conv2d(in_channels, f, kernel_size=k, padding='same')
        self.body = nn.Conv2d(f, f, kernel_size=k, padding='same')
        self.residual_groups = nn.ModuleList([Dense_RG_SOCA_stable(f, n_blocks=n_blocks, input_shape=input_shape) for i in range(n_groups)])

    def forward(self, x):
        head = self.head(x)
        x = head
        for group in self.residual_groups:
            x = group(x)
        x = self.body(x) + head

        return x

# Residual Channel Attention Block, Geometric merging
class RCAN_Dense_SOCAG(nn.Module):
    def __init__(self, in_channels, f=64, k=(3, 3), n_groups=GROUPS, n_blocks=BLOCKS, input_shape=(48, 48)):
        super(RCAN_Dense_SOCAG, self).__init__()
        self.head = nn.Conv2d(in_channels, f, kernel_size=k, padding='same')
        self.body = nn.Conv2d(f, f, kernel_size=k, padding='same')
        self.residual_groups = nn.ModuleList([Dense_RG_SOCAG(f, n_blocks=n_blocks, input_shape=input_shape) for i in range(n_groups)])

    def forward(self, x):
        head = self.head(x)
        x = head
        for group in self.residual_groups:
            x = group(x)
        x = self.body(x) + head

        return x

'''
여기서부터
'''
# Residual Channel Attention Block, Dilated Convolutions
class RCAN_Dense_Dilated(nn.Module):
    def __init__(self, in_channels, f=64, k=(3, 3), n_groups=GROUPS, n_blocks=BLOCKS):
        super(RCAN_Dense_Dilated, self).__init__()
        self.head = nn.Conv2d(in_channels, f, kernel_size=k, padding='same')
        self.body = nn.Conv2d(f, f, kernel_size=k, padding='same')
        self.residual_groups = nn.ModuleList([Dense_RG_dilated(f, n_blocks=n_blocks) for i in range(n_groups)])

    def forward(self, x):
        head = self.head(x)
        x = head
        for group in self.residual_groups:
            x = group(x)
        x = self.body(x) + head

        return x

# Residual Channel Attention Block, Dilated Convolutions
class rcan_dilated(nn.Module):
    def __init__(self, in_channels, f=64, k=(3, 3), n_groups=GROUPS, n_blocks=BLOCKS, d=1):
        super(rcan_dilated, self).__init__()
        self.head = nn.Conv2d(in_channels, f, kernel_size=k, padding='same')
        self.body = nn.Conv2d(f, f, kernel_size=k, padding='same')
        self.residual_groups = nn.ModuleList([RG_dilated(f, n_blocks=n_blocks, d=d) for i in range(n_groups)])

    def forward(self, x):
        head = self.head(x)
        x = head
        for group in self.residual_groups:
            x = group(x)
        x = self.body(x) + head

        return x

# Residual Channel Attention Block, DD
class rcan_dilated_all_shared(nn.Module):
    def __init__(self, in_channels, f=64, k=(3, 3), n_groups=GROUPS, n_blocks=BLOCKS, d=1):
        super(rcan_dilated_all_shared, self).__init__()
        self.head = nn.Conv2d(in_channels, f, kernel_size=k, padding='same')
        self.body = nn.Conv2d(f, f, kernel_size=k, padding='same')
        self.residual_group = RG_dilated(f, n_blocks=n_blocks, d=d)
        self.n_groups = n_groups

    def forward(self, x):
        head = self.head(x)
        x = head
        for i in range(self.n_groups):
            x = self.residual_group(x)
        x = self.body(x) + head

        return x

# Residual Channel Attention Block, summed groups
class rcan_summed_features(nn.Module):
    def __init__(self, in_channels, f=64, k=(3, 3), n_groups=GROUPS, n_blocks=BLOCKS):
        super(rcan_summed_features, self).__init__()
        self.head = nn.Conv2d(in_channels, f, kernel_size=k, padding='same', bias=True)
        self.residual_groups = nn.ModuleList([RG_features_summed(f, n_blocks=n_blocks) for i in range(n_groups)])

    def forward(self, x):
        head = self.head(x)
        features_summed = head
        for group in self.residual_groups:
            head += group(x)

        return head

# Residual Channel Attention Block, multiple supervision
class rcan_multi_supervision(nn.Module):
    def __init__(self, in_channels, f=64, k=(3, 3), n_groups=GROUPS, n_blocks=BLOCKS, c=1):
        super(rcan_multi_supervision, self).__init__()
        self.head = nn.Conv2d(in_channels, f, kernel_size=k, padding='same', bias=True)
        self.body = nn.Conv2d(f, f, kernel_size=k, padding='same', bias=True)
        self.res_blocks = nn.ModuleList([residual_channel_attention_block(f) for i in range(n_blocks)])
        self.tails = nn.ModuleList([nn.Conv2d(in_channels=f, out_channels=c, kernel_size=k, padding='same', bias=True) for i in range(n_groups)])

    def forward(self, x):
        head = self.head(x)
        x = head
        outputs = []
        for block, tail in zip(self.res_blocks, self.tails):
            x = block(x)
            x = tail(x)
            outputs.append(x)

        return outputs

# Residual Channel Attention Block with Inception
class RCAN_with_inception(nn.Module):
    def __init__(self, in_channels, f=64, k=(3, 3), n_groups=GROUPS, n_blocks=BLOCKS):
        super(RCAN_with_inception, self).__init__()
        self.head = nn.Conv2d(in_channels, f, kernel_size=k, padding='same', bias=True)
        self.body = nn.Conv2d(f, f, kernel_size=k, padding='same', bias=True)
        self.residual_groups = nn.ModuleList([residual_group_with_inception(f, n_blocks=n_blocks) for i in range(n_groups)])

    def forward(self, x):
        head = self.head(x)
        x = head
        for group in self.residual_groups:
            x = group(x)
        x = self.body(x) + head

        return x

# Residual Channel Attention Block with Inception modules that have different dilation rates
class RCAN_incept_dilated(nn.Module):
    def __init__(self, in_channels, f=64, k=(3, 3), n_groups=GROUPS, n_blocks=BLOCKS):
        super(RCAN_incept_dilated, self).__init__()
        self.head = nn.Conv2d(in_channels, f, kernel_size=k, padding='same')
        self.body = nn.Conv2d(f, f, kernel_size=k, padding='same')
        self.residual_groups = nn.ModuleList([res_group_incept_dilated(f, n_blocks=n_blocks) for i in range(n_groups)])

    def forward(self, x):
        head = self.head(x)
        x = head
        for group in self.residual_groups:
            x = group(x)
        x = self.body(x) + head
        return x

# Residual Channel Attention Block with Scale Attention
class RCAN_with_Scale_Attention(nn.Module):
    def __init__(self, in_channels, f=64, k=(3, 3), n_groups=GROUPS, n_blocks=BLOCKS):
        super(RCAN_with_Scale_Attention, self).__init__()
        self.head = nn.Conv2d(in_channels, f, kernel_size=k, padding='same')
        self.body = nn.Conv2d(f, f, kernel_size=k, padding='same')
        self.residual_groups = nn.ModuleList([residual_group_with_Scale_Attention(f, n_blocks=n_blocks) for i in range(n_groups)])

    def forward(self, x):
        head = self.head(x)
        x = head
        for group in self.residual_groups:
            x = group(x)
        x = self.body(x) + head

        return x

# Residual Channel Attention Block
class rcan_soca(nn.Module):
    def __init__(self, in_channels, f=64, k=(3, 3), n_groups=GROUPS, n_blocks=BLOCKS):
        super(rcan_soca, self).__init__()
        self.head = nn.Conv2d(in_channels, f, kernel_size=k, padding='same')
        self.body = nn.Conv2d(f, f, kernel_size=k, padding='same')
        self.residual_groups = nn.ModuleList([RG_SOCA(f, n_blocks=n_blocks) for i in range(n_groups)])

    def forward(self, x):
        head = self.head(x)
        x = head
        for group in self.residual_groups:
            x = group(x)
        x = self.body(x) + head

        return x

# Residual Channel Attention Block
class rcan_dd(nn.Module):
    def __init__(self, in_channels, f=64, k=(3, 3), n_groups=GROUPS, n_blocks=BLOCKS):
        super(rcan_dd, self).__init__()
        self.head = nn.Conv2d(in_channels, f, kernel_size=k, padding='same')
        self.body = nn.Conv2d(f, f, kernel_size=k, padding='same')
        self.residual_groups = nn.ModuleList([RG_dd(f, n_blocks=n_blocks) for i in range(n_groups)])

    def forward(self, x):
        head = self.head(x)
        x = head
        for group in self.residual_groups:
            x = group(x)
        x = self.body(x) + head

        return x

# Residual Channel Attention Block, Dilated Convolutions
class rcan_dd_comp(nn.Module):
    def __init__(self, in_channels, f=64, k=(3, 3), n_groups=GROUPS, n_blocks=BLOCKS):
        super(rcan_dd_comp, self).__init__()
        self.head = nn.Conv2d(in_channels, f, kernel_size=k, padding='same')
        self.body = nn.Conv2d(f, f, kernel_size=k, padding='same')
        self.residual_groups = nn.ModuleList([RG_dd(f, n_blocks=n_blocks) for i in range(n_groups)])

    def forward(self, x):
        head = self.head(x)
        x = head
        for group in self.residual_groups:
            x = group(x)
        x = self.body(x) + head

        return x