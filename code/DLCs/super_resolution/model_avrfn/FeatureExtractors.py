import torch
import torch.nn as nn
from ResidualBlocks import *
import numpy as np


class CombineEdgeDetection(nn.Module) :
    def __init__(self) :
        super().__init__()

    def forward(self, x) :
        s_e = torch.image.sobel(x, dim=[2, 3])
        s_e = torch.squeeze(s_e, dim=-2)
        return torch.cat([x, s_e], dim=-1)


def torch_repeat(a, repeats, axis=0) :
    a = a.unsqueeze(-1)
    a = a.repeat([1] * axis + [repeats] + [1] * (len(a.shape) - axis - 1))
    a = a.flatten(0, axis)
    return a


def torch_batch_map_coordinates(input, coords, order=1):
    input_shape = input.shape
    batch_size = input_shape[0]
    input_size = input_shape[2]

    n_coords = coords.shape[1]
    coords = torch.clamp(coords, 0, input_size - 1)
    coords_lt = coords.floor().to(dtype=torch.int32)
    coords_rb = coords.ceil().to(dtype=torch.int32)
    coords_lb = torch.stack([coords_lt[..., 0], coords_rb[..., 1]], dim=-1)
    coords_rt = torch.stack([coords_rb[..., 0], coords_lt[..., 1]], dim=-1)

    idx = torch_repeat(torch.arange(batch_size), n_coords)

    def _get_vals_by_coords(input, coords):
        indices = torch.stack([
            idx, coords[..., 0].flatten(), coords[..., 1].flatten()
        ], dim=-1)
        vals = input[indices[:, 0], indices[:, 1], indices[:, 2]]
        vals = vals.reshape(batch_size, n_coords)
        return vals

    vals_lt = _get_vals_by_coords(input, coords_lt)
    vals_rb = _get_vals_by_coords(input, coords_rb)
    vals_lb = _get_vals_by_coords(input, coords_lb)
    vals_rt = _get_vals_by_coords(input, coords_rt)

    coords_offset_lt = coords - coords_lt.to(dtype=coords.dtype)
    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[..., 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[..., 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[..., 1]

    return mapped_vals


def torch_repeat_2d(a, repeats):
    assert len(a.shape) == 2
    a = a.unsqueeze(0)
    a = a.repeat([repeats] + [1] * (len(a.shape) - 1))
    return a


def torch_batch_map_offsets(input, offsets, order=1):
    input_shape = input.shape
    batch_size = input_shape[0]
    input_size = input_shape[2]

    offsets = offsets.reshape(batch_size, -1, 2)
    grid = torch.meshgrid(
        torch.arange(input_size), torch.arange(input_size)
    )
    grid = torch.stack(grid, dim=-1)
    grid = grid.to(dtype=torch.float32)
    grid = grid.reshape(-1, 2)
    grid = torch_repeat_2d(grid, batch_size)
    coords = offsets + grid

    mapped_vals = torch_batch_map_coordinates(input, coords)
    return mapped_vals


class ConvOffset2D(nn.Conv2d):
    """
    ConvOffset2D
    Convolutional layer responsible for learning the 2D offsets and output the
    deformed feature map using bilinear interpolation
    Note that this layer does not perform convolution on the deformed feature
    map. See get_deform_cnn in cnn.py for usage
    """

    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        """
        Init
        Parameters
        ----------
        filters : int
            Number of channel of the input feature map
        init_normal_stddev : float
            Normal kernel initialization
        **kwargs:
            Pass to superclass. See nn.Conv2d layer in PyTorch
        """

        self.filters = filters
        super(ConvOffset2D, self).__init__(
            self.filters * 2,  self.filters * 2, kernel_size=3,
            padding=1, bias=False,
            kernel_initializer=torch.nn.init.normal_(mean=0, std=init_normal_stddev),
            **kwargs
        )

    def forward(self, x):
        """Return the deformed featured map"""
        x_shape = x.shape
        offsets = super(ConvOffset2D, self).forward(x)

        # offsets: (b*c, h, w, 2)
        offsets = self._to_bc_h_w_2(offsets, x_shape)

        # x: (b*c, h, w)
        x = self._to_bc_h_w(x, x_shape)

        # X_offset: (b*c, h, w)
        x_offset = torch.nn.functional.grid_sample(x, offsets, mode='bilinear', padding_mode='border')

        # x_offset: (b, h, w, c)
        x_offset = self._to_b_h_w_c(x_offset, x_shape)

        return x_offset

    def compute_output_shape(self, input_shape):
        """
        Output shape is the same as input shape
        Because this layer does only the deformation part
        """
        return input_shape

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, h, w, 2c) -> (b*c, h, w, 2)"""
        x = x.transpose(1, 2).transpose(2, 3)
        x = x.reshape(-1, x_shape[1], x_shape[2], 2)
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, h, w, c) -> (b*c, h, w)"""
        x = x.transpose(1, 2).transpose(2, 3)
        x = x.reshape(-1, x_shape[1], x_shape[2])
        return x

    @staticmethod
    def _to_b_h_w_c(x, x_shape):
        """(b*c, h, w) -> (b, h, w, c)"""
        x = x.reshape(-1, x_shape[3], x_shape[1], x_shape[2])
        x = x.transpose(1, 2).transpose(2, 3)
        return x


class MemNet(nn.Module):
    def __init__(self, in_c, c, num_mem_blocks=2, n_resblocks=2):
        super(MemNet, self).__init__()
        self.feature_ext = BN_RELU(c, in_c)
        self.recon = BN_RELU(c, in_c)
        self.memory = nn.ModuleList([MemoryBlock(c, n_resblocks, i + 1) for i in range(num_mem_blocks)])

    def forward(self, x):
        skip = x
        x = self.feature_ext(x)
        ys = [x]
        for mem in self.memory:
            x = mem(x, ys)
        x = self.recon(x)
        x = x + skip
        return x


class MemoryBlock(nn.Module):
    def __init__(self, c, n_resblock, n_memblock):
        super(MemoryBlock, self).__init__()
        self.recursive_unit = nn.ModuleList([ResidualBlock(c) for i in range(n_resblock)])
        self.gate = BN_RELU((n_resblock + n_memblock) * c, c, 1, 1, 0)

    def forward(self, x, ys):
        xs = []
        skip = x
        for layer in self.recursive_unit:
            x = layer(x)
            xs.append(x)
        gate_out = self.gate(torch.cat(xs + ys, dim=1))
        return ys + [gate_out]


class ResidualBlock(nn.Module):
    def __init__(self, c, k=3, s=1, p=1):
        super(ResidualBlock, self).__init__()
        self.relu_conv1 = BN_RELU(c, c, k, s, p)
        self.relu_conv2 = BN_RELU(c, c, k, s, p)

    def forward(self, x):
        skip = x
        x = self.relu_conv1(x)
        x = self.relu_conv2(x)
        x = x + skip
        return x


class BN_RELU(nn.Module):
    def __init__(self, in_c, c, k=3, s=1, p=1):
        super(BN_RELU, self).__init__()
        self.bn = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_c, c, kernel_size=k, stride=s, padding=p)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return x

class SmoothDilated(nn.Module):
    def __init__(self, f_in, f_out, k, padding, dilation_factor, activation):
        super(SmoothDilated, self).__init__()
        self.dilation_factor = dilation_factor
        self.fix_w_size = dilation_factor * 2 - 1
        self.fix_w = torch.zeros([self.fix_w_size, self.fix_w_size, 1, 1, 1])
        self.mask = np.zeros([self.fix_w_size, self.fix_w_size, 1, 1, 1])
        self.mask[dilation_factor - 1][dilation_factor - 1][0][0][0] = 1
        self.conv = nn.Conv2d(in_channels=1, out_channels=f, kernel_size=k, padding=padding, dilation=dilation_factor)
        self.activation = nn.ReLU() if activation == 'relu' else None

    def forward(self, x):
        fix_w = torch.add(self.fix_w, torch.from_numpy(self.mask))
        x = torch.unsqueeze(x, dim=1)
        x = nn.functional.conv3d(x, fix_w, padding=1, groups=1, stride=1, dilation=1)
        x = torch.squeeze(x, dim=1)
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x