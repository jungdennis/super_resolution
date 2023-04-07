import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchsummary import summary

class SOCA(nn.Module) :
    def __init__(self, filters, reduction = 1) :
        super(SOCA, self).__init__()
        self.identity = nn.Identity()
        self.conv_du = nn.Sequential(
            nn.Conv2d(filters, filters // reduction, kernel_size = (3, 3), padding = "same"),
            nn.ReLU(),
            nn.Conv2d(filters, filters, kernel_size = (3, 3), padding = "same"),
            nn.Sigmoid()
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def normalizeCov(self, x, iterN) :
        batch_size, channel = x.shape[0], x.shape[-1]

        I3 = torch.eye(channel, channel).to(self.device)
        I3 = I3.reshape((1, channel, channel)).to(self.device)
        I3 = I3.repeat(batch_size, 1, 1).to(self.device)
        I3 = 3 * I3.to(self.device)

        normA = (1/3) * torch.sum(torch.matmul(x, I3), dim=(1, 2)).to(self.device)
        A = x / torch.reshape(normA, (batch_size, 1, 1)).to(self.device)

        Y = torch.zeros((batch_size, channel, channel)).to(self.device)

        Z = torch.eye(channel, channel).to(self.device)
        Z = Z.reshape((1, channel, channel)).to(self.device)
        Z = Z.repeat(batch_size, 1, 1).to(self.device)

        ZY = 0.5 * (I3 - A).to(self.device)
        Y = torch.matmul(A, ZY).to(self.device)
        Z = ZY.to(self.device)

        for i in range(1, iterN - 1) :
            ZY = 0.5 * (I3 - torch.matmul(Z, Y)).to(self.device)
            Y = torch.matmul(Y, ZY).to(self.device)
            Z = torch.matmul(ZY, Z).to(self.device)

        ZY = 0.5 * torch.matmul(Y, I3 - torch.matmul(Z, Y)).to(self.device)
        y = ZY * torch.sqrt(normA.view(batch_size, 1, 1)).to(self.device)
        y = torch.mean(y, dim = 1).view(batch_size, channel, 1, 1).to(self.device)

        return y

    def forward(self, x):
        skip = self.identity(x)

        batch_size, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        M = h * w
        x = x.view(batch_size, c, M).to(self.device)
        Minv = torch.tensor(1 / M, dtype=torch.float32).to(self.device)
        I_hat = Minv * (torch.eye(M).to(self.device) - Minv * torch.ones((M, M)).to(self.device))
        cov = torch.matmul(torch.matmul(x, I_hat), x.transpose(1, 2)).to(self.device)

        y_cov = self.normalizeCov(cov, 5).to(self.device)
        y_cov = self.conv_du(y_cov).to(self.device)

        return y_cov * skip

# RCAB_dense_dilated_SOCA
class ResBlock(nn.Module):
    def __init__(self, filters, d=2):
        super(ResBlock, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.identity = nn.Identity()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same'),
            nn.ReLU()
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same', dilation = d),
            nn.ReLU()
        )
        self.conv2 = nn.Conv2d(filters * 2, filters, kernel_size=(3, 3), padding='same')
        self.channel_attention = SOCA(filters).to(self.device)

    def forward(self, x) :
        skip = self.identity(x)

        f1 = self.conv1_1(x)
        f2 = self.conv1_2(x)

        x = torch.cat([f1, f2], dim = 1)
        x = self.conv2(x)

        x = self.channel_attention(x)

        x = x + skip

        return x

# Dense_RG_SOCA
class ResGroup(nn.Module):
    def __init__(self, filters=64, n_blocks=5):
        super(ResGroup, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.identity = nn.Identity()
        self.blocks = nn.ModuleList([ResBlock(filters).to(self.device) for i in range(n_blocks)]).to(self.device)
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same')

    def forward(self, x):
        skip_conn = self.identity(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv1(x)
        x = x + skip_conn
        return x

'''
class Pixel_Shuffle(nn.Module):
    def __init__(self, scaling_factor):
        super(Pixel_Shuffle, self).__init__()
        self.scaling_factor = scaling_factor

    def forward(self, x):
        sf = self.scaling_factor

        b, c, h, w = x.size()
        c //= sf ** 2

        x = x.view(b, c, h, w, sf, sf)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(b, h * sf, w * sf, c)
        return x
'''

class Upscale_Module(nn.Module):
    def __init__(self, filters, scaling_factor):
        super(Upscale_Module, self).__init__()
        self.scaling_factor = scaling_factor
        if scaling_factor == 3:
            self.conv1 = nn.Conv2d(filters, filters*9, kernel_size=(1,1), padding='same')
            self.pixel_shuffle = nn.PixelShuffle(scaling_factor)

        elif scaling_factor & (scaling_factor - 1) == 0:
            self.log_scale_factor = int(torch.log2(torch.tensor(scaling_factor)))
            self.conv_layers = nn.ModuleList()
            self.pixel_shuffle_layers = nn.ModuleList()

            for i in range(self.log_scale_factor):
                conv_layer = nn.Conv2d(filters, filters*4, kernel_size=(1,1), padding='same')
                pixel_shuffle_layer = nn.PixelShuffle(2)
                self.conv_layers.append(conv_layer)
                self.pixel_shuffle_layers.append(pixel_shuffle_layer)

        else:
            raise NotImplementedError("Not Supported Scale Factor %d" % scaling_factor)

    def forward(self, x):
        if self.scaling_factor == 3:
            x = self.conv1(x)
            x = self.pixel_shuffle(x)

        elif self.scaling_factor & (self.scaling_factor - 1) == 0:
            for conv, pix in zip(self.conv_layers, self.pixel_shuffle_layers):
                x = conv(x)
                x = pix(x)

        return x

class AVRFN(nn.Module):
    def __init__(self, upscale, f=64, k=(3, 3), n_blocks=6, n_groups=3, channels=1):
        super(AVRFN, self).__init__()
        self.channels = channels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.identity = nn.Identity()
        self.conv_head = nn.Conv2d(channels, f, kernel_size=k, padding='same')
        self.conv_body = nn.Conv2d(f, f, kernel_size=k, padding='same')
        self.residual_groups = nn.ModuleList([ResGroup(f, n_blocks=n_blocks).to(self.device) for i in range(n_groups)]).to(self.device)
        self.Upscale_Module = Upscale_Module(f, upscale).to(self.device)
        self.conv_tail = nn.Conv2d(f, channels, kernel_size=(3, 3), padding='same')

    def forward(self, x):
        B, C, H, W = x.size()

        if self.channels == 1:
            if C != 1:
                x = x[:,0,:,:].view(B, 1, H, W)

        x = self.conv_head(x)
        skip = self.identity(x)
        for group in self.residual_groups:
            x = group(x)
        x = self.conv_body(x) + skip

        x = self.Upscale_Module(x)

        x = self.conv_tail(x)
        x = torch.clamp(x, 0, 255)

        if self.channels == 1:
            x = torch.concat([x,x,x], dim=1).to(self.device)

        return x