import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np


# Channel Attention Layer
class Channel_Attention(nn.Module) :
    def __init__(self, filters, reduction = 1) :
        super(Channel_Attention, self).__init__()

        self.conv2D_1 = nn.Sequential(
            nn.Conv2d(filters, filters // reduction, kernel_size = (1, 1), padding = 'same'),
            nn.ReLU()
        )
        self.conv2D_2 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=(1, 1), padding='same'),
            nn.ReLU()
        )
        self.identity = nn.Identity()

    def foward(self, x) :
        skip_conn = self.identity(x)
        channel = x.shape[1]

        x = torch.reshape(torch.mean(x, (2, 3)), (-1, channel, 1, 1))
        x = self.conv2D_1(x)
        x = self.conv2D_2(x)
        x = torch.multiply(skip_conn, x)

        return x

# Channel Attention Layer which returns vector if scalars
class Scalar_CA(nn.Module) :
    def __init__ (self, filters, reduction = 1) :
        super(Scalar_CA, self).__init__()

        self.conv2D_1_relu = nn.Sequential(
            nn.Conv2d(filters, filters // reduction, kernel_size = (1, 1), padding = 'same'),
            nn.ReLU()
        )
        self.conv2D_1_sig = nn.Sequential(
            nn.Conv2d(filters, filters // reduction, kernel_size=(1, 1), padding='same'),
            nn.Sigmoid()
        )
        self.identity = nn.Identity()
        self.dense_1 = nn.Linear(2)

    def foward(self, x) :
        skip_conn = self.identity(x)
        channel = x.shape[1]

        x = torch.reshape(torch.mean(x, (2, 3)), (-1, channel, 1, 1))
        x = self.conv2D_1_relu(x)
        x = self.conv2D_1_sig(x)
        x = torch.reshape(x, (-1, channel))
        x = self.dense_1(x)

        return x

class Scale_Attention(nn.Module) :
    def __init__(self, filters, reduction = 1) :
        super(Scale_Attention, self).__init__()

        # local attention
        self.merge2D_1 = nn.Conv2d(1, kernel_size = (7), padding = "same", stride = 1)

        self.conv2D_1_linear = nn.Conv2d(filters, filters // reduction, kernel_size = (1, 1), padding = 'same')
        self.conv2D_1_sig = nn.Sequential(
            nn.Conv2d(filters, filters // reduction, kernel_size=(1, 1), padding='same'),
            nn.Sigmoid()
        )
        self.identity = nn.Identity()

    def foward(self, x) :
        skip_conn = self.identity(x)

        a = self.merge2D_1(x)
        x = self.conv2D_1_linear(a)
        x = self.conv2D_1_sig(x)

        return torch.multiply(skip_conn, x)

# Second Order Channel Attention Layer
class SOCA(nn.Module) :
    def __init__(self, filters, reduction = 1, input_shape = (48, 48)) :
        super(SOCA, self).__init__()

        self.conv_du = nn.Sequential(
            nn.Conv2d(filters, filters // reduction, kernel_size = (3, 3), padding = "same"),
            nn.ReLU(),
            nn.Conv2d(filters, filters, kernel_size = (3, 3), padding = "same"),
            nn.Sigmoid()
        )

        h, w = input_shape[0], input_shape[1]
        h, w = min(80, h), min(80, w)

        self.crop = transforms.Compose([
            transforms.CenterCrop(h, w)
        ])

    def normalizeCov(self, x, iterN) :
        batch_size, channel = x.shape[0], x.shape[1]
        h, w = x.shape[2], x.shape[3]

        I3 = torch.eye(channel, channel)
        I3 = I3.reshape((1, channel, channel))
        I3 = I3.repeat(batch_size, 1, 1)
        I3 = 3 * I3

        normA = torch.multiply((1/3), torch.sum(torch.multiply(x, I3), axis = [2, 3]))
        A = x / torch.reshape(normA, (batch_size, 1, 1))
        Y = torch.zeros((batch_size, channel, channel))

        Z = torch.eye(channel, channel)
        Z = Z.reshape((1, channel, channel))
        Z = Z.repeat(batch_size, 1, 1)

        ZY = 0.5 * (I3 - A)
        Y = torch.matmul(A, ZY)
        Z = ZY

        for i in range(1, iterN - 1) :
            ZY = 0.5 * (I3 - torch.matmul(Z, Y))
            Y = torch.matmul(Y, ZY)
            Z = torch.matmul(ZY, Z)
        ZY = 0.5 * torch.matmul(Y, I3 - torch.matmul(Z, Y))
        y = ZY * torch.sqrt(normA.view(batch_size, channel, 1, 1))
        y = torch.mean(y, dim = 1).view(batch_size, channel, 1, 1)

        return self.conv_du(y)

    def forward(self, x):
        crop = transforms.CenterCrop([
            transforms.CenterCrop((min(48, x.shape[2]), min(48, x.shape[3])))
        ])
        x = crop(x)
        x_sub = x.unsqueeze(1)

        h1, w1 = 200, 200
        c, h, w = x_sub.shape[1], x_sub.shape[2], x_sub.shape[3]
        batch_size = x.shape[0]
        M = h * w
        x_sub = x_sub.view(batch_size, c, M)
        Minv = torch.tensor(1 / M, dtype=torch.float32)
        I_hat = Minv * (torch.eye(M) - Minv * torch.ones((M, M)))
        cov = torch.matmul(torch.matmul(x_sub, I_hat), x_sub.transpose(1, 2))

        y_cov = self.normalizeCov(cov, 5)
        return y_cov * x