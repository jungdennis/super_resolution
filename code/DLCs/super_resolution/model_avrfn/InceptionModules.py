import torch
import torch.nn as nn

class InceptionModule(nn.Module) :
    def __init__(self, filters, reduction=1, name=None) :
        super(InceptionModule, self).__init__()
        self.merge2D_1 = nn.Conv2d(in_channels=filters, out_channels=filters // 2, kernel_size=(1, 1), padding='same')
        self.merge2D_2 = nn.Conv2d(in_channels=filters, out_channels=filters // 2, kernel_size=(3, 3), padding='same')
        self.merge2D_3 = nn.Conv2d(in_channels=filters, out_channels=filters // 2, kernel_size=(5, 5), padding='same')
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding='same')

    def forward(self, x) :
        a = self.merge2D_1(x)
        b = self.merge2D_2(x)
        c = self.merge2D_3(x)
        d = self.pool(x)
        return torch.cat([a, b, c, d], dim=1)


class InceptDilated(nn.Module) :
    def __init__(self, filters, reduction=2, name=None) :
        super(InceptDilated, self).__init__()
        self.merge2D_1 = nn.Sequential(
            nn.Conv2d(in_channels=filters, out_channels=filters // reduction, kernel_size=(1, 1), padding='same'),
            nn.ReLU()
        )
        self.merge2D_2 = nn.Sequential(
            nn.Conv2d(in_channels=filters, out_channels=filters // reduction, kernel_size=(3, 3), padding='same', dilation=1),
            nn.ReLU()
        )
        self.merge2D_3 = nn.Sequential(
            nn.Conv2d(in_channels=filters, out_channels=filters // reduction, kernel_size=(5, 5), padding='same', dilation=1),
            nn.RuLU()
        )

    def forward(self, x) :
        a = self.merge2D_1(x)
        b = self.merge2D_2(x)
        c = self.merge2D_3(x)
        return torch.cat([a, b, c], dim=1)
