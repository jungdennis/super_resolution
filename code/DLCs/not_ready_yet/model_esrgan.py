##########################################################################################
#original code from https://github.com/xinntao/ESRGAN
#re-written by https://github.com/Lornatang/ESRGAN-PyTorch/blob/main/model.py
#
#check 
#       https://github.com/Lornatang/ESRGAN-PyTorch/blob/main/train_rrdbnet.py
#       and
#       https://github.com/Lornatang/ESRGAN-PyTorch/blob/main/config.py
#       for more details (Scale Factor = 4, Patch = 128x128)
#
#----------------------------------------------------------------------------------------
#How to use (RRDBNet / ESRGAN)
#   1. for Net(RRDBNet) train & test
#       #init
#       from model_esrgan import Generator as RRDBNet
#       model_sr  = RRDBNet()
#       pixel_criterion = torch.nn.L1Loss() #pytorch
#
#       #model & loss use
#       predicted = model_sr(input_data)
#       loss = pixel_criterion(predicted, input_data)
#
#
#
#   2. for GAN(ESRGAN) train & test
#       #init
#       from model_esrgan import Generator, Discriminator, ContentLoss
#       
#       pixel_criterion = torch.nn.L1Loss() #pytorch
#       content_criterion = ContentLoss(#feature extraction node
#                                       "features.34"
#                                       #feature_model_normalize_mean
#                                      ,[0.485, 0.456, 0.406]
#                                       #feature_model_normalize_std
#                                      ,[0.229, 0.224, 0.225]
#                                      )
#       adversarial_criterion = torch.nn.BCEWithLogitsLoss() #pytorch
#       
#       #model & loss use
#       model_g  = Generator()
#       model_d = Discriminator() #check input image size!
#       
#       tensor_g_hypo    = model_g(tensor_x)      #LR_4_noise10 input
#       tensor_d_hypo_hr = model_d(tensor_y)      #Original input
#       tensor_d_hypo_sr = model_d(tensor_g_hypo) #SR input
#       
#       tensor_d_answer_real = torch.full([current_batch_size, 1], 1.0, dtype=torch.float, device=torch.device(device))
#       tensor_d_answer_fake = torch.full([current_batch_size, 1], 0.0, dtype=torch.float, device=torch.device(device))
#
#       loss_d = (criterion_d(tensor_d_hypo_hr - torch.mean(tensor_d_hypo_sr), tensor_d_answer_real) * 0.5
#                +criterion_d(tensor_d_hypo_sr - torch.mean(tensor_d_hypo_hr), tensor_d_answer_fake) * 0.5
#                )
#       loss_g = (pixel_criterion(predicted, answer) * 0.01
#                +content_criterion(predicted, answer) * 1.0
#                +loss_d * 0.005
#                ) 
#
#
#
#
##########################################################################################


# <1> ***********************************************************************************************
# from https://github.com/Lornatang/ESRGAN-PyTorch/blob/main/model.py
# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms

__all__ = [
    "ResidualDenseBlock", "ResidualResidualDenseBlock",
    "Discriminator", "Generator",
    "ContentLoss"
]


class ResidualDenseBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.
    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels + growth_channels * 0, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(channels + growth_channels * 1, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(channels + growth_channels * 2, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(channels + growth_channels * 3, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(channels + growth_channels * 4, channels, (3, 3), (1, 1), (1, 1))

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.identity(self.conv5(torch.cat([x, out1, out2, out3, out4], 1)))
        out = torch.mul(out5, 0.2)
        out = torch.add(out, identity)

        return out


class ResidualResidualDenseBlock(nn.Module):
    """Multi-layer residual dense convolution block.
    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(ResidualResidualDenseBlock, self).__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(channels, growth_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = torch.mul(out, 0.2)
        out = torch.add(out, identity)

        return out


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 128 x 128
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 64 x 64
            nn.Conv2d(64, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 32 x 32
            nn.Conv2d(128, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 16 x 16
            nn.Conv2d(256, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 8 x 8
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 4 x 4
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        # The first layer of convolutional layer.
        self.conv1 = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))

        # Feature extraction backbone network.
        trunk = []
        for _ in range(23):
            trunk.append(ResidualResidualDenseBlock(64, 32))
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks.
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))

        # Upsampling convolutional layer.
        self.upsampling1 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )
        self.upsampling2 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        # Reconnect a layer of convolution block after upsampling.
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        # Output layer.
        self.conv4 = nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1))

    # The model should be defined in the Torch.script method.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(x)
        out = self.trunk(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling1(F.interpolate(out, scale_factor=2, mode="nearest"))
        out = self.upsampling2(F.interpolate(out, scale_factor=2, mode="nearest"))
        out = self.conv3(out)
        out = self.conv4(out)

        out = torch.clamp_(out, 0.0, 1.0)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.1
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

'''
class ContentLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.
    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.
     """

    def __init__(self, feature_model_extractor_node: str,
                 feature_model_normalize_mean: list,
                 feature_model_normalize_std: list) -> None:
        super(ContentLoss, self).__init__()
        # Get the name of the specified feature extraction node
        self.feature_model_extractor_node = feature_model_extractor_node
        # Load the VGG19 model trained on the ImageNet dataset.
        model = models.vgg19(True)
        # Extract the thirty-fifth layer output in the VGG19 model as the content loss.
        self.feature_extractor = create_feature_extractor(model, [feature_model_extractor_node])
        # set to validation mode
        self.feature_extractor.eval()

        # The preprocessing method of the input data.
        # This is the VGG model preprocessing method of the ImageNet dataset
        self.normalize = transforms.Normalize(feature_model_normalize_mean, feature_model_normalize_std)

        # Freeze model parameters.
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False

    def forward(self, sr_tensor: torch.Tensor, hr_tensor: torch.Tensor) -> torch.Tensor:
        # Standardized operations
        sr_tensor = self.normalize(sr_tensor)
        hr_tensor = self.normalize(hr_tensor)

        sr_feature = self.feature_extractor(sr_tensor)[self.feature_model_extractor_node]
        hr_feature = self.feature_extractor(hr_tensor)[self.feature_model_extractor_node]

        # Find the feature map difference between the two images
        content_loss = F.l1_loss(sr_feature, hr_feature)

        return content_loss
'''


# <2> ***********************************************************************************************

#fixed for loss calc with "Automatic Mixed Precision" -> not verified yet
class ContentLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.
    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.
     """

    def __init__(self, feature_model_extractor_node: str,
                 feature_model_normalize_mean: list,
                 feature_model_normalize_std: list) -> None:
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        super(ContentLoss, self).__init__()
        # Get the name of the specified feature extraction node
        self.feature_model_extractor_node = feature_model_extractor_node
        # Load the VGG19 model trained on the ImageNet dataset.
        model = models.vgg19(True)
        model.to(device)
        # Extract the thirty-fifth layer output in the VGG19 model as the content loss.
        self.feature_extractor = create_feature_extractor(model, [self.feature_model_extractor_node])
        # set to validation mode
        self.feature_extractor.eval()
        
        # Freeze model parameters.
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False
        
        # The preprocessing method of the input data.
        # This is the VGG model preprocessing method of the ImageNet dataset
        self.normalize = transforms.Normalize(feature_model_normalize_mean, feature_model_normalize_std)
        
    
    def forward(self, sr_tensor: torch.Tensor, hr_tensor: torch.Tensor, is_AMP: bool) -> torch.Tensor:
        # Standardized operations
        sr_tensor = self.normalize(sr_tensor)
        hr_tensor = self.normalize(hr_tensor)
        
        #does loss calclation processed under "torch.cuda.amp.autocast(enabled=True):" effect?
        #print("is_AMP", is_AMP)
        #print("self.prev_is_AMP", self.prev_is_AMP)
        
        if is_AMP:
            sr_feature = self.feature_extractor(sr_tensor.to(torch.float32))[self.feature_model_extractor_node]
            hr_feature = self.feature_extractor(hr_tensor.to(torch.float32))[self.feature_model_extractor_node]
        else:
            sr_feature = self.feature_extractor(sr_tensor)[self.feature_model_extractor_node]
            hr_feature = self.feature_extractor(hr_tensor)[self.feature_model_extractor_node]
        
        with torch.cuda.amp.autocast(enabled=is_AMP):
            # Find the feature map difference between the two images
            content_loss = F.l1_loss(sr_feature, hr_feature)
        
        return content_loss

print("End of model_esrgan.py")