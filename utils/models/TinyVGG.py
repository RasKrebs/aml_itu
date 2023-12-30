# Library imports
import torch
import torch
from math import ceil
from torch import nn
import torchvision
torchvision.disable_beta_transforms_warning()


# Convolution block
class ConvBlock(nn.Module):
    def __init__(self, in_channesl, out_channls, kernel_size = 3, stride=1, pool_kernel = 2, dropout_rate = .2, padding=1, skip_last_dropout=False):
        super(ConvBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=in_channesl,
                              out_channels=out_channls,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding),
            nn.SELU(True),
            nn.BatchNorm2d(out_channls),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=out_channls,
                              out_channels=out_channls,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding),
            nn.SELU(True),
            nn.BatchNorm2d(out_channls),
            nn.MaxPool2d(kernel_size = pool_kernel),
            nn.Dropout(dropout_rate) if not skip_last_dropout else nn.Identity())

    def forward(self, x):
        return self.main(x)

# Fully Connected Dense Block
class Dense(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate):
        super(Dense, self).__init__()
        
        self.main = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.ReLU(True),
            nn.BatchNorm1d(out_features),
            nn.Dropout(dropout_rate),
        )
    
    def forward(self, x):
        return self.main(x)

# Smallest version
class TinyVGG(torch.nn.Module):
    def __init__(self, filters = 32, num_classes = 10, kernel_size = 3,  stride = 1, in_channels = 3, pool_kernel_size = 2, padding=1):
        super(TinyVGG, self).__init__()
        # First Convolution Block
        self.main = nn.Sequential(
            ConvBlock(in_channesl=in_channels, out_channls=filters, kernel_size=kernel_size, stride=stride, pool_kernel=pool_kernel_size, dropout_rate=.25),
            ConvBlock(in_channesl=filters, out_channls=filters*2, kernel_size=kernel_size, stride=stride, pool_kernel=pool_kernel_size, dropout_rate=.3),
            ConvBlock(in_channesl=filters*2, out_channls=filters*2, kernel_size=kernel_size, stride=stride, pool_kernel=pool_kernel_size, dropout_rate=.3),
            nn.Flatten(),   
            Dense(in_features=3072, out_features=128, dropout_rate=.3), # img size L = 52224 # Old small 12288, 3 channels medium sized input images 6144 - Img size L with stride 2 = 26112
            nn.Linear(in_features=128, out_features=num_classes)) # img size L = 52224 # Old small 12288

    def forward(self, x):
        # Pass the data through the convolutional blocks
        x = self.main(x)
        return x