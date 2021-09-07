import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import padding

# https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution/blob/master/models.py
class SubPixelConvolutionalBlock(nn.Module):
    """
    A subpixel convolutional block, comprising convolutional, pixel-shuffle, and PReLU activation layers.
    """

    def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2):
        """
        :param kernel_size: kernel size of the convolution
        :param n_channels: number of input and output channels
        :param scaling_factor: factor to scale input images by (along both dimensions)
        """
        super(SubPixelConvolutionalBlock, self).__init__()

        # A convolutional layer that increases the number of channels by scaling factor^2, followed by pixel shuffle and PReLU
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * (scaling_factor ** 2),
                              kernel_size=kernel_size, padding=kernel_size // 2, padding_mode="circular")
        # These additional channels are shuffled to form additional pixels, upscaling each dimension by the scaling factor
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        self.prelu = nn.PReLU()

    def forward(self, input):
        """
        Forward propagation.
        :param input: input images, a tensor of size (N, n_channels, w, h)
        :return: scaled output images, a tensor of size (N, n_channels, w * scaling factor, h * scaling factor)
        """
        output = self.conv(input)  # (N, n_channels * scaling factor^2, w, h)
        output = self.pixel_shuffle(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
        output = self.prelu(output)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return output


class SuperResolutionNet(nn.Module):

    def __init__(self, scaling_factor=4, num_layers=3):
        super().__init__()

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, 
                                        stride=1, padding=1, padding_mode="circular"))
            else:
                layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, 
                                        stride=1, padding=1, padding_mode="circular"))
            layers.append(nn.ReLU())
        
        n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
        for i in range(n_subpixel_convolution_blocks):
            layers.append(
                SubPixelConvolutionalBlock(kernel_size=3, n_channels=64, scaling_factor=2))
        
        layers.append(nn.Conv2d(in_channels=64, out_channels=1, kernel_size=9, stride=1, 
                                padding=4, padding_mode="circular"))
        
        self.model = nn.Sequential(*layers)
        self.scaling_factor = scaling_factor
    
    def forward(self, field2d):
        input_shape = field2d.shape
        if(len(input_shape) == 3):
            bs, h, w = field2d.shape
            field2d = field2d.reshape((bs, 1, h, w))
    
        highres_field2d = self.model(field2d)
        bs, _, ho, wo = highres_field2d.shape
        residual = F.interpolate(field2d, (ho, wo), mode="nearest")
        output = highres_field2d + residual

        if(len(input_shape) == 3):
            output = output.reshape((bs, ho, wo))
        return output