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

    def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2, activation=None):
        """
        :param kernel_size: kernel size of the convolution
        :param n_channels: number of input and output channels
        :param scaling_factor: factor to scale input images by (along both dimensions)
        """
        super(SubPixelConvolutionalBlock, self).__init__()

        # A convolutional layer that increases the number of channels by scaling factor^2, followed by pixel shuffle and PReLU
        conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * (scaling_factor ** 2),
                         kernel_size=kernel_size, padding=kernel_size // 2, padding_mode="circular")
        # These additional channels are shuffled to form additional pixels, upscaling each dimension by the scaling factor
        pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)

        layers = [conv, pixel_shuffle]

        # An activation layer, if wanted
        if activation is None:
            pass
        elif activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        else:
            raise ValueError("Recognitized activation %s" %activation)

        
        self.subpixel_conv = nn.Sequential(*layers)

    def forward(self, input):
        """
        Forward propagation.
        :param input: input images, a tensor of size (N, n_channels, w, h)
        :return: scaled output images, a tensor of size (N, n_channels, w * scaling factor, h * scaling factor)
        """
        output = self.subpixel_conv(input)  # (N, n_channels * scaling factor^2, w, h)
        return output


class ConvolutionalBlock(nn.Module):
    """
    A convolutional block, comprising convolutional, BN, activation layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channe;s
        :param kernel_size: kernel size
        :param stride: stride
        :param batch_norm: include a BN layer?
        :param activation: Type of activation; None if none
        """
        super(ConvolutionalBlock, self).__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh'}

        # A container that will hold the layers in this convolutional block
        layers = list()

        # A convolutional layer
        layers.append(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, padding_mode="circular"))

        # A batch normalization (BN) layer, if wanted
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        # An activation layer, if wanted
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        # Put together the convolutional block as a sequence of the layers in this container
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        """
        Forward propagation.
        :param input: input images, a tensor of size (N, in_channels, w, h)
        :return: output images, a tensor of size (N, out_channels, w, h)
        """
        output = self.conv_block(input)  # (N, out_channels, w, h)

        return output


class ResidualBlock(nn.Module):
    """
    A residual block, comprising two convolutional blocks with a residual connection across them.
    """

    def __init__(self, kernel_size=3, n_channels=64):
        """
        :param kernel_size: kernel size
        :param n_channels: number of input and output channels (same because the input must be added to the output)
        """
        super(ResidualBlock, self).__init__()

        # The first convolutional block
        self.conv_block1 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation='PReLu')

        # The second convolutional block
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation=None)

    def forward(self, input):
        """
        Forward propagation.
        :param input: input images, a tensor of size (N, n_channels, w, h)
        :return: output images, a tensor of size (N, n_channels, w, h)
        """
        residual = input  # (N, n_channels, w, h)
        output = self.conv_block1(input)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)
        output = output + residual  # (N, n_channels, w, h)

        return output


class SuperResolutionNet(nn.Module):

    def __init__(self, large_kernel_size=9, small_kernel_size=3, scaling_factor=4, num_layers=3, apply_interp=False):
        super().__init__()

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(ConvolutionalBlock(in_channels=1, out_channels=64, kernel_size=large_kernel_size,
                                                 batch_norm=False, activation="PReLU"))
            else:
                layers.append(ResidualBlock(kernel_size=small_kernel_size, n_channels=64))
        
        layers.append(ConvolutionalBlock(in_channels=64, out_channels=64, 
                                         kernel_size=small_kernel_size, batch_norm=True, activation=None))
        
        n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
        for i in range(n_subpixel_convolution_blocks):
            layers.append(
                SubPixelConvolutionalBlock(kernel_size=small_kernel_size, n_channels=64, scaling_factor=2))
        
        layers.append(ConvolutionalBlock(in_channels=64, out_channels=1, kernel_size=large_kernel_size,
                                         batch_norm=False, activation=None))
        
        self.model = nn.Sequential(*layers)
        self.scaling_factor = scaling_factor
        self.apply_interp = apply_interp
    
    def forward(self, field2d):
        input_shape = field2d.shape
        if(len(input_shape) == 3):
            bs, h, w = field2d.shape
            field2d = field2d.reshape((bs, 1, h, w))
    
        highres_field2d = self.model(field2d)
        bs, _, ho, wo = highres_field2d.shape
        if self.apply_interp:
            residual = F.interpolate(field2d, (ho, wo), mode="bilinear")
            highres_field2d += residual

        if(len(input_shape) == 3):
            highres_field2d = highres_field2d.reshape((bs, ho, wo))
        return highres_field2d


class Generator(nn.Module):
    """
    The generator in the SRGAN, as defined in the paper. Architecture identical to the SRResNet.
    """

    def __init__(self, large_kernel_size=9, small_kernel_size=3, scaling_factor=4, num_layers=3):
        """
        """
        super(Generator, self).__init__()

        # The generator is simply an SRResNet, as above
        self.net = SuperResolutionNet(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size,
                                      num_layers=num_layers, scaling_factor=scaling_factor)

    def initialize_with_srresnet(self, srresnet_checkpoint):
        """
        Initialize with weights from a trained SRResNet.
        :param srresnet_checkpoint: checkpoint filepath
        """
        srresnet = torch.load(srresnet_checkpoint)['model']
        self.net.load_state_dict(srresnet.state_dict())

        print("\nLoaded weights from pre-trained SRResNet.\n")

    def forward(self, lr_imgs):
        """
        Forward prop.
        :param lr_imgs: low-resolution input images, a tensor of size (N, 3, w, h)
        :return: super-resolution output images, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        """
        sr_imgs = self.net(lr_imgs)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return sr_imgs


class Discriminator(nn.Module):
    """
    The discriminator in the SRGAN, as defined in the paper.
    """

    def __init__(self, kernel_size=3, n_channels=64, n_blocks=8, fc_size=1024):
        """
        :param kernel_size: kernel size in all convolutional blocks
        :param n_channels: number of output channels in the first convolutional block, after which it is doubled in every 2nd block thereafter
        :param n_blocks: number of convolutional blocks
        :param fc_size: size of the first fully connected layer
        """
        super(Discriminator, self).__init__()

        in_channels = 1

        # A series of convolutional blocks
        # The first, third, fifth (and so on) convolutional blocks increase the number of channels but retain image size
        # The second, fourth, sixth (and so on) convolutional blocks retain the same number of channels but halve image size
        # The first convolutional block is unique because it does not employ batch normalization
        conv_blocks = list()
        for i in range(n_blocks):
            out_channels = (n_channels if i is 0 else in_channels * 2) if i % 2 is 0 else in_channels
            conv_blocks.append(
                ConvolutionalBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=1 if i % 2 is 0 else 2, batch_norm=i is not 0, activation='LeakyReLu'))
            in_channels = out_channels
        self.conv_blocks = nn.Sequential(*conv_blocks)

        # An adaptive pool layer that resizes it to a standard size
        # For the default input size of 96 and 8 convolutional blocks, this will have no effect
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Linear(out_channels * 6 * 6, fc_size)

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.fc2 = nn.Linear(1024, 1)

        # Don't need a sigmoid layer because the sigmoid operation is performed by PyTorch's nn.BCEWithLogitsLoss()

    def forward(self, imgs):
        """
        Forward propagation.
        :param imgs: high-resolution or super-resolution images which must be classified as such, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        :return: a score (logit) for whether it is a high-resolution image, a tensor of size (N)
        """
        batch_size = imgs.size(0)
        output = self.conv_blocks(imgs)
        output = self.adaptive_pool(output)
        output = self.fc1(output.view(batch_size, -1))
        output = self.leaky_relu(output)
        logit = self.fc2(output)

        return logit