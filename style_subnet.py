# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 11:42:24 2018

@author: pingc
"""
import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

class ResidualBlock(torch.nn.Module):
    """
        ResidualBlock
        introduced in: https://arxiv.org/abs/1512.03385
        recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out
    
    
class ConvLayer(torch.nn.Module):
    """
        out = ((in+2*padding-(kernel_size-1)-1)/stride)+1
        
        when kernel_size is: odd number, the output is only affected by stride
                             even number, use the formula.
        
        For example: if kernel size is odd, and with stride 1 then out = in, but
        channel changes. With stride 2, out is halved.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out
    
    
class Resize_Conv(torch.nn.Module):
    """
        This layer consists of a interpolation(not learnable) layer for upsampling
        or downsampling and a convolutional layer.
        
        For upsampling use mode='nearest', for downsampling use mode='bilinear',
        and also set align_corners=True.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, scale=None, mode='nearest'):
        super(Resize_Conv, self).__init__()
        self.scale = scale
        self.mode = mode
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        
    def forward(self, x):
        x_in = x
        if self.scale:
            # This will only affect the size of image. Conv will affect the channel
            # as well as the size of the images.
            x_in = F.interpolate(x, scale_factor=self.scale, mode=self.mode)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
        
        
class RGB_Block(torch.nn.Module):
    """
        In reference to: https://arxiv.org/pdf/1612.01895.pdf
        The RGB_Block comprises three strided(1, 2, 2) convolutional layers (
        kernel size: 9, 3, 3 respectively, the later are used for downsampling)
        and three residual blocks. Besides, all non-residual convolutional layers
        are followed by instance normalization and ReLU.
        
        NOTE: the paper didn't give the size of kernel and stride explicitly, hence
        I follow the convention of Johnson's paper.
        
    """
    
    def __init__(self):
        super(RGB_Block, self).__init__()
        # Each conv layer follow by a instance normalization
        self.conv1 = ConvLayer(3, 16, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(16, affine=True)
        self.conv2 = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv3 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(64, affine=True)
        
        # Then 3 residual blocks
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        
        # ReLU
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        return y
    
    
class L_Block(torch.nn.Module):
    """
        Has the same structure to RGB_Block except  that the channel of the first
        convolution layer is different.
    """
    

    def __init__(self):
        super(L_Block, self).__init__()
        # Each conv layer follow by a instance normalization
        self.conv1 = ConvLayer(1, 16, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(16, affine=True)
        self.conv2 = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv3 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(64, affine=True)
        
        # Then 3 residual blocks
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        
        # ReLU
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        return y
    
    
class Conv_Block(torch.nn.Module):
    """
        Take the concatenated(in channel) output of RGB_Block and L_Block. Hence 
        the input has shape of (batch, 128*2, 64, 64). This is based on input image
        has size of (256, 256).
        Consists of three residual blocks, two resize-convolution layers for 
        upsampling and a last 3*3 convolutional layer to obtain the output RGB,
        namely 3 channels, image y_hat.
    """
    
    def __init__(self):
        super(Conv_Block, self).__init__()
        # Three res take concatenated RGBL as input
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        
        # Upsampling layers
        self.upsam1 = Resize_Conv(128, 64, kernel_size=3, stride=1, scale=2)
        self.in1 = torch.nn.InstanceNorm2d(64, affine=True)
        self.upsam2 = Resize_Conv(64, 32, kernel_size=3, stride=1, scale=2)
        self.in2 = torch.nn.InstanceNorm2d(32, affine=True)
        # in the paper kernel size = 3
        self.upsam3 = ConvLayer(32, 3, kernel_size=3, stride=1)
        
        # ReLU
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        y = self.res1(x)
        y = self.res2(y)
        y = self.res3(y)
        
        y = self.relu(self.in1(self.upsam1(y)))
        y = self.relu(self.in2(self.upsam2(y)))
        y = self.upsam3(y)
        # The return y should be our first y_hat output with size of (256, 256)
        return y

class Style_Subnet(torch.nn.Module):
    """
        input:
                RGB images with shape of (batch_size, 3, 512, 512)
        output:
                RGB images with shape of (batch_size, 3, 256, 256)
                
        Pipeline: 
            1.input first downsampled by
              F.interpolate(input, scale_factor=0.5, mode='bilinear', align_corners=True)
              
            2.Then goes separately to RGB_Block and L_Block. Transform RGB to Gray before
            pass to L_Block. For Luminance channel I refer to this paper: https://arxiv.org/pdf/1606.05897.pdf.
            They state by using YIQ space, that Y represents the luminance. 
            
            3. Concatenate outputs of RGB_Block and L_Block, and pass it to Conv_Block.
            
            4. Output stylized image(256): y_hat1.
    """
    
    def __init__(self):
        super(Style_Subnet, self).__init__()
        # use this tensor to get luminance later
        self.gray = torch.nn.Parameter(torch.tensor([[[[0.2989]],
                                                      [[0.5870]],
                                                      [[0.1140]]]]))
        self.rgb_block = RGB_Block()
        self.l_block = L_Block()
        self.conv_block = Conv_Block()
        
    def forward(self, x):       
        # using bilinear interpolation get downsampled image 
        #resized_x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        
        #resized_x = x.clone()
        x_rgb = x
        
        # the input for RGB_Block and L_Block
        with torch.no_grad(): x_l = F.conv2d(x.clone(), self.gray)
        
        y_rgb = self.rgb_block(x_rgb)
        y_l = self.l_block(x_l)
        
        x_conv = torch.cat((y_rgb, y_l), dim=1)
        y = self.conv_block(x_conv)
                
        # Clamp the value into range [0,1] AFTER denormalization
        # Note: It will be back to [0,1] after denormalization but y
        # won't be in range[0,1]. So deno first when plot them.
        # the number outside the range will set to min/max value of it
        # R(-1.6221, 1.7224), G(-2.0357, 2.4286) B(-1.8044, 2.6400)
        
        y[0][0].clamp_((0-0.485)/0.299, (1-0.485)/0.299)
        y[0][1].clamp_((0-0.456)/0.224, (1-0.456)/0.224)
        y[0][2].clamp_((0-0.406)/0.225, (1-0.406)/0.225)
        
        # we need resized_x for calculating the first content loss
        return y, x
    