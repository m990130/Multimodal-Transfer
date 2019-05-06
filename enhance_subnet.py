# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 15:55:03 2018

@author: pingc
"""

import torch
import torch.nn.functional as F
from style_subnet import ResidualBlock, ConvLayer, Resize_Conv


class Enhance_Subnet(torch.nn.Module):
    """
        The enhance subnet has a similar structure as the style subnet. The only 
        difference is taht the enhance subnet has one more ConvLayer(with stride = 2) 
        for downsampling and one more Resize_Conv(with scale = 2) for upsampling, 
        since its input is bigger than style subnet's input. (512 compared to 256)
    
        Besides, during the training process, the enhance subnet has a larger style 
        weight than its predecessor, which makes it more sensitive to the stylization.
        
        For simplicity I won't define the small block separately but all in this class.
    """

    def __init__(self):
            super(Enhance_Subnet, self).__init__()
            # Our input has the size of (1, 3, 512, 512)
            
            # RGB_Block
            self.rgb_conv1 = ConvLayer(3, 32, kernel_size=9, stride=1) # (1, 32, 512, 512)
            self.rgb_in1 = torch.nn.InstanceNorm2d(32, affine=True)
            self.rgb_conv2 = ConvLayer(32, 64, kernel_size=3, stride=2) # (1, 64, 256, 256)
            self.rgb_in2 = torch.nn.InstanceNorm2d(64, affine=True)
            self.rgb_conv3 = ConvLayer(64, 128, kernel_size=3, stride=2) # (1, 128, 128, 128)
            self.rgb_in3 = torch.nn.InstanceNorm2d(128, affine=True)
            # one more ConvLayer to downsampling
            self.rgb_conv4 = ConvLayer(128, 256, kernel_size=3, stride=2) # (1, 256, 64, 64)
            self.rgb_in4 = torch.nn.InstanceNorm2d(256, affine=True)
            
            self.rgb_res1 = ResidualBlock(256)
            self.rgb_res2 = ResidualBlock(256)
            self.rgb_res3 = ResidualBlock(256)
            
            self.conv_res1 = ResidualBlock(256)
            self.conv_res2 = ResidualBlock(256)
            self.conv_res3 = ResidualBlock(256)
            
            # Upsampling layers, one more than Style Subnet
            self.conv_upsam1 = Resize_Conv(256, 128, kernel_size=3, stride=1, scale=2) # (1, 128, 128, 128)
            self.conv_in1 = torch.nn.InstanceNorm2d(128, affine=True)
            self.conv_upsam2 = Resize_Conv(128, 64, kernel_size=3, stride=1, scale=2) # (1, 64, 256, 256)
            self.conv_in2 = torch.nn.InstanceNorm2d(64, affine=True)
            self.conv_upsam3 = Resize_Conv(64, 32, kernel_size=3, stride=1, scale=2) # (1, 32, 512, 512)
            self.conv_in3 = torch.nn.InstanceNorm2d(32, affine=True)
            
            self.conv_upsam4 = ConvLayer(32, 3, kernel_size=3, stride=1)
            
            
            self.relu = torch.nn.ReLU()
            
    def forward(self, x):
        # x is the ouput of style subnet with size (1, 3, 256, 256)
        
        # neareast neighbor upsampling
        
        # neareast neighbor upsampling
        with torch.no_grad(): x = F.interpolate(x, scale_factor=2, mode='nearest') # (1, 3, 512, 512)
        
        
        x_rgb = self.relu(self.rgb_in1(self.rgb_conv1(x)))
        x_rgb = self.relu(self.rgb_in2(self.rgb_conv2(x_rgb)))
        x_rgb = self.relu(self.rgb_in3(self.rgb_conv3(x_rgb)))
        x_rgb = self.relu(self.rgb_in4(self.rgb_conv4(x_rgb)))
        
        x_rgb = self.rgb_res1(x_rgb)
        x_rgb = self.rgb_res2(x_rgb)
        x_rgb = self.rgb_res3(x_rgb)
        

        output = self.conv_res1(x_rgb)
        output = self.conv_res2(output)
        output = self.conv_res3(output)

        output = self.relu(self.conv_in1(self.conv_upsam1(output)))
        output = self.relu(self.conv_in2(self.conv_upsam2(output)))
        output = self.relu(self.conv_in3(self.conv_upsam3(output)))
        
        y = self.conv_upsam4(output)
        
        y[0][0].clamp_((0-0.485)/0.299, (1-0.485)/0.299)
        y[0][1].clamp_((0-0.456)/0.224, (1-0.456)/0.224)
        y[0][2].clamp_((0-0.406)/0.225, (1-0.406)/0.225)      
        
        return y, x