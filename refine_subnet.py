# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 18:19:58 2018

@author: pingc
"""

import torch
import torch.nn.functional as F
from style_subnet import ResidualBlock, ConvLayer, Resize_Conv


class Refine_Subnet(torch.nn.Module):
    """
        The refine subnet consists of three ConvLayer, three Res, two Resize_Conv
        and one last ConvLayer to obtain the final output.
        
        Note that the upsampling layer between enhance subnet and refine subnet 
        is only inserted at test time, so during training the input to refine 
        subnet is still of size 512, which hugely reduces the required
        memory and speeds up the training process.
    """

    def __init__(self):
            super(Refine_Subnet, self).__init__() 
            
            
            self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1) # train: 512->256->128 | test: 1024->512->256
            self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
            self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
            self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
            self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2) 
            self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
            
            # Residual BLock
            self.res1 = ResidualBlock(128)
            self.res2 = ResidualBlock(128)
            self.res3 = ResidualBlock(128)
            
            self.upsam1 = Resize_Conv(128, 64, kernel_size=3, stride=1, scale=2) 
            self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
            self.upsam2 = Resize_Conv(64, 32, kernel_size=3, stride=1, scale=2) 
            self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
            self.upsam3 = ConvLayer(32, 3, kernel_size=3, stride=1) 
            
            self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        # the input has size of (1, 3, 512, 512)
        
        #resized_x = x.clone()
        in_x = x
        residual = in_x.clone()
        
        y = self.relu(self.in1(self.conv1(in_x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        
        y = self.relu(self.in4(self.upsam1(y)))
        y = self.relu(self.in5(self.upsam2(y)))
        y = self.upsam3(y)            
        
        # Identity connection
        y = y + residual
        
        y[0][0].clamp_((0-0.485)/0.299, (1-0.485)/0.299)
        y[0][1].clamp_((0-0.456)/0.224, (1-0.456)/0.224)
        y[0][2].clamp_((0-0.406)/0.225, (1-0.406)/0.225)
        
        return y, residual