import os
import math
import torch
import torch.nn as nn
import numpy as np
import sys
from collections import namedtuple

# wavenet_info_constructor = namedtuple("wavenet_info","layers blocks 


# Inpl is, for the time being, based on user vincentherrmann's implementation:
# I will refactor this after I fully comprehend everything
# but constants are hardcoded for now, for ease of work in sajibang environment.
# link : https://github.com/vincentherrmann/pytorch-wavenet/blob/master/wavenet_model.py
class WaveNetModel(nn.Module):
 def __init__(self, layerinfo):
  super(WaveNetModel,self).__init()
  self.layers = 10
  self.blocks = 4
  self.dilation_channels = 32
  self.residual_channels = 32
  self.skip_channels = 256
  self.classes = 256
  self.kernel_size = 2
  self.dtype = dtype
  self.bias=bias
  # 1*1 conv to create channels.
  self.conv_one = nn.Conv1d(in_channels = self.classes,
                            out_channels = residual_channels,
                            kernel_size = 1,
                            bias = self.bias) # dilation is 1.

  # dilating convolutions
  self.filter_convs=[]
  self.gate_convs = []
  self.residual_convs = []
  self.skip_convs = []
  for b in range(self.blocks):
   dilation = 1
   for i in range(self.layers):
   self.filter_convs.append(nn.Conv1d(in_channels=residual_channels,
                                      out_channels=dilation_channels,
                                      kernel_size=kernel_size,dilation=dilation
                                      bias=self.bias))
   self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                      out_channels=dilation_channels,
                                      kernel_size=kernel_size,dilation=dilation
                                      bias=self.bias))
   self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                      out_channels=residual_channels,
                                      kernel_size=kernel_size,dilation=dilation
                                      bias=self.bias))
   self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                      out_channels=skip_channels,
                                      kernel_size=kernel_size,dilation=dilation
                                      bias=self.bias))
   dilation *= 2

 def wavenet(self,input):
  x = self.start_conv(input)
  skip = 0
  for i in range(self.blocks*self.layers):
   
