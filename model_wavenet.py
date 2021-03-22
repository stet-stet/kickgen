import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from collections import namedtuple

# wavenet_info_constructor = namedtuple("wavenet_info","layers blocks 


# Impl is, for the time being, based on user vincentherrmann's implementation:
# but I will not limit output length, since sample length is pretty small anyways.
# I will refactor this after I fully comprehend everything
# but constants are hardcoded for now, for ease of work in sajibang environment.
# link : https://github.com/vincentherrmann/pytorch-wavenet/blob/master/wavenet_model.py
class WaveNetModel(nn.Module):
 def __init__(self):
  super(WaveNetModel,self).__init__()
  self.layers = 10
  self.blocks = 4
  self.dilation_channels = 32
  self.residual_channels = 32
  self.skip_channels = 256
  self.end_channels = 256
  self.classes = 256
  self.kernel_size = 2
  self.dtype = torch.FloatTensor
  self.bias = True
  # 1*1 conv to create channels & match dimensions.
  self.conv_one = nn.Conv1d(in_channels = self.classes,
                            out_channels = self.residual_channels,
                            kernel_size = 1,
                            bias = self.bias)

  # dilating convolutions
  self.filter_convs=[]
  self.gate_convs = []
  self.residual_convs = []
  self.skip_convs = []
  for b in range(self.blocks):
   for i in range(self.layers):
    dilation = 2**i
    self.filter_convs.append(nn.Conv1d(in_channels=self.residual_channels,
                                      out_channels=self.dilation_channels,
                                      kernel_size=self.kernel_size,padding=dilation,dilation=dilation,
                                      bias=self.bias))
    self.gate_convs.append(nn.Conv1d(in_channels=self.residual_channels,
                                      out_channels=self.dilation_channels,
                                      kernel_size=self.kernel_size,padding=dilation,dilation=dilation,
                                      bias=self.bias))
    self.residual_convs.append(nn.Conv1d(in_channels=self.dilation_channels,
                                      out_channels=self.residual_channels,
                                      kernel_size=self.kernel_size,padding=dilation,dilation=dilation,
                                      bias=self.bias))
    self.skip_convs.append(nn.Conv1d(in_channels=self.dilation_channels,
                                      out_channels=self.skip_channels,
                                      kernel_size=self.kernel_size,padding=dilation,dilation=dilation,
                                      bias=self.bias))
  self.end_conv_1 = nn.Conv1d(in_channels = self.skip_channels,
                            out_channels = self.end_channels,
                            kernel_size = 1,
                            bias = True)
  self.end_conv_2 = nn.Conv1d(in_channels = self.end_channels,
                            out_channels = self.classes,
                            kernel_size = 1,
                            bias = True)

 def wavenet(self,input):
  x = self.conv_one(input)
  input_dims = x.shape[2]
  skip = 0
  for i in range(self.blocks * self.layers):
   block, layer = i//self.layers,i%self.layers
   dilation = 2**layer
   # fun begins
   residual = x
   # input to gated activation
   filter = self.filter_convs[i](residual)[:,:,:input_dims]
   filter = F.tanh(filter)
   gate = self.gate_convs[i](residual)[:,:,:input_dims]
   gate = F.sigmoid(gate)
   x = filter * gate
   # skip
   s = x
   s = self.skip_convs[i](s)[:,:,:input_dims]
   try:
    skip = skip[:,:,-s.size(2):]
   except: # except...which?
    skip = 0
   skip = s + skip
   x = self.residual_convs[i](x)[:,:,:input_dims]
   x = x + residual
  x = F.relu(skip)
  x = F.relu(self.end_conv_1(x))
  x = self.end_conv_2(x)
  return x

 def forward(self,input):
  x = self.wavenet(input)
  [n,c,l] = x.size()
  x = x[:,:,:]
  x = x.transpose(1,2).contiguous()
  x = x.view(n*min(l,x.shape[1]), c)
  return x


