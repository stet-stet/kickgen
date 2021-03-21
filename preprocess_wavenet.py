import os
import math
import torch
import numpy as np
import librosa as lr




# pretty much everything below are copied from
# https://github.com/vincentherrmann/pytorch-wavenet/blob/master/audio_data.py

def mu_law_compression(data,mu):
 mu_x = np.sign(data) * np.log(1 + mu * abs(data) / np.log(1 + mu)
 return mu_x

def mu_law_decompression(data,mu):
 s = np.sign(data) * (np.exp(np.abs(data)*np.log(mu+1)) -1) / mu
 return s

def quantize_data(data,classes):
 mu_x = mu_law_encoding(data,classes)
 bins = np.linspace(-1,1,classes)
 quantized = np.digitize(mu_x,bins)-1
 return quantized

# I am not going to use torch.utils.data.Dataset this time.
# (pedagogical reasons)

class WavenetDataset():
 def __init__(self,dataset_path):
  self.mono = True
  self.normalize = False
  self.sampling_rate = 44100
  self.classes = 256
  self.dtype=uint8
  self.processed_data = []
  for root,dirs,files in os.walk(dataset_path):
   for i, file in enumerate(files):
    if not file.endswith(".wav"): continue
    file_data, _ = lr.load(path=file, sr=self.sampling_rate, mono=self.mono)
    if self.normalize:
     file_data = lr.util.normalize(file_data)
    quantized_data = quantize_data(file_data,self.classes).astype(self.dtype)
    self.processed_data.append(quantized_data)

 def get_data():
  return self.processed_data

 def __getitem__(self,idx):
  pass
