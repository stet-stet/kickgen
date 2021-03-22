import os
import math
import torch
import numpy as np
import librosa as lr

# pretty much everything below are copied from
# https://github.com/vincentherrmann/pytorch-wavenet/blob/master/audio_data.py

def mu_law_compression(data,mu):
 mu_x = np.sign(data) * np.log(1 + mu * abs(data)) / np.log(1 + mu)
 return mu_x

def mu_law_decompression(data,mu):
 s = np.sign(data) * (np.exp(np.abs(data)*np.log(mu+1)) -1) / mu
 return s

def quantize_data(data,classes):
 mu_x = mu_law_compression(data,classes)
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
  self.dtype=np.uint8
  self.item_length = []
  self.processed_data = []
  self.raw_filename = []
  for root,dirs,files in os.walk(dataset_path):
   for i, file in enumerate(files):
    if i%1000 ==0: print(str(i)+" out of "+str(len(files))+" files")
    if not file.endswith(".wav"): continue
    full_file = root+'/'+file
    file_data, _ = lr.load(path=full_file, sr=self.sampling_rate, mono=self.mono) #np.ndarray
    if self.normalize:
     file_data = lr.util.normalize(file_data)
    quantized_data = quantize_data(file_data,self.classes).astype(self.dtype)
    self.processed_data.append(quantized_data)
    self.item_length.append(quantized_data.shape[0])
    self.raw_filename.append(full_file)
  for n,data in enumerate(self.processed_data[:]):
   if n%1000==0: print(n)
   stophere=-1
   while data[stophere] == 127:
    stophere-=1
   if stophere < -1:
    self.processed_data[n] = data[:stophere+1]
    self.item_length[n] = self.processed_data[n].shape[0]

 def get_data(self):
  return self.processed_data

 def __getitem__(self,idx): # total data is smaller than 1GB - implementation is allowed to be naive.
  now_data = self.processed_data[idx]
  now_data = torch.from_numpy(now_data).type(torch.LongTensor)
  one_hot = torch.FloatTensor(self.classes, self.item_length[idx]).zero_()
  one_hot.scatter_(0,now_data[:self.item_length[idx]].unsqueeze(0),1.)
  target = now_data.unsqueeze(0)
  return one_hot,target

 def __len__(self):
  return len(self.processed_data)
