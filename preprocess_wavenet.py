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
    if file_data.shape[0] > 100000: continue
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
   self.processed_data[n] = np.append(self.processed_data[n],[255 for _ in range(500)]) # 255 means end of sample
   self.item_length[n] = self.processed_data[n].shape[0] # excluding the "startgen" tag(which is 0)

 def get_data(self):
  return self.processed_data

 def __getitem__(self,idx): # total data is smaller than 1GB - implementation is allowed to be naive.
  now_data = np.append([127],self.processed_data[idx])[:-1] # 0 means "start generation"
  now_data = torch.from_numpy(now_data).type(torch.LongTensor)
  one_hot = torch.FloatTensor(self.classes, self.item_length[idx]).zero_()
  one_hot.scatter_(0,now_data.unsqueeze(0),1.)
  target = torch.from_numpy(self.processed_data[idx]).type(torch.LongTensor).unsqueeze(0)
  identity = torch.FloatTensor(1,len(self.processed_data),1).zero_()
  identity[0,idx,0] = 1.
  return one_hot,target,identity

 def __len__(self):
  return len(self.processed_data)
