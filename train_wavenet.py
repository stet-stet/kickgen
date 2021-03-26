import torch
import torch.optim as optim
import torch.nn.functional as F
import time
from datetime import datetime
from torch.autograd import Variable
from random import randint

def print_last_loss(opt):
 print("loss: ",opt.losses[-1])

class WavenetTrainer:
 def __init__(self,model,dataset):
  self.model = model
  self.dataset = dataset
  self.lr = 0.001
  self.weight_decay=0
  self.optimizer_type = optim.Adam
  self.optimizer = self.optimizer_type(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
  self.dtype = torch.FloatTensor
  self.ltype = torch.LongTensor
 # TODO: LOGGER??
 def get_random_training_data(self): #data,target
  return self.dataset[randint(0,len(self.dataset)*99//100-1)]

 def get_random_eval_data(self):
  return self.dataset[randint(len(self.dataset)*99//100,len(self.dataset)-1)]

 def train(self,batch_size,epochs):
  self.model.train() # into training mode
  step = 0
  for current_epoch in range(epochs):
   print(f"epoch {current_epoch}",flush=True)
   tic = time.time()
   for iter in range(len(self.dataset)//batch_size):
    self.optimizer.zero_grad()
    losses = []
    for batch in range(batch_size):
     x,target,identity = self.get_random_training_data()
     x = Variable(x.type(self.dtype).unsqueeze(0))
     target = Variable(target.view(-1).type(self.ltype))

     output = self.model(x,identity)
     loss = F.cross_entropy(output.squeeze(),target.squeeze())
     loss.backward()
     losses.append(loss.item())
     print(batch,end=" ",flush=True)
    self.optimizer.step()
    step+=1
    lll = sum(losses)
    if step <30 and step %3 == 0:
     toc = time.time()
     print(f"each training step took approx. {(toc-tic)/step} seconds.",flush=True)
    print(f"step: {step} \t loss: {lll}",flush=True)
   if iter % 3 == 0:
    time_string = time.strftime("%Y-%m-%d_%H-%M-%S",time.gmtime())
    print("saving...",flush=True)
    torch.save(self.model,"./snapshots/"+time_string)
    print("saved.",flush=True)
