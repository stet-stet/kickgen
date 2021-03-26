import importlib
import random
import model_wavenet
import preprocess_wavenet
import train_wavenet
importlib.reload(model_wavenet)
importlib.reload(preprocess_wavenet)
importlib.reload(train_wavenet)
dat = preprocess_wavenet.WavenetDataset("./dataset_hashed")
data_no = len(dat)

md = model_wavenet.WaveNetModel(len(dat))

trainer = train_wavenet.WavenetTrainer(model=md,dataset=dat)
trainer.train(batch_size=32,epochs=10)
