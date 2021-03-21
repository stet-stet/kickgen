import os

for root,dirs,files in os.walk("dataset"):
 for file in files:
  if not file.endswith(".wav"):
   os.remove(root+'/'+file)
   print("removed "+root+'/'+file)
