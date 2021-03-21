import os
import sys

def count(folder):
 ret = 0
 for root,dirs,files in os.walk(folder):
  for file in files:
   ret += 1
 return ret

if __name__=="__main__":
 if len(sys.argv) == 1:
  print("call me with appropriate arguments!")
 for i in range(1,len(sys.argv)):
  print(f"folder name {sys.argv[i]} \n    {count(sys.argv[i])} entries")
