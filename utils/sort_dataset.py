import os
import sys
from hashlib import md5

# generates a .sh script that can be executed to move datasets.
# new names for the wav files will be a md5 hash.

# for unlucky hash collisions we will append _1, _2... to each file in collision.

def get_file_hash(full_filename):
 with open(full_filename,'rb') as file:
  r = file.read()
 hshfunc = md5()
 hshfunc.update(r[::-1])
 return hshfunc.hexdigest()

def make_data_dict(folder_name):
 ret = {}
 for root,dirs,files in os.walk(folder_name):
  for file in files:
   fullname = root+'/'+file
   hsh = get_file_hash(fullname)
   if hsh in ret:
    ret[hsh].append(fullname)
   else:
    ret[hsh] = [fullname]
 return ret

def make_relocating_script(input_folder,output_folder,script_name):
 print("hashing...")
 d = make_data_dict(input_folder)
 buffer = []
 for key in d:
  for n,entry in enumerate(d[key]):
   oldname = entry
   name = oldname.split('/')[-1]
   newname = output_folder + '/' + key + "_" + str(n) + ".wav"
   buffer.append(f'cp "{oldname}" "{output_folder}"')
   buffer.append(f'mv "{output_folder}/{name}" "{newname}"')
 with open(script_name,'w') as file:
  for line in buffer:
   file.write(line+'\n')

if __name__=="__main__":
 if len(sys.argv) <=3:
  print(f"usage: python {sys.argv[0]} (dataset) (output folder) (relocating script name)")
  exit(0)
 os.makedirs(sys.argv[2],exist_ok=False)
 make_relocating_script(sys.argv[1],sys.argv[2],sys.argv[3])
