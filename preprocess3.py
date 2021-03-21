from pydub import AudioSegment
import os

def match_target_amplitude(sound,target_dBFS):
 change_in_dBFS = target_dBFS - sound.dBFS
 return sound.apply_gain(change_in_dBFS)


wrongly_done = []
for root,dirs,files in os.walk("./dataset"):
 for file in files:
  try:
   p = root+'/'+file
   print(p)
   sound = AudioSegment.from_wav(p)
   normalized = match_target_amplitude(sound, -20.0)
   normalized.export(p,format="wav")
  except:
   wrongly_done.append(root+'/'+file)

with open("wrongly_preprocessed","w") as file:
 for entry in wrongly_done:
  file.write(entry+'\n')
