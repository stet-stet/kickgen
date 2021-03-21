shopt -s globstar nullglob
for f in dataset/**/*.wav
do
  ffmpeg -i "$f" -acodec pcm_s16le -ar 44100 -ac 1 "${f%.wav}.new.wav"
  mv -f "${f%.wav}.new.wav" "$f"
done
