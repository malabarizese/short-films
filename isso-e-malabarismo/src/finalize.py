# finalize.py — junta video (sem recodificar) + trilha (AAC) e confere o arquivo final
import sys, os, subprocess, json, re
sys.path.insert(0, '/home/claude/filme/src')
FINAL = '/home/claude/filme/final'
wav = sys.argv[1]
out = sys.argv[2]
vid = os.path.join(FINAL, 'video.mp4')
tmp = out[:-4] + '.tmp.mp4'
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', vid, '-i', wav, '-map', '0:v', '-map', '1:a', '-c:v', 'copy',
                '-c:a', 'aac', '-b:a', '320k', '-ar', '48000', '-ac', '2', '-movflags', '+faststart', '-shortest',
                '-metadata', 'title=Malabarize-se', '-metadata', 'artist=Malabarize-se', tmp], check=True)
os.replace(tmp, out)
pr = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                     'stream=codec_name,codec_type,width,height,r_frame_rate,nb_frames,duration,bit_rate,sample_rate,channels,pix_fmt,color_space:format=duration,size,bit_rate',
                     '-of', 'json', out], capture_output=True, text=True)
j = json.loads(pr.stdout)
for s in j['streams']:
    print({k: v for k, v in s.items()})
print(j['format'])
lo = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-i', out, '-map', '0:a', '-af', 'ebur128=peak=true', '-f', 'null', '-'],
                    capture_output=True, text=True).stderr
tail = lo[lo.rfind('Summary:'):]
print('audio no arquivo final: I = %s LUFS, LRA = %s LU, pico real = %s dBTP' % (
    re.search(r'I:\s*(-?[\d.]+) LUFS', tail).group(1), re.search(r'LRA:\s*(-?[\d.]+) LU', tail).group(1),
    re.search(r'Peak:\s*(-?[\d.]+) dBFS', tail).group(1)))
