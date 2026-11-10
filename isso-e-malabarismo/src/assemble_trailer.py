# assemble_trailer.py — junta os trechos do teaser vertical e a trilha
import sys, os, subprocess, json, re
os.environ['FILM_H'] = '1920'; os.environ['FILM_ASPECT'] = '9:16'; os.environ['FILM_TIMELINE'] = 'trailer'
sys.path.insert(0, '/home/claude/filme/src')
import trailer, render_all
from core import FPS
OUT = '/home/claude/filme/out_v'
FINAL = '/home/claude/filme/final'
wav, out = sys.argv[1], sys.argv[2]
lst = os.path.join(FINAL, 'lista_teaser.txt')
tot = 0
with open(lst, 'w') as f:
    for n in trailer.ORDER:
        sg = trailer.make(n)
        for k, (a, b) in enumerate(render_all.chunks_of(sg.n)):
            p = render_all.chunk_path(OUT, n, k)
            assert os.path.exists(p), 'falta ' + p
            f.write("file '%s'\n" % p); tot += b - a
print('quadros esperados:', tot, '= %.2f s' % (tot / FPS))
vid = os.path.join(FINAL, 'teaser_video.mp4')
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', vid], check=True)
tmp = out[:-4] + '.tmp.mp4'
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', vid, '-i', wav, '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac',
                '-b:a', '256k', '-ar', '48000', '-ac', '2', '-movflags', '+faststart', '-shortest', tmp], check=True)
os.replace(tmp, out)
pr = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-show_entries',
                     'stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration:format=duration,size,bit_rate', '-of', 'json', out],
                    capture_output=True, text=True)
j = json.loads(pr.stdout)
for s in j['streams']:
    print(s)
print(j['format'])
lo = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-i', out, '-map', '0:a', '-af', 'ebur128=peak=true', '-f', 'null', '-'],
                    capture_output=True, text=True).stderr
tail = lo[lo.rfind('Summary:'):]
print('audio: I = %s LUFS, pico real = %s dBTP' % (re.search(r'I:\s*(-?[\d.]+) LUFS', tail).group(1), re.search(r'Peak:\s*(-?[\d.]+) dBFS', tail).group(1)))
