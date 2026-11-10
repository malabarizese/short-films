# assemble.py — junta os trechos renderizados (sem recodificar) e, se houver, a trilha
import sys, os, subprocess, json
os.environ.setdefault('FILM_H', '1440')
sys.path.insert(0, '/home/claude/filme/src')
import timeline, render_all
from core import FPS
OUT = '/home/claude/filme/out'
FINAL = '/home/claude/filme/final'
os.makedirs(FINAL, exist_ok=True)
lst = os.path.join(FINAL, 'lista.txt')
tot = 0
with open(lst, 'w') as f:
    for n in timeline.ORDER:
        sh = timeline.make(n)
        cs = render_all.chunks_of(sh.n)
        for k, (a, b) in enumerate(cs):
            p = render_all.chunk_path(OUT, n, k)
            assert os.path.exists(p), 'falta ' + p
            f.write("file '%s'\n" % p)
            tot += b - a
print('quadros esperados:', tot, '= %.3f s' % (tot / FPS))
vid = os.path.join(FINAL, 'video.tmp.mp4')
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', '-movflags', '+faststart', vid], check=True)
os.replace(vid, os.path.join(FINAL, 'video.mp4'))
pr = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries',
                     'stream=nb_read_frames,duration,r_frame_rate,avg_frame_rate,width,height,pix_fmt,color_space,color_range,bit_rate',
                     '-of', 'json', os.path.join(FINAL, 'video.mp4')], capture_output=True, text=True)
print(pr.stdout)
