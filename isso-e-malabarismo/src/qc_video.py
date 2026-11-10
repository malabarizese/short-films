# qc_video.py — luminancia e diferenca entre quadros do filme inteiro; confere cortes e emendas de trechos
import sys, os, subprocess, json
os.environ.setdefault('FILM_H', '1440')
sys.path.insert(0, '/home/claude/filme/src')
import numpy as np, cv2
import timeline, render_all
from core import FPS
path = sys.argv[1]
W, H = 320, 180
p = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-i', path, '-vf', 'scale=%d:%d:flags=area,format=gray' % (W, H), '-f', 'rawvideo', '-'],
                     stdout=subprocess.PIPE)
raw = p.stdout.read(); p.wait()
fr = np.frombuffer(raw, np.uint8).reshape(-1, H, W)
print('quadros lidos:', len(fr))
np.save('/home/claude/filme/work/qc_gray.npy', fr)
luma = fr.reshape(len(fr), -1).mean(1)
blur = np.stack([cv2.GaussianBlur(f, (0, 0), 2.0) for f in fr]).astype(np.float32)
diff = np.abs(blur[1:] - blur[:-1]).reshape(len(fr) - 1, -1).mean(1)
st = timeline.starts(); dur = timeline.durations()
cuts = {int(round(st[n] * FPS)): n for n in timeline.ORDER}
seams = {}
for n in timeline.ORDER:
    cs = render_all.chunks_of(int(round(dur[n] * FPS)))
    for (a, b) in cs[1:]:
        seams[int(round(st[n] * FPS)) + a] = n
print('--- cortes (quadro, plano que comeca, luma antes -> depois, diferenca)')
for f in sorted(cuts):
    if f > 0:
        print('  %5d %-11s %6.1f -> %6.1f   dif %.1f' % (f, cuts[f], luma[f - 1], luma[f], diff[f - 1]))
print('--- emendas de trechos dentro do mesmo plano: diferenca na emenda x mediana dos vizinhos')
worst = []
for f in sorted(seams):
    nb = np.median(np.r_[diff[f - 6:f - 1], diff[f:f + 5]])
    worst.append((diff[f - 1] / max(nb, 0.05), f, seams[f], diff[f - 1], nb))
for r, f, n, d, nb in sorted(worst, reverse=True)[:8]:
    print('  quadro %5d (%s): %.2f x a mediana (%.2f contra %.2f)' % (f, n, r, d, nb))
print('--- saltos fora de cortes (diferenca > 4x a mediana local)')
cnt = 0
for i in range(8, len(diff) - 8):
    f = i + 1
    if f in cuts:
        continue
    loc = np.median(np.r_[diff[i - 8:i], diff[i + 1:i + 9]])
    if diff[i] > 4 * max(loc, 0.15) and diff[i] > 1.0:
        cnt += 1
        if cnt <= 30:
            n = [k for k in timeline.ORDER if st[k] <= f / FPS][-1]
            print('  quadro %5d  t=%.2f s (%s, %.2f s no plano): dif %.2f contra %.2f' % (f, f / FPS, n, f / FPS - st[n], diff[i], loc))
print('total de saltos:', cnt)
print('luma por plano (media, min, max):')
for n in timeline.ORDER:
    a = int(round(st[n] * FPS)); b = a + int(round(dur[n] * FPS))
    print('  %-11s %6.1f %6.1f %6.1f' % (n, luma[a:b].mean(), luma[a:b].min(), luma[a:b].max()))
np.save('/home/claude/filme/work/qc_luma.npy', luma); np.save('/home/claude/filme/work/qc_diff.npy', diff)
