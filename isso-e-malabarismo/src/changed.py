# quais quadros mudam com o brilho do sol suavizado (para refazer so os trechos afetados)
import sys, os
os.environ['FILM_H'] = '1440'
sys.path.insert(0, '/home/claude/filme/src')
import numpy as np
import timeline, render_all, film
from core import *
res = {}
for n in timeline.ORDER:
    if n not in film.SHOTS:
        continue
    sh = film.SHOTS[n]()
    c = sh.cam0; c.W, c.H = CFG.W, CFG.H; c.update()
    W, H = CFG.W, CFG.H
    ch = []
    flags = []
    for i in range(sh.n):
        t = i / FPS
        s = np.asarray(sh.sun(t), float)
        el = np.degrees(np.arcsin(np.clip(s[1], -1, 1)))
        sx, sy, sz = c.project_dir(s)
        if not (sz > 0 and el > -1.2):
            flags.append(None); continue
        d = max(-sx / W, sx / W - 1.0, -sy / H, sy / H - 1.0, 0.0)
        old = 1.0 if ((-0.2 * W < sx < 1.2 * W) and (-0.2 * H < sy < 1.2 * H)) else 0.0
        new = 1.0 - float(smooth(0.2, 0.9, d))
        flags.append(old)
        if abs(new - old) > 0.002:
            ch.append(i)
    cs = render_all.chunks_of(sh.n)
    ks = sorted(set(k for k, (a, b) in enumerate(cs) for i in ch if a <= i < b))
    cross = sum(1 for a, b in zip(flags[:-1], flags[1:]) if a is not None and b is not None and a != b)
    if ch:
        print('%-11s quadros alterados: %d (%d..%d)  cruzamentos: %d  trechos: %s' % (n, len(ch), ch[0], ch[-1], cross, ks))
        res[n] = ks
print(res)
