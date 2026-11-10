# extrai os eventos das coreografias no tempo do filme (para sincronizar som e musica)
import sys, os, json
os.environ['FILM_H'] = '720'
sys.path.insert(0, '/home/claude/filme/src')
import numpy as np
import film, timeline
from core import FPS
st = timeline.starts()
ev = {}


def inv(sim, dur, ts):
    """tempo de filme (dentro do plano) em que sim(t) = ts."""
    grid = np.arange(0, dur + 1e-9, 1 / 600.0)
    s = np.array([sim(t) for t in grid])
    out = []
    for v in ts:
        if v < s[0] or v > s[-1]:
            out.append(None); continue
        i = int(np.searchsorted(s, v))
        out.append(float(grid[min(i, len(grid) - 1)]))
    return out


for name in ('tres', 'cinco', 'sete_falha', 'sete_ok', 'torre', 'final', 'tl1', 'tl2', 'tl3'):
    sh = film.SHOTS[name]()
    ch = sh.choreo
    sim = sh.sim if sh.sim is not None else (lambda t: t)
    ts = [e[0] for e in ch.sfx]
    ft = inv(sim, sh.dur, ts)
    lst = []
    for (t, typ, inten), f in zip(ch.sfx, ft):
        if f is not None:
            lst.append((round(st[name] + f, 4), typ, round(float(inten), 3), round(f, 4)))
    ev[name] = lst
    print(name, len(lst), 'eventos; tipos:', {k: sum(1 for e in lst if e[1] == k) for k in set(e[1] for e in lst)})
ch, T_d = film.final_choreo()
info = dict(T_d=T_d, t_touch=ch.t_touch, t_last=ch.t_last, starts=st)
sh = film.SHOTS['torre']()
for key, v in (('T_d', T_d), ('t_touch', ch.t_touch), ('t_last', ch.t_last)):
    info['torre_' + key] = st['torre'] + inv(sh.sim, sh.dur, [v])[0]
ev['_info'] = info
json.dump(ev, open('/home/claude/filme/audio/events.json', 'w'))
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items() if k != 'starts'})
for name in ('tres', 'sete_falha', 'torre'):
    print(name, [(e[0], e[1], e[2]) for e in ev[name]][:60])
