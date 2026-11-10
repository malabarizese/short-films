# thumb.py — still-base e miniatura (16:9) a partir do proprio motor do filme
import sys, os
sys.path.insert(0, '/home/claude/filme/src')
import numpy as np, cv2
import film
from core import *
from shots import orbit_cam


def thumb_shot(src='torre', phi=-16.0, dist=4.2, height=0.45, vfov=62.0, target_h=3.3, yaw_off=0.0, radius=0.075, sil=0.9,
               moon=None):
    sh = film.SHOTS[src]()
    sh.cam0 = orbit_cam(phi, dist, height=height, target_h=target_h, vfov=vfov, off=(yaw_off, 0.0))
    sh.meadow_kw = dict(d_far=dist + 1.3, focus=dist, d_near=0.7)
    bk = dict(sh.ball_kw or {}); bk['radius'] = radius
    sh.ball_kw = bk
    sh.fig_kw = dict(silhouette=sil)
    sh.shake = 0.0
    sh.grain_amt = 0.006
    if moon is not None:
        sh.moon = film.moon_fn(0.25, az=moon[0], el=moon[1])
    return sh


if __name__ == '__main__':
    tiles = []
    for sp in sys.argv[2:]:
        kv = dict(x.split('=') for x in sp.split(','))
        t = float(kv.pop('t')); src = kv.pop('src', 'torre')
        kw = {k: float(v) for k, v in kv.items()}
        sh = thumb_shot(src, **kw); sh.setup()
        fr = sh.frame(int(round(t * FPS)))
        im = np.ascontiguousarray(cv2.resize(fr, (800, 450), interpolation=cv2.INTER_AREA))
        cv2.putText(im, sp[:70], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im)
    cols = int(sys.argv[1])
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)]
    cv2.imwrite('/home/claude/filme/work/prev_thumb.jpg', np.ascontiguousarray(np.vstack(rows)[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print('ok')
