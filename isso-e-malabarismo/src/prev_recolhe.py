import sys, os
os.environ.setdefault('FILM_H', '720')
sys.path.insert(0, '/home/claude/filme/src')
import numpy as np, cv2
import film
from core import *
tiles = []
specs = sys.argv[1:]
for sp in specs:
    kv = dict(x.split('=') for x in sp.split(','))
    t = float(kv.pop('t'))
    for k, v in kv.items():
        film.RECOLHE_CAM[k] = float(v)
    sh = film.SHOTS['recolhe'](); sh.setup()
    fr = sh.frame(int(round(t * FPS)))
    im = np.ascontiguousarray(cv2.resize(fr, (960, 540), interpolation=cv2.INTER_AREA))
    cv2.putText(im, sp, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(im)
cols = 3
while len(tiles) % cols:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)]
cv2.imwrite('/home/claude/filme/work/prev_recolhe.jpg', np.ascontiguousarray(np.vstack(rows)[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 88])
print('ok')
