import sys, os
os.environ.setdefault('FILM_H', '960'); os.environ.setdefault('FILM_ASPECT', '9:16')
sys.path.insert(0, '/home/claude/filme/src')
import numpy as np, cv2
import trailer
from core import *
specs = sys.argv[2:]
cols = int(sys.argv[1])
tiles = []
cache = {}
tw, th = 300, 533
for sp in specs:
    name, t = sp.split('@'); t = float(t)
    if name not in cache:
        cache.clear()
        cache[name] = trailer.make(name).setup()
    fr = cache[name].frame(int(round(t * FPS)))
    im = np.ascontiguousarray(cv2.resize(fr, (tw, th), interpolation=cv2.INTER_AREA))
    cv2.putText(im, '%s %.1f' % (name[2:], t), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(im)
while len(tiles) % cols:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)]
cv2.imwrite('/home/claude/filme/work/prev_trailer.jpg', np.ascontiguousarray(np.vstack(rows)[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 88])
print('ok', len(tiles))
