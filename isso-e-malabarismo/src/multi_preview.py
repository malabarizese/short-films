import sys, os
os.environ.setdefault('FILM_H', '720')
sys.path.insert(0, '/home/claude/filme/src')
import numpy as np, cv2
import timeline
from core import *
tiles = []
cols = int(sys.argv[1])
for spec in sys.argv[2:]:
    name, t = spec.split('@'); t = float(t)
    sh = timeline.make(name); sh.setup()
    fr = sh.frame(int(round(t * FPS)))
    im = cv2.resize(fr, (960, 540), interpolation=cv2.INTER_AREA)
    im = np.ascontiguousarray(im)
    cv2.putText(im, '%s t=%.2f' % (name, t), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(im)
while len(tiles) % cols:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)]
cv2.imwrite('/home/claude/filme/work/multi.jpg', np.ascontiguousarray(np.vstack(rows)[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 88])
print('ok')
