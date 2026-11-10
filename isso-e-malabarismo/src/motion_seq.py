import sys, os
os.environ.setdefault('FILM_H', '720')
sys.path.insert(0, '/home/claude/filme/src')
import numpy as np, cv2
import timeline
name, i0, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
sh = timeline.make(name); sh.setup()
prev = None; out = []
for i in range(i0, i0 + n):
    fr = cv2.GaussianBlur(sh.frame(i).astype(np.float32), (0, 0), 2.0)   # tira o grao
    if prev is not None:
        out.append(np.abs(fr - prev).mean())
    prev = fr
print(name, ' '.join('%.2f' % v for v in out))
