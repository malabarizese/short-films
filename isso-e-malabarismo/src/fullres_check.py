import sys, os, time
os.environ['FILM_H'] = '1440'
sys.path.insert(0, '/home/claude/filme/src')
import numpy as np, cv2
import film
from core import *
name, t = sys.argv[1], float(sys.argv[2])
crop = [float(v) for v in sys.argv[3].split(',')] if len(sys.argv) > 3 else None
if name == 'olho':
    from eye import EyeShot
    sh = EyeShot()
elif name == 'bola':
    from closeups import HandBallShot
    sh = HandBallShot()
else:
    sh = film.SHOTS[name]()
t0 = time.time(); sh.setup(); t1 = time.time()
i = int(round(t * FPS))
fr = sh.frame(i); t2 = time.time()
fr2 = sh.frame(i + 1); t3 = time.time()
fr3 = sh.frame(i + 2); t4 = time.time()
print('%s setup %.1fs, 1o quadro %.2fs, seguintes %.2fs %.2fs' % (name, t1 - t0, t2 - t1, t3 - t2, t4 - t3))
H, W = fr.shape[:2]
if crop:
    x0, y0, w, h = int(crop[0] * W), int(crop[1] * H), int(crop[2] * W), int(crop[3] * H)
    fr = fr[y0:y0 + h, x0:x0 + w]
cv2.imwrite('/home/claude/filme/work/full_%s.jpg' % name, np.ascontiguousarray(fr[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 90])
