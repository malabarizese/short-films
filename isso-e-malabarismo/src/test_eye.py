import sys, os, time
os.environ.setdefault('FILM_H', '720')
sys.path.insert(0, '/home/claude/filme/src')
import numpy as np, cv2
from core import *
from eye import EyeShot
from shots import preview
e = EyeShot()
t0 = time.time(); e.setup(); print('setup %.1fs' % (time.time() - t0))
times = [float(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [0.3, 2.0, 3.06, 4.5]
preview(e, times, '/home/claude/filme/work/prev_olho.jpg', cols=2)
