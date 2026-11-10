import sys, os
os.environ.setdefault('FILM_H', '720')
sys.path.insert(0, '/home/claude/filme/src')
from closeups import *
from shots import preview
which = sys.argv[1]
if which == 'bola':
    s = HandBallShot(); preview(s, [1.0, 2.4, 3.3, 3.55, 3.9, 4.8], '/home/claude/filme/work/prev_bola.jpg', cols=3)
else:
    s = GroundBallShot(); preview(s, [0.8, 2.0, 2.45, 3.0], '/home/claude/filme/work/prev_bola_chao.jpg', cols=2)
