import sys, os
os.environ.setdefault('FILM_H', '720')
sys.path.insert(0, '/home/claude/filme/src')
from cards import *
from shots import preview
t = TextCard(); preview(t, [2.4, 4.6, 7.5, 10.0], '/home/claude/filme/work/prev_texto.jpg', cols=4, tile=(640, 360))
l = LogoCard(); preview(l, [0.6, 1.4, 1.58, 1.9, 2.3, 2.7, 3.35, 5.0], '/home/claude/filme/work/prev_logo.jpg', cols=4, tile=(640, 360))
