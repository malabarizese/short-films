import sys, os
os.environ.setdefault('FILM_H', '720')
sys.path.insert(0, '/home/claude/filme/src')
import film
from shots import preview
name = sys.argv[1]
times = [float(x) for x in sys.argv[2].split(',')]
cols = int(sys.argv[3]) if len(sys.argv) > 3 else 3
sh = film.SHOTS[name]()
preview(sh, times, '/home/claude/filme/work/prev_%s.jpg' % name, cols=cols)
print('ok', name, sh.n, 'quadros')
