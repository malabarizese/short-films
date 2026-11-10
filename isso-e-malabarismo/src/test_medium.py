import sys, time, os
os.environ.setdefault('FILM_H', '720')
import numpy as np, cv2
sys.path.insert(0, '/home/claude/filme/src')
from core import *
from scene import *
import juggle

hours = [float(h) for h in sys.argv[1].split(',')] if len(sys.argv) > 1 else [6.3, 17.4]
nb = int(sys.argv[2]) if len(sys.argv) > 2 else 3
dist = float(sys.argv[3]) if len(sys.argv) > 3 else 6.5
cam = Camera((-1.3, float(mound(-1.3, -dist)) + 1.05, -dist), yaw=np.radians(9), pitch=np.radians(9.0), vfov=30.0)
W_ = World(cam, meadow_kw=dict(d_far=dist + 0.9, focus=dist, d_near=0.9))
tau, r = {3: (0.27, 0.62), 5: (0.215, 0.42), 7: (0.20, 0.30)}[nb]
ch = juggle.Choreo(nb, 2); ch.hold(-1)
ch.run_async(0.0, [nb], tau, r=r, n_beats=80, jit=0.008); ch.finalize()
origin = np.array([0.0, float(mound(0, 0)), 0.0])
tiles = []
for h in hours:
    for t in (3.37, 3.52):
        sun = skylib.sun_dir(h)
        L = W_.lighting(sun)
        img, occ, info = W_.background(sun, L, dict(coverage=0.35, sigma=55.0, tmax=95.0), dict(amount=0.2, opacity=0.12), fog=0.7 if h < 9 else 0.3)
        pose = juggler_pose(ch, t, origin, n_balls=nb)
        J = pose.solve()
        balls = [ball_world_fn(ch, b, origin) for b in range(nb)]
        W_.foreground(img, cam, L, t, occ, figure=dict(pose=pose, J=J), balls=balls, tree=dict(age=0.25, leaves=1.0, bloom=0.0))
        disp = W_.finish(img, L, info, occ)
        tiles.append(cv2.resize(to_u8(disp), (960, 540), interpolation=cv2.INTER_AREA))
rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
cv2.imwrite('/home/claude/filme/work/test_medium.jpg', np.ascontiguousarray(np.vstack(rows)[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 90])
