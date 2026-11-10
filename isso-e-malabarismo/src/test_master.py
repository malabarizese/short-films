import sys, time, os
os.environ.setdefault('FILM_H', '720')
import numpy as np, cv2
sys.path.insert(0, '/home/claude/filme/src')
from core import *
from scene import *
from tree import Petals
import juggle

hours = [float(h) for h in sys.argv[1].split(',')] if len(sys.argv) > 1 else [6.25, 12.5, 17.5, 18.1]
age = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
t0 = time.time()
cam = Camera((-2.5, float(mound(-2.5, -17.0)) + 1.0, -17.0), yaw=np.radians(6), pitch=np.radians(12.6), vfov=36.0)
W_ = World(cam)
print('world build %.1fs' % (time.time() - t0), 'strips', len(W_.meadow.strips))
ch = juggle.Choreo(5, 2); ch.hold(-1)
ch.run_async(0.0, [5], 0.215, r=0.42, n_beats=60, jit=0.008); ch.finalize()
origin = np.array([0.0, float(mound(0, 0)), 0.0])
pet = Petals(W_.tree, age, mound)
tiles = []
for h in hours:
    t = 3.37
    sun = skylib.sun_dir(h)
    t1 = time.time()
    L = W_.lighting(sun)
    fogamt = float(lerp(0.9, 0.12, smooth(6.3, 10.5, h))) if h < 12 else float(lerp(0.12, 0.5, smooth(16.5, 19.0, h)))
    img, occ, info = W_.background(sun, L, dict(coverage=0.40, sigma=55.0, tmax=95.0), dict(amount=0.20, opacity=0.12), fog=fogamt)
    t2 = time.time()
    pose = juggler_pose(ch, t, origin, n_balls=5)
    J = pose.solve()
    balls = [ball_world_fn(ch, b, origin) for b in range(5)]
    bloomv = 1.0 if age > 0.5 else 0.0
    W_.foreground(img, cam, L, t, occ, figure=dict(pose=pose, J=J), balls=balls,
                  tree=dict(age=age, leaves=0.15 if bloomv else 1.0, bloom=bloomv), petals=(pet, 1.0, {}) if bloomv else None)
    t3 = time.time()
    disp = W_.finish(img, L, info, occ)
    t4 = time.time()
    print('h=%.2f el=%.1f bg %.2f fg %.2f post %.2f' % (h, L['el'], t2 - t1, t3 - t2, t4 - t3))
    tiles.append(cv2.resize(to_u8(disp), (960, 540), interpolation=cv2.INTER_AREA))
n = len(tiles)
rows = [np.hstack(tiles[i:i + 2]) for i in range(0, n, 2)]
if len(rows) > 1 and rows[-1].shape[1] < rows[0].shape[1]:
    rows[-1] = np.hstack([rows[-1], np.zeros((540, 960, 3), np.uint8)])
cv2.imwrite('/home/claude/filme/work/test_master.jpg', np.ascontiguousarray(np.vstack(rows)[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 90])
