import sys, os
os.environ.setdefault('FILM_H', '720')
import numpy as np, cv2
sys.path.insert(0, '/home/claude/filme/src')
from core import *
from figure import *
from scene import juggler_pose, ball_world_fn, BALL_R
import juggle, sky as skylib
from atmos import Atmos

def light(hour):
    at = Atmos(Camera((0, 1, -5)))
    return at.light(skylib.sun_dir(hour))

def tile(pose_fn, cam_yaw, hour, t, nb, sil=0.0, W=420, H=560, bg=None):
    cam = Camera((0, 1.0, 0), yaw=0, pitch=0, vfov=26, W=W, H=H)
    # camera em orbita ao redor do malabarista
    d = 4.6
    a = np.radians(cam_yaw)
    cam.pos = np.array([-np.sin(a) * d, 1.0, -np.cos(a) * d]); cam.yaw = a; cam.update()
    L = light(hour)
    img = np.zeros((H, W, 3), np.float32) + (bg if bg is not None else L['hor'] * 0.9)
    ch, pose = pose_fn(t)
    if ch is not None and pose_fn is ppick:
        t = ch.t_end - 1.25 + t
    J = pose.solve()
    zf = 4.6
    for b in range(ch.n if ch else 0):
        p = ch.ball_local(b, t)
        pw = pose.to_world(p)
        zb = (pw - cam.pos) @ cam.fwd
        if zb > zf - 0.05:
            draw_ball(img, cam, [pw], BALL_R, L, rimk=1.2)
    draw_figure(img, cam, J, L, silhouette=sil, bag=pose.bag)
    for b in range(ch.n if ch else 0):
        p = ch.ball_local(b, t)
        pw = pose.to_world(p)
        zb = (pw - cam.pos) @ cam.fwd
        if zb <= zf - 0.05:
            draw_ball(img, cam, [pw], BALL_R, L, rimk=1.2)
    from atmos import exposure_for
    return to_u8(tonemap(img, exposure_for(L['el'])))

def mk(nb, tau, r):
    ch = juggle.Choreo(nb, 2); ch.hold(-1); ch.run_async(0.0, [nb], tau, r=r, n_beats=90, jit=0.008); ch.finalize()
    return lambda t: (ch, juggler_pose(ch, t, np.zeros(3), n_balls=nb))

p3 = mk(3, 0.40, 0.50); p5 = mk(5, 0.215, 0.42); p7 = mk(7, 0.20, 0.30)
from scene import walk_pose
def pwalk(t):
    return None, walk_pose(t, np.zeros(3), 0.0, slope=0.15)
def ppick(t):
    ch = juggle.Choreo(3, 5); ch.hold(-1)
    ch.run_async(0.0, [3], 0.40, r=0.5, n_beats=40, jit=0.01, bad={9: (0.5, 0, 0.35)})
    te = ch.pickup(ch.t_end + 0.2); ch.finalize()
    tt = ch.t_end - 1.25 + t
    return ch, juggler_pose(ch, tt, np.zeros(3), n_balls=3)

rows = []
for hour, sil in ((6.3, 0.6), (15.5, 0.0)):
    row = []
    for (fn, nb, t, yaw) in ((p3, 3, 3.3, 0), (p3, 3, 3.5, 35), (p5, 5, 3.3, 35), (p7, 7, 3.4, 20), (pwalk, 0, 0.55, 90), (pwalk, 0, 1.0, 60), (ppick, 3, 0.45, 40)):
        row.append(tile(fn, yaw, hour, t, nb, sil))
    rows.append(np.hstack(row))
cv2.imwrite('/home/claude/filme/work/test_fig.jpg', np.ascontiguousarray(np.vstack(rows)[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 90])
