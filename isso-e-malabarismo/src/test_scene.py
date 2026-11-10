import sys, time, os
os.environ.setdefault('FILM_H', '720')
import numpy as np, cv2
sys.path.insert(0, '/home/claude/filme/src')
from core import *
import sky as skylib
from atmos import Atmos, exposure_for
from terrain import Ridges
import fx

CDIV = 4
NSTEP = 56

def render(cam, hour, cov=0.42, fog=0.6, seed=0, cirrus_amt=0.22, cir_op=0.14):
    at = Atmos(cam, cloud_div=CDIV)
    sun = skylib.sun_dir(hour)
    L = at.light(sun)
    t0 = time.time()
    grad = at.sky_gradient(sun)
    cl, P = at.clouds(sun, L, dict(coverage=cov, sigma=55.0, tmax=95.0), seed=seed, N=NSTEP)
    ci = at.cirrus(sun, L, amount=cirrus_amt, opacity=cir_op)
    t1 = time.time()
    img, ca, info = at.compose(grad, L, cl, ci, stars=float(smooth(-5, -11, L['el'])), milky=1.0)
    hc = at.horizon_colors(grad)
    t2 = time.time()
    return img, L, hc, (t1 - t0, t2 - t1), ca, info

if __name__ == '__main__':
    global_vars = globals()
    which = sys.argv[1] if len(sys.argv) > 1 else 'master'
    hours = [float(h) for h in sys.argv[2].split(',')] if len(sys.argv) > 2 else [5.9, 6.12, 7.0, 12.5, 17.6, 18.05]
    if which == 'master':
        cam = Camera((-2.5, 0.05, -17.0), yaw=np.radians(6), pitch=np.radians(12.6), vfov=36.0)
    else:
        cam = Camera((-2.5, 0.05, -17.0), yaw=np.radians(3), pitch=np.radians(-0.6), vfov=13.0)
        CDIV = 2; NSTEP = 90
    rd = Ridges(cam)
    tiles = []
    for h in hours:
        img, L, hc, tm, ca, info = render(cam, h)
        t0 = time.time()
        fogamt = float(lerp(0.95, 0.10, smooth(6.3, 10.5, h))) if h < 12 else float(lerp(0.12, 0.45, smooth(16.0, 19.0, h)))
        rd.render(img, L, hc, fog=fogamt, vis_km=float(lerp(48, 70, smooth(6, 11, h))))
        t1 = time.time()
        if 'sun_px' in info:
            occ = ca[..., 0].copy()
            for Lr in rd.layers:
                if not Lr['skip']:
                    occ[Lr['r0']:Lr['r1']] = np.maximum(occ[Lr['r0']:Lr['r1']], Lr['alpha'])
            vis = fx.sun_visibility(occ, info['sun_px'], info['sun_px'][2])
            fx.god_rays(img, info['sun_px'], occ, strength=0.55, thresh=0.6 / exposure_for(L['el']) * 0.2)
            fx.sun_glow(img, info['sun_px'], L['sunCol'] * 1.6, vis=0.25 + 0.75 * vis)
            fx.lens_flare(img, info['sun_px'], L['sunCol'], vis=vis)
        img = bloom(img, 0.10)
        vignette(img, 0.25)
        ex = exposure_for(L['el'])
        disp = tonemap(img, ex)
        grain(disp, 0.018)
        t2 = time.time()
        print('h=%.2f el=%.1f sky %.2f comp %.2f ridges %.2f post %.2f expo %.2f' % (h, L['el'], tm[0], tm[1], t1 - t0, t2 - t1, ex))
        u = to_u8(disp)
        tiles.append(cv2.resize(u, (960, 540), interpolation=cv2.INTER_AREA))
    n = len(tiles)
    cols = 3 if n > 4 else 2
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, n, cols)]
    if len(rows) > 1 and rows[-1].shape[1] < rows[0].shape[1]:
        rows[-1] = np.hstack([rows[-1], np.zeros((540, rows[0].shape[1] - rows[-1].shape[1], 3), np.uint8)])
    sheet = np.vstack(rows)
    cv2.imwrite('/home/claude/filme/work/test_%s.jpg' % which, np.ascontiguousarray(sheet[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 88])
