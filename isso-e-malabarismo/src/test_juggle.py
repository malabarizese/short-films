import sys, numpy as np, cv2
sys.path.insert(0, '/home/claude/filme/src')
from juggle import *

def trace(c, t0, t1, name, scale=220, H=900, W=700, y0=0.6):
    img = np.zeros((H, W, 3), np.uint8)
    cols = [(255, 255, 255), (255, 180, 80), (120, 200, 255), (160, 255, 160), (255, 120, 200), (255, 255, 120), (160, 160, 255)]
    ts = np.arange(t0, t1, 1 / 240.0)
    for b in range(c.n):
        pts = np.array([c.ball_local(b, t) for t in ts])
        px = (W / 2 + pts[:, 0] * scale).astype(int)
        py = (H - 30 - (pts[:, 1] - y0) * scale).astype(int)
        for x, y in zip(px, py):
            if 0 <= x < W and 0 <= y < H:
                img[y, x] = cols[b % 7]
    for h in (L, R):
        pts = np.array([c.hand_local(h, t) for t in ts])
        px = (W / 2 + pts[:, 0] * scale).astype(int)
        py = (H - 30 - (pts[:, 1] - y0) * scale).astype(int)
        for x, y in zip(px, py):
            if 0 <= x < W and 0 <= y < H:
                img[y, x] = (0, 0, 255) if h == L else (0, 255, 0)
    cv2.putText(img, name, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img

tiles = []
# 3 bolas
c = Choreo(3, 1); c.hold(-1); te = c.run_async(0.0, [3], 0.27, r=0.62, n_beats=30, jit=0.01); c.finalize()
print('3: fails', getattr(c, 'n_fail', 0), 'throws', c.n_throws, 'apex', max(s[2].apex() for tr in c.tracks for s in tr if s[1] == 'fly') - 1.0)
tiles.append(trace(c, 1.0, 5.0, '3 cascata', scale=420, y0=0.8))
# 5 bolas -> (6x,4)*
c = Choreo(5, 2); c.hold(-1)
tau = 0.215
te = c.run_async(0.0, [5], tau, r=0.42, n_beats=31, jit=0.008)      # ultima batida = 30 (mao R em pares)
print('5 cascade end', te, 'held', {h: len(q) for h, q in c.held.items()})
# sincrono comeca na proxima batida par
k0 = 32
pairs = [((6, True), (4, False)), ((4, False), (6, True))]
te2 = c.run_sync(k0 * tau, pairs, tau, r=0.42, n_steps=16, jit=0.008)
c.finalize()
print('5: fails', getattr(c, 'n_fail', 0), 'throws', c.n_throws)
tiles.append(trace(c, 1.5, 6.0, '5 cascata', scale=300, y0=0.8))
tiles.append(trace(c, 8.2, 13.0, '(6x,4)*', scale=300, y0=0.8))
# 7 bolas -> db97531
c = Choreo(7, 3); c.hold(-1)
tau = 0.20
n7 = 28
te = c.run_async(0.0, [7], tau, r=0.30, n_beats=n7, jit=0.006)
tow = Geom(0.33, 0.21, 0.99, 1.09, 0.32)
te = c.run_async(n7 * tau, [13, 11, 9, 7, 5, 3, 1], tau, r=0.30, n_beats=7, geom=tow, jit=0.004)
c.finalize()
print('7: fails', getattr(c, 'n_fail', 0), 'throws', c.n_throws, 'held', {h: len(q) for h, q in c.held.items()})
fl = [s[2] for tr in c.tracks for s in tr if s[1] == 'fly' and s[2].t0 >= n7 * tau - 1e-6]
for f in sorted(fl, key=lambda f: f.t0):
    print('  v=%2d t0=%.2f t1=%.2f T=%.2f apex=%.2f m  hand %d->%d' % (f.value, f.t0, f.t1, f.T, f.apex() - 1.05, f.h_from, f.h_to))
tiles.append(trace(c, 2.0, 5.0, '7 cascata', scale=200, y0=0.8))
tiles.append(trace(c, n7 * tau - 0.2, n7 * tau + 3.2, 'db97531', scale=100, y0=0.6))
# instante da torre
tt = n7 * tau + 6.6 * tau
print('torre em t=%.2f:' % tt, [(round(c.ball_local(b, tt)[0], 2), round(c.ball_local(b, tt)[1], 2)) for b in range(7)])
cv2.imwrite('/home/claude/filme/work/test_juggle.jpg', np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
