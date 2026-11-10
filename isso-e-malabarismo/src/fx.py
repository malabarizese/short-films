# fx.py — brilho do sol, raios, flare de lente, particulas
import numpy as np
import cv2
from core import CFG, f32, smooth, add, blur


def sun_visibility(occ, px, r, soft=1.2):
    """fracao visivel do disco solar. occ = mapa de oclusao (H,W) 0..1 (1 = tapado).
    soft > 1.2 amostra uma regiao maior (transicao gradual quando nuvens cruzam o sol em time-lapse).
    Sol fora do quadro: usa a oclusao na borda mais proxima (continuo ao cruzar a borda)."""
    H, W = occ.shape[:2]
    R = int(max(2, r * soft))
    x = int(round(min(max(px[0], R), W - R - 1)))
    y = int(round(min(max(px[1], R), H - R - 1)))
    xa, xb, ya, yb = max(x - R, 0), min(x + R + 1, W), max(y - R, 0), min(y + R + 1, H)
    if xb <= xa or yb <= ya:
        return 0.6
    reg = occ[ya:yb, xa:xb]
    if soft > 2.0:
        yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
        w = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * (0.5 * R) ** 2))
        return float(1.0 - (reg * w).sum() / max(float(w.sum()), 1e-6))
    return float(1.0 - reg.mean())


def sun_glow(img, px, col, vis=1.0, size=1.0, strength=1.0):
    """halo aditivo em torno do sol (dispersao Mie + veu de lente)."""
    H, W = img.shape[:2]
    if vis <= 0.001:
        return
    div = 4
    h, w = H // div, W // div
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = (xx + 0.5) * div - px[0], (yy + 0.5) * div - px[1]
    r = np.sqrt(dx * dx + dy * dy) / (H * 0.05 * size)
    g = 0.9 / (1.0 + r * r) ** 1.5 + 0.10 / (1.0 + (r / 4.0) ** 2) + 0.025 / (1.0 + (r / 14.0) ** 2)
    g = (g * vis * strength)[..., None] * np.asarray(col, dtype=np.float32)
    img += cv2.resize(g.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)


def god_rays(img, px, occ, strength=0.5, decay=0.965, n=40, thresh=1.2, tint=(1.0, 0.85, 0.65)):
    """raios crepusculares em espaco de tela: desfoque radial das areas claras nao ocluidas."""
    H, W = img.shape[:2]
    div = 8
    w, h = W // div, H // div
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    lum = small.max(-1)
    src = np.maximum(lum - thresh, 0)
    src = np.minimum(src, thresh * 14)
    if occ is not None:
        o = cv2.resize(occ, (w, h), interpolation=cv2.INTER_AREA)
        src *= (1 - o)
    # peso radial: so perto do sol
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = px[0] / div, px[1] / div
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / h
    src *= np.exp(-rr * 2.2)
    acc = np.zeros_like(src)
    wgt = 1.0
    tot = 0.0
    for i in range(n):
        sc = 1.0 + 0.9 * (i / n)        # amplia a partir do sol
        M = np.float32([[sc, 0, cx * (1 - sc)], [0, sc, cy * (1 - sc)]])
        acc += cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_LINEAR) * wgt
        tot += wgt
        wgt *= decay
    acc /= tot
    acc = cv2.GaussianBlur(acc, (0, 0), 1.2)
    rays = cv2.resize(acc, (W, H), interpolation=cv2.INTER_CUBIC)
    np.maximum(rays, 0, out=rays)
    img += rays[..., None] * (np.asarray(tint, dtype=np.float32) * strength)


def lens_flare(img, px, col, vis=1.0, strength=1.0):
    """fantasmas de lente no eixo sol-centro + risco anamorfico discreto."""
    if vis <= 0.01:
        return
    H, W = img.shape[:2]
    c = np.array([W / 2, H / 2])
    s = np.array(px[:2], dtype=np.float64)
    v = c - s
    col = np.asarray(col, dtype=np.float32)
    ghosts = [(0.35, 0.018, (1.0, 0.55, 0.25), 0.050), (0.62, 0.030, (0.35, 0.8, 1.0), 0.030),
              (1.15, 0.055, (0.9, 0.5, 1.0), 0.022), (1.45, 0.020, (0.4, 1.0, 0.6), 0.040),
              (1.90, 0.090, (1.0, 0.75, 0.4), 0.014), (2.30, 0.035, (0.5, 0.6, 1.0), 0.028)]
    lum = float(col.max())
    for k, rad, tint, a in ghosts:
        p = s + v * k
        R = rad * H
        if not (-R < p[0] < W + R and -R < p[1] < H + R):
            continue
        n = int(2 * R + 4)
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        d = np.sqrt((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / R
        m = np.clip(1 - d, 0, 1) ** 0.6 * (0.55 + 0.45 * smooth(0.6, 1.0, d))
        m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(1.0, R * 0.05))
        add(img, m[..., None] * (np.asarray(tint, dtype=np.float32) * (a * lum * vis * strength)), int(p[0] - n / 2), int(p[1] - n / 2))
    # risco horizontal
    y0 = int(round(s[1]))
    if 0 <= y0 < H:
        xs = np.arange(W, dtype=np.float32)
        fall = 1.0 / (1.0 + ((xs - s[0]) / (W * 0.16)) ** 2)
        hh = max(2, int(3 * CFG.S))
        for dy in range(-3 * hh, 3 * hh + 1):
            y = y0 + dy
            if 0 <= y < H:
                img[y] += (fall * np.exp(-(dy / hh) ** 2))[:, None] * (np.float32([0.55, 0.75, 1.0]) * (0.012 * lum * vis * strength))


class Motes:
    """particulas suspensas (polen/poeira/insetos) iluminadas em contraluz."""

    def __init__(self, n=260, seed=9, box=((-14, 14), (0.1, 5.0), (-12, 22))):
        r = np.random.default_rng(seed)
        self.box = box
        self.p = np.stack([r.uniform(*box[0], n), r.uniform(*box[1], n), r.uniform(*box[2], n)], 1)
        self.ph = r.random((n, 3)) * 6.283
        self.sp = r.uniform(0.2, 1.0, n)
        self.sz = r.uniform(0.004, 0.011, n)
        self.b = r.uniform(0.3, 1.0, n)

    def positions(self, t, wind=(0.25, 0.0, 0.08)):
        p = self.p.copy()
        w = np.asarray(wind)
        p += w[None] * t * self.sp[:, None]
        p[:, 0] += 0.25 * np.sin(self.ph[:, 0] + t * 0.7 * self.sp)
        p[:, 1] += 0.18 * np.sin(self.ph[:, 1] + t * 0.9 * self.sp) + 0.02 * t * (self.sp - 0.5)
        p[:, 2] += 0.25 * np.sin(self.ph[:, 2] + t * 0.6 * self.sp)
        for k in range(3):
            lo, hi = self.box[k]
            p[:, k] = lo + np.mod(p[:, k] - lo, hi - lo)
        return p

    def draw(self, img, cam, t, col, gain=1.0, focus=16.0, zmax=None):
        from core import splat_disc
        p = self.positions(t)
        x, y, z = cam.project(p)
        H, W = img.shape[:2]
        for i in range(len(p)):
            if z[i] < 0.4 or (zmax is not None and z[i] > zmax):
                continue
            if not (-40 < x[i] < W + 40 and -40 < y[i] < H + 40):
                continue
            r = max(self.sz[i] * cam.f / z[i], 0.55 * CFG.S)
            coc = abs(1.0 / z[i] - 1.0 / focus) * cam.f * 0.035      # circulo de confusao
            rr = r + coc
            a = self.b[i] * gain * (r * r) / (rr * rr) * (0.6 + 0.4 * np.sin(self.ph[i, 0] + t * 3.0 * self.sp[i]))
            if a < 0.003:
                continue
            splat_disc(img, x[i], y[i], rr, col, alpha=a, soft=max(1.0, rr * 0.35), additive=True)
