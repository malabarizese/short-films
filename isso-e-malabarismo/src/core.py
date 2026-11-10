# core.py — configuracao, camera, composicao e pos-processamento
import os
import numpy as np
import cv2

cv2.setNumThreads(int(os.environ.get('CV_THREADS', '2')))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORK = os.path.join(ROOT, 'work')
OUT = os.path.join(ROOT, 'out')
BRAND = os.path.join(ROOT, 'brand')
FPS = 30


def _aspect():
    a, b = os.environ.get('FILM_ASPECT', '16:9').split(':')
    return float(a) / float(b)


class Cfg:
    def __init__(self, H=1440):
        self.H = int(H)
        self.W = int(round(H * _aspect() / 2)) * 2      # FILM_ASPECT=9:16 para o formato vertical
        self.S = H / 1440.0


CFG = Cfg(int(os.environ.get('FILM_H', '1440')))


def set_height(H):
    global CFG
    CFG.H = int(H)
    CFG.W = int(round(H * _aspect() / 2)) * 2
    CFG.S = H / 1440.0


def f32(x):
    return np.ascontiguousarray(x, dtype=np.float32)


def smooth(a, b, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def ease(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def ease_io(t, p=2.0):
    t = np.clip(t, 0.0, 1.0)
    return np.where(t < 0.5, 0.5 * (2 * t) ** p, 1 - 0.5 * (2 * (1 - t)) ** p)


def srgb_to_lin(c):
    c = np.asarray(c, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def hexlin(h):
    h = h.lstrip('#')
    return srgb_to_lin([int(h[i:i + 2], 16) for i in (0, 2, 4)])


class Camera:
    """Mundo: +x direita (sul), +y cima, +z frente (leste). Unidades: metros."""

    def __init__(self, pos, yaw=0.0, pitch=0.0, vfov=36.0, W=None, H=None):
        self.W = W or CFG.W
        self.H = H or CFG.H
        self.pos = np.asarray(pos, dtype=np.float64)
        self.yaw, self.pitch, self.vfov = yaw, pitch, vfov
        self.update()

    def update(self):
        cy, sy, cp, sp = np.cos(self.yaw), np.sin(self.yaw), np.cos(self.pitch), np.sin(self.pitch)
        self.right = np.array([cy, 0.0, -sy])
        self.up = np.array([-sy * sp, cp, -cy * sp])
        self.fwd = np.array([sy * cp, sp, cy * cp])
        self.tany = np.tan(np.radians(self.vfov) / 2)
        self.tanx = self.tany * self.W / self.H
        self.f = (self.H / 2) / self.tany

    def basis(self):
        return f32(np.concatenate([self.right, self.up, self.fwd]))

    def project(self, P):
        d = np.asarray(P, dtype=np.float64) - self.pos
        xc = d @ self.right
        yc = d @ self.up
        zc = d @ self.fwd
        zs = np.where(np.abs(zc) < 1e-6, 1e-6, zc)
        return self.W / 2 + self.f * xc / zs, self.H / 2 - self.f * yc / zs, zc

    def project_dir(self, d):
        """direcao unitaria -> pixel (objetos no infinito)."""
        d = np.asarray(d, dtype=np.float64)
        xc, yc, zc = d @ self.right, d @ self.up, d @ self.fwd
        zs = np.where(np.abs(zc) < 1e-6, 1e-6, zc)
        return self.W / 2 + self.f * xc / zs, self.H / 2 - self.f * yc / zs, zc

    def px_per_m(self, depth):
        return self.f / depth

    def elev_to_y(self, el_deg, az_deg=0.0):
        d = dir_from(np.radians(az_deg), np.radians(el_deg))
        return self.project_dir(d)[1]


def dir_from(az, el):
    return np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])


# ---------------- composicao ----------------
def over(dst, rgb, a, x0=0, y0=0):
    """dst (H,W,3) float; rgb premultiplicado (h,w,3); a (h,w) ou (h,w,1)."""
    H, W = dst.shape[:2]
    h, w = rgb.shape[:2]
    x0, y0 = int(x0), int(y0)
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + w, W), min(y0 + h, H)
    if xb <= xa or yb <= ya:
        return
    sx, sy = xa - x0, ya - y0
    r = rgb[sy:sy + yb - ya, sx:sx + xb - xa]
    al = a[sy:sy + yb - ya, sx:sx + xb - xa]
    if al.ndim == 2:
        al = al[..., None]
    roi = dst[ya:yb, xa:xb]
    roi *= (1.0 - al)
    roi += r


def add(dst, rgb, x0=0, y0=0):
    H, W = dst.shape[:2]
    h, w = rgb.shape[:2]
    x0, y0 = int(x0), int(y0)
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + w, W), min(y0 + h, H)
    if xb <= xa or yb <= ya:
        return
    sx, sy = xa - x0, ya - y0
    dst[ya:yb, xa:xb] += rgb[sy:sy + yb - ya, sx:sx + xb - xa]


def blur(img, sigma):
    if sigma <= 0.05:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma)


def resize(img, W, H, interp=cv2.INTER_LINEAR):
    return cv2.resize(img, (int(W), int(H)), interpolation=interp)


_disc_cache = {}


def soft_disc(r, soft=1.0):
    """disco antialiasado, raio r px. retorna (patch, half)."""
    key = (round(r * 8) / 8, soft)
    if key in _disc_cache:
        return _disc_cache[key]
    R = int(np.ceil(r + soft + 1))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
    d = np.sqrt(xx * xx + yy * yy)
    m = np.clip((r - d) / soft + 0.5, 0, 1).astype(np.float32)
    _disc_cache[key] = (m, R)
    return m, R


def splat_disc(dst, x, y, r, col, alpha=1.0, soft=1.0, additive=False):
    """desenha disco suave com posicao subpixel."""
    m, R = soft_disc(r, soft)
    xi, yi = int(np.floor(x)), int(np.floor(y))
    fx, fy = x - xi, y - yi
    if fx > 0.02 or fy > 0.02:
        M = np.float32([[1, 0, fx], [0, 1, fy]])
        m = cv2.warpAffine(m, M, (m.shape[1] + 1, m.shape[0] + 1), flags=cv2.INTER_LINEAR)
    a = m * alpha
    rgb = a[..., None] * np.asarray(col, dtype=np.float32)
    if additive:
        add(dst, rgb, xi - R, yi - R)
    else:
        over(dst, rgb, a, xi - R, yi - R)


# ---------------- pos ----------------
def bloom(img, amount=0.10, S=None, tint=(1.0, 0.92, 0.85), wide=1.0):
    S = S or CFG.S
    H, W = img.shape[:2]
    s4 = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    s16 = cv2.resize(s4, (W // 16, H // 16), interpolation=cv2.INTER_AREA)
    a = cv2.GaussianBlur(s4, (0, 0), 2.2 * S) * 0.34 + cv2.GaussianBlur(s4, (0, 0), 7.0 * S) * 0.26
    b = cv2.GaussianBlur(s16, (0, 0), 5.0 * S) * 0.22 + cv2.GaussianBlur(s16, (0, 0), 16.0 * S * wide) * 0.18
    a += cv2.resize(b, (W // 4, H // 4), interpolation=cv2.INTER_LINEAR)
    up = cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
    up *= np.asarray(tint, dtype=np.float32) * amount
    img *= (1.0 - amount)
    img += up
    return img


_vig_cache = {}


def vignette(img, strength=0.28):
    H, W = img.shape[:2]
    key = (H, W, strength)
    if key not in _vig_cache:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2 * 0.72
        v = 1.0 - strength * smooth(0.25, 1.75, r2)
        _vig_cache[key] = v.astype(np.float32)[..., None]
    img *= _vig_cache[key]
    return img


def tonemap(img, expo=1.0, sat=1.0, contrast=1.0, lift=(0, 0, 0), gain=(1, 1, 1)):
    x = img * np.float32(expo)
    np.maximum(x, 0, out=x)
    if sat != 1.0:
        l = (x[..., 0:1] * 0.2126 + x[..., 1:2] * 0.7152 + x[..., 2:3] * 0.0722)
        x = l + (x - l) * np.float32(sat)
        np.maximum(x, 0, out=x)
    x *= np.asarray(gain, dtype=np.float32)
    y = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    np.clip(y, 0, 1, out=y)
    y = np.power(y, np.float32(1 / 2.2), out=y)
    if contrast != 1.0:
        y = 0.5 + (y - 0.5) * np.float32(contrast) + 0.0
        # curva em S suave mantendo extremos
    if any(lift):
        y = y + np.asarray(lift, dtype=np.float32) * (1.0 - y) ** 2
    return y


_grain = None
GRAIN_SCALE = 0.40      # grao discreto: o bastante para quebrar faixas, sem custar taxa de bits


def grain(img8f, amount=0.035, rng=None):
    """img em espaco de display [0,1]."""
    global _grain
    H, W = img8f.shape[:2]
    if _grain is None or _grain.shape[0] < H + 200 or _grain.shape[1] < W + 200:
        r = np.random.default_rng(1234)
        g = r.standard_normal((H + 256, W + 256)).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), 0.55 * max(CFG.S, 0.6))
        g /= g.std()
        _grain = g
    rng = rng or np.random.default_rng()
    oy, ox = int(rng.integers(0, 256)), int(rng.integers(0, 256))
    g = _grain[oy:oy + H, ox:ox + W]
    if rng.random() < 0.5:
        g = g[:, ::-1]
    lum = img8f[..., 1:2]
    w = 0.35 + 2.6 * lum * (1.0 - lum)
    img8f += (g[..., None] * (amount * GRAIN_SCALE)) * w
    return img8f


def to_u8(img):
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def save_jpg(path, img_disp, q=92, scale=None):
    """img em display RGB float [0,1]."""
    u = to_u8(img_disp)[..., ::-1]
    if scale:
        u = cv2.resize(u, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, np.ascontiguousarray(u), [cv2.IMWRITE_JPEG_QUALITY, q])
