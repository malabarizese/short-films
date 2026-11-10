# sky.py — wrapper do skylib.so: atmosfera + nuvens volumetricas
import ctypes
import os
import numpy as np
from scipy.special import erf
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(os.path.dirname(HERE), 'work')
os.makedirs(WORK, exist_ok=True)

_lib = ctypes.CDLL(os.path.join(HERE, 'skylib.so'))
_fp = ctypes.POINTER(ctypes.c_float)
_lib.atm_init.argtypes = [ctypes.c_float] * 4
_lib.sun_trans.argtypes = [ctypes.c_float, ctypes.c_float, _fp]
_lib.sky_render.argtypes = [_fp, ctypes.c_int, ctypes.c_int, _fp, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                            _fp, ctypes.c_float, ctypes.c_int, ctypes.c_float, _fp, ctypes.c_float, _fp]
_lib.clouds_set_noise.argtypes = [_fp, ctypes.c_int, _fp, ctypes.c_int, _fp, ctypes.c_int]
_lib.clouds_render.argtypes = [_fp, ctypes.c_int, ctypes.c_int, _fp, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                               _fp, _fp, _fp, _fp, _fp, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.cirrus_set.argtypes = [_fp, ctypes.c_int]
_lib.cirrus_render.argtypes = [_fp, ctypes.c_int, ctypes.c_int, _fp, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                               _fp, _fp, _fp, _fp]
_lib.cloud_shadow_map.argtypes = [_fp, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                                  ctypes.c_float, _fp]


def _p(a):
    return a.ctypes.data_as(_fp)


def f32(x):
    return np.ascontiguousarray(x, dtype=np.float32)


# ---------- ruido ----------
def fbm_nd(shape, beta, seed, lo=1.0, hi=None):
    rng = np.random.default_rng(seed)
    ks = np.meshgrid(*[np.fft.fftfreq(n) * n for n in shape], indexing='ij')
    kk = np.sqrt(sum(k * k for k in ks))
    kk[(0,) * len(shape)] = 1.0
    amp = kk ** (-beta / 2.0)
    amp[kk < lo] = 0.0
    if hi is not None:
        amp[kk > hi] = 0.0
    ph = rng.random(shape) * 2 * np.pi
    f = np.fft.ifftn(amp * np.exp(1j * ph)).real
    f = (f - f.mean()) / f.std()
    return f


def uniformize(f):
    return (0.5 * (1.0 + erf(f / np.sqrt(2.0)))).astype(np.float32)


def worley3(n, cells, seed):
    rng = np.random.default_rng(seed)
    g = np.arange(cells)
    c = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)
    pts = (c + rng.random((cells ** 3, 3))) / cells
    pts = np.mod(pts, 1.0)
    tree = cKDTree(pts, boxsize=1.0)
    a = (np.arange(n) + 0.5) / n
    q = np.stack(np.meshgrid(a, a, a, indexing='ij'), -1).reshape(-1, 3)
    d, _ = tree.query(q, k=1, workers=2)
    return (1.0 - np.clip(d * cells, 0, 1)).reshape(n, n, n).astype(np.float32)


def build_noise():
    fn = os.path.join(WORK, 'cloudnoise_v2.npz')
    if os.path.exists(fn):
        z = np.load(fn)
        return z['base'], z['detail'], z['weather']
    n = 128
    per = uniformize(fbm_nd((n, n, n), 3.4, 11, lo=2.5, hi=40))
    w4, w8, w16, w32 = [worley3(n, c, 20 + c) for c in (4, 8, 16, 32)]
    wlow = 0.625 * w4 + 0.25 * w8 + 0.125 * w16
    whigh = 0.625 * w8 + 0.25 * w16 + 0.125 * w32
    pw = wlow + per * (1.0 - wlow)            # remap(perlin, 0, 1, worley, 1)
    base = (pw - (whigh - 1.0)) / (1.0 - (whigh - 1.0))   # remap(pw, whigh-1, 1, 0, 1)
    base = np.clip(base, 0, 1).astype(np.float32)
    nd = 32
    d2, d4, d8 = [worley3(nd, c, 50 + c) for c in (2, 4, 8)]
    detail = np.clip(0.625 * d2 + 0.25 * d4 + 0.125 * d8, 0, 1).astype(np.float32)
    nw = 512
    cov = uniformize(fbm_nd((nw, nw), 2.9, 7, lo=2.0, hi=90))
    typ = uniformize(fbm_nd((nw, nw), 3.2, 8, lo=1.5, hi=40))
    weather = np.stack([cov, typ], -1).astype(np.float32)
    np.savez(fn, base=base, detail=detail, weather=weather)
    return base, detail, weather


def build_cirrus():
    fn = os.path.join(WORK, 'cirrus_v1.npy')
    if os.path.exists(fn):
        return np.load(fn)
    n = 1024
    rng = np.random.default_rng(31)
    kx, ky = np.meshgrid(np.fft.fftfreq(n) * n, np.fft.fftfreq(n) * n, indexing='ij')
    kk = np.sqrt((kx * 5.0) ** 2 + ky ** 2)
    kk[0, 0] = 1.0
    amp = kk ** (-1.35)
    amp[np.sqrt(kx ** 2 + ky ** 2) < 2.0] = 0
    f = np.fft.ifft2(amp * np.exp(1j * rng.random((n, n)) * 2 * np.pi)).real
    f = (f - f.mean()) / f.std()
    # dobra de dominio leve para curvar os fios
    w = fbm_nd((n, n), 3.5, 77, lo=1.0, hi=12)
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    import cv2
    mx = ((xx + w * 40.0) % n).astype(np.float32)
    my = ((yy + np.roll(w, 300, 0) * 40.0) % n).astype(np.float32)
    f = cv2.remap(f.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    out = uniformize(f)
    np.save(fn, out)
    return out


_noise = None
_cirrus = None


def init(mie_mult=1.0, ozone_mult=1.0, mie_g=0.80, mie_h=1.2):
    global _noise
    _lib.atm_init(mie_mult, ozone_mult, mie_g, mie_h)
    if _noise is None:
        b, d, w = build_noise()
        _noise = (f32(b), f32(d), f32(w))
        _lib.clouds_set_noise(_p(_noise[0]), _noise[0].shape[0], _p(_noise[1]), _noise[1].shape[0],
                              _p(_noise[2]), _noise[2].shape[0])
        global _cirrus
        _cirrus = f32(build_cirrus())
        _lib.cirrus_set(_p(_cirrus), _cirrus.shape[0])


def basis(yaw, pitch):
    """x=direita (sul), y=cima, z=frente (leste) com yaw=0."""
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    right = [cy, 0.0, -sy]
    up = [-sy * sp, cp, -cy * sp]
    fwd = [sy * cp, sp, cy * cp]
    return f32(right + up + fwd)


def dir_from(az, el):
    return f32([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])


def sun_dir(hour, lat_deg=-22.0, decl_deg=0.0):
    """hora solar local (12 = meio-dia). Mundo: +z leste, +x sul, +y cima."""
    lat, dec = np.radians(lat_deg), np.radians(decl_deg)
    H = np.radians((hour - 12.0) * 15.0)
    up = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(H)
    east = -np.cos(dec) * np.sin(H)
    north = np.cos(lat) * np.sin(dec) - np.sin(lat) * np.cos(dec) * np.cos(H)
    v = np.array([-north, up, east], dtype=np.float64)
    return f32(v / np.linalg.norm(v))


def sun_trans(h_km, mu):
    T = np.zeros(3, np.float32)
    _lib.sun_trans(h_km, float(mu), _p(T))
    return T


def twilight_fill(sun_el_deg, k=1.0, night=0.012):
    """luz difusa de segunda ordem no crepusculo (azul), independente da sombra da Terra."""
    e = 1.0 if sun_el_deg >= 0 else float(np.exp(-((-sun_el_deg) / 4.2) ** 1.35))
    e = max(e, night)
    return f32([0.10, 0.30, 1.0]) * (0.05 * k * e)


def render_sky(W, H, bas, tanx, tany, sun, h0=1.0, sunE=20.0, N=32, msK=0.7, msTint=(0.30, 0.55, 1.0), msLift=14.0,
               tw=None):
    out = np.zeros((H, W, 3), np.float32)
    if tw is None:
        tw = twilight_fill(np.degrees(np.arcsin(np.clip(sun[1], -1, 1))))
    _lib.sky_render(_p(out), W, H, _p(bas), tanx, tany, h0, _p(f32(sun)), sunE, N, msK, _p(f32(msTint)), msLift,
                    _p(f32(tw)))
    return out


CP_FIELDS = ['hb', 'ht', 'bs', 'ds', 'ws', 'bwx', 'bwz', 'evol', 'dwx', 'dwy', 'dwz', 'wwx', 'wwz',
             'coverage', 'cov_soft', 'type', 'type_var', 'sigma', 'erode', 'dens_mult', 'tmax',
             'sun_boost', 'amb_boost', 'powder', 'farK', 'cx', 'cz', 'lcmax']
CP_DEFAULT = dict(hb=2.7, ht=4.1, bs=1 / 4.6, ds=1 / 0.75, ws=1 / 60.0, bwx=0, bwz=0, evol=0, dwx=0, dwy=0, dwz=0,
                  wwx=0, wwz=0, coverage=0.5, cov_soft=0.30, type=0.9, type_var=0.5, sigma=30.0, erode=0.40,
                  dens_mult=1.0, tmax=130.0, sun_boost=7.0, amb_boost=0.55, powder=0.6, farK=1.0, cx=0.0, cz=0.0, lcmax=0.72)


def cloud_params(**kw):
    d = dict(CP_DEFAULT)
    d.update(kw)
    return f32([d[k] for k in CP_FIELDS])


def render_clouds(W, H, bas, tanx, tany, sun, sunCol, ambTop, ambBot, params, h0=1.0, N=56, NL=6, seed=0):
    out = np.zeros((H, W, 5), np.float32)
    _lib.clouds_render(_p(out), W, H, _p(bas), tanx, tany, h0, _p(f32(sun)), _p(f32(sunCol)), _p(f32(ambTop)),
                       _p(f32(ambBot)), _p(params), N, NL, seed)
    return out


def shadow_map(W, H, x0, z0, x1, z1, params):
    out = np.zeros((H, W), np.float32)
    _lib.cloud_shadow_map(_p(out), W, H, x0, z0, x1, z1, _p(params))
    return out


def render_cirrus(W, H, bas, tanx, tany, sun, sunCol, amb, h0=1.0, alt=9.5, scale=1 / 55.0, offx=0.0, offz=0.0,
                  amount=0.30, soft=0.35, opacity=0.22, rot=0.5, stretch=1.0, sun_boost=5.0, amb_boost=0.5, ss=2):
    import cv2
    out = np.zeros((H * ss, W * ss, 4), np.float32)
    prm = f32([alt, scale, offx, offz, amount, soft, opacity, rot, stretch, sun_boost, amb_boost])
    _lib.cirrus_render(_p(out), W * ss, H * ss, _p(bas), tanx, tany, h0, _p(f32(sun)), _p(f32(sunCol)), _p(f32(amb)),
                       _p(prm))
    if ss > 1:
        out = cv2.resize(out, (W, H), interpolation=cv2.INTER_AREA)
    return out


def soften_far(c, d0=35.0, d1=110.0, sigma=2.2):
    """suaviza nuvens distantes (ruido de amostragem + bruma)."""
    import cv2
    w = np.clip((c[..., 4:5] - d0) / (d1 - d0), 0, 1) * (c[..., 3:4] > 0)
    b = cv2.GaussianBlur(c[..., :4], (0, 0), sigma)
    wb = cv2.GaussianBlur(w, (0, 0), sigma)[..., None]
    wb = np.maximum(w, wb * 0.999)
    out = c.copy()
    out[..., :4] = c[..., :4] * (1 - wb) + b * wb
    return out
