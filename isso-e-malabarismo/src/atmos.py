# atmos.py — placa de ceu (gradiente fisico + cirrus + cumulus + sol/lua/estrelas) e dados de luz
import numpy as np
import cv2
import sky as skylib
from core import CFG, Camera, f32, smooth, blur, splat_disc, dir_from

SUN_E = 20.0
skylib.init()

LAT = np.radians(-22.0)
POLE = np.array([np.cos(LAT), np.sin(-LAT) * -1.0, 0.0])   # polo sul celeste: ao sul (+x), elevacao 22 graus
POLE = np.array([np.cos(np.radians(22.0)), np.sin(np.radians(22.0)), 0.0])


def exposure_for(el_deg, day=0.085, night=15.0):
    """exposicao automatica aproximada em funcao da elevacao do sol (graus)."""
    pts_el = np.array([-18, -12, -8, -6, -4, -2.5, -1, 0, 1.5, 4, 8, 15, 90], dtype=np.float64)
    pts_ex = np.array([night, night * 0.85, 7.5, 4.6, 2.4, 1.05, 0.50, 0.32, 0.22, 0.155, 0.115, 0.095, day])
    return float(np.exp(np.interp(el_deg, pts_el, np.log(pts_ex))))


class Stars:
    def __init__(self, n=9000, seed=5):
        rng = np.random.default_rng(seed)
        v = rng.standard_normal((n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        mag = rng.power(2.6, n) * 6.5            # muitas fracas, poucas fortes
        self.flux = 10 ** (-0.4 * (mag - 6.5)) * 0.0035
        temp = rng.normal(0.0, 1.0, n)
        col = np.stack([1.0 + 0.22 * temp, np.ones(n), 1.0 - 0.28 * temp], 1)
        self.col = np.clip(col, 0.45, 1.6)
        # via lactea: concentra estrelas fracas perto do plano galactico
        self.gal_n = np.array([0.35, 0.50, 0.79])
        self.gal_n /= np.linalg.norm(self.gal_n)
        m = 2600
        u = rng.standard_normal((m, 3))
        u -= np.outer(u @ self.gal_n, self.gal_n)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        u += self.gal_n * rng.normal(0, 0.10, (m, 1))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        self.v = np.concatenate([v, u])
        self.flux = np.concatenate([self.flux, 10 ** (-0.4 * (rng.uniform(5.0, 7.4, m) - 6.5)) * 0.0035])
        self.col = np.concatenate([self.col, np.clip(np.stack([1 + 0.15 * rng.normal(size=m), np.ones(m),
                                                               1 - 0.2 * rng.normal(size=m)], 1), 0.5, 1.5)])
        self.tw = rng.random(len(self.v)) * 6.283

    def rotated(self, ang):
        """rotaciona o ceu em torno do polo celeste (ang em rad)."""
        k = POLE
        c, s = np.cos(ang), np.sin(ang)
        v = self.v
        return v * c + np.cross(k, v) * s + np.outer(v @ k, k) * (1 - c)

    def render(self, cam, ang, t=0.0, gain=1.0, trail=0.0):
        H, W = cam.H, cam.W
        layer = np.zeros((H, W, 3), np.float32)
        v = self.rotated(ang)
        x, y, z = cam.project_dir(v)
        ok = (z > 0.05) & (x > 1) & (x < W - 2) & (y > 1) & (y < H - 2) & (v[:, 1] > -0.02)
        x, y = x[ok], y[ok]
        fl = self.flux[ok] * gain * (0.8 + 0.2 * np.sin(self.tw[ok] + t * 7.0))
        # extincao perto do horizonte
        fl = fl * np.clip(v[ok, 1] * 9.0, 0.0, 1.0)
        col = self.col[ok] * fl[:, None]
        xi, yi = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = (x - xi)[:, None], (y - yi)[:, None]
        np.add.at(layer, (yi, xi), col * (1 - fx) * (1 - fy))
        np.add.at(layer, (yi, xi + 1), col * fx * (1 - fy))
        np.add.at(layer, (yi + 1, xi), col * (1 - fx) * fy)
        np.add.at(layer, (yi + 1, xi + 1), col * fx * fy)
        s = max(CFG.S, 0.5)
        layer = cv2.GaussianBlur(layer, (0, 0), 0.62 * s) * (1.6 * s * s) + cv2.GaussianBlur(layer, (0, 0), 2.6 * s) * 1.2
        return layer

    def milky(self, cam, ang, gain=1.0):
        """faixa difusa da via lactea (baixa resolucao)."""
        w, h = cam.W // 8, cam.H // 8
        xs = (np.arange(w) + 0.5) / w * 2 - 1
        ys = 1 - (np.arange(h) + 0.5) / h * 2
        u, vv = np.meshgrid(xs * cam.tanx, ys * cam.tany)
        d = cam.fwd[None, None] + u[..., None] * cam.right + vv[..., None] * cam.up
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        # desfaz rotacao do ceu
        k = POLE
        c, s = np.cos(-ang), np.sin(-ang)
        d0 = d * c + np.cross(k, d) * s + (d @ k)[..., None] * k * (1 - c)
        b = d0 @ self.gal_n
        e1 = np.cross(self.gal_n, [0, 1, 0]); e1 /= np.linalg.norm(e1)
        e2 = np.cross(self.gal_n, e1)
        lon = np.arctan2(d0 @ e2, d0 @ e1)
        n1 = _mw_noise((lon / 6.283 + 0.5) * 1024, (b * 2.2 + 0.5) * 256)
        band = np.exp(-(b / 0.115) ** 2) * (0.55 + 0.9 * n1) + 0.5 * np.exp(-(b / 0.30) ** 2)
        band *= (0.6 + 0.6 * np.cos(lon - 0.6))
        dust = np.exp(-((b - 0.012 * np.sin(lon * 5)) / 0.028) ** 2) * (0.35 + 0.65 * n1)
        band *= (1 - 0.6 * dust)
        band *= np.clip(d[..., 1] * 5.0, 0, 1)
        colr = np.stack([band * 0.95, band * 0.92, band * 1.08], -1).astype(np.float32) * (0.0045 * gain)
        return cv2.resize(colr, (cam.W, cam.H), interpolation=cv2.INTER_CUBIC)


_mwn = None


def _mw_noise(x, y):
    global _mwn
    if _mwn is None:
        _mwn = skylib.uniformize(skylib.fbm_nd((256, 1024), 2.3, 91, lo=3.0))
    xi = (np.asarray(x) % 1024).astype(np.float32)
    yi = np.clip(np.asarray(y), 0, 255).astype(np.float32)
    return cv2.remap(_mwn, xi, yi, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


STARS = Stars()

_moon_tex = None


def moon_texture(n=256):
    global _moon_tex
    if _moon_tex is None:
        yy, xx = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
        r = np.sqrt(xx * xx + yy * yy)
        f = skylib.uniformize(skylib.fbm_nd((n, n), 2.6, 404, lo=1.5))
        maria = smooth(0.52, 0.70, f)
        alb = 0.95 - 0.42 * maria + 0.06 * skylib.fbm_nd((n, n), 1.8, 405, lo=8)
        limb = np.sqrt(np.clip(1 - r * r, 0, 1)) ** 0.35
        a = np.clip((1.0 - r) * n * 0.5, 0, 1)
        _moon_tex = (alb * limb).astype(np.float32), a.astype(np.float32)
    return _moon_tex


DESPECKLE = 1.0


class Atmos:
    def __init__(self, cam, sky_div=8, cloud_div=4):
        self.cam = cam
        self.sw, self.sh = cam.W // sky_div, cam.H // sky_div
        self.cw, self.ch = cam.W // cloud_div, cam.H // cloud_div

    def light(self, sun):
        """cores de luz para a cena (irradiancia do sol e ambiente do ceu)."""
        el = float(np.degrees(np.arcsin(np.clip(sun[1], -1, 1))))
        sunCol = SUN_E * skylib.sun_trans(1.0, sun[1])
        sunCloud = SUN_E * skylib.sun_trans(3.4, sun[1])
        sunHigh = SUN_E * skylib.sun_trans(9.5, sun[1])
        # hemisferio em baixa resolucao (painel az x el)
        zen = skylib.render_sky(6, 6, skylib.basis(0.0, np.radians(78)), 0.35, 0.35, sun).mean((0, 1))
        hs = []
        for yaw in (0.0, np.pi / 2, np.pi, -np.pi / 2):
            hs.append(skylib.render_sky(6, 4, skylib.basis(yaw, np.radians(14)), 0.9, 0.22, sun).mean((0, 1)))
        hor = np.mean(hs, 0)
        skyAvg = 0.45 * zen + 0.55 * hor
        return dict(el=el, sun=np.asarray(sun, dtype=np.float64), sunCol=sunCol, sunCloud=sunCloud, sunHigh=sunHigh,
                    zen=zen, hor=hor, skyAvg=skyAvg)

    def sky_gradient(self, sun, **kw):
        c = self.cam
        return skylib.render_sky(self.sw, self.sh, c.basis(), c.tanx, c.tany, sun, **kw)

    def clouds(self, sun, L, cloud_kw=None, seed=0, N=56, div=None):
        c = self.cam
        cw, ch = (self.cw, self.ch) if div is None else (c.W // div, c.H // div)
        P = skylib.cloud_params(**(cloud_kw or {}))
        sunv = np.asarray(sun, dtype=np.float64)
        lightdir = sunv
        sunCol = L['sunCloud']
        if L.get('moon_dir') is not None and L['el'] < -6:
            lightdir = L['moon_dir']
            sunCol = L['moonCol']
        ambTop = L['zen'] * 2.6 + L['hor'] * 0.5
        ambBot = L['zen'] * 0.9 + L['hor'] * 0.55 + L.get('groundBounce', 0.0)
        out = skylib.render_clouds(cw, ch, c.basis(), c.tanx, c.tany, lightdir, sunCol, ambTop, ambBot, P, N=N, seed=seed)
        out = skylib.soften_far(out)
        if DESPECKLE > 0:
            # tira o pontilhado do ruido de amostragem (mediana) e arredonda de leve
            rgba = cv2.medianBlur(np.ascontiguousarray(out[..., :4]), 5 if DESPECKLE >= 2 else 3)
            rgba = cv2.GaussianBlur(rgba, (0, 0), 0.75 * DESPECKLE)
            dist = cv2.medianBlur(np.ascontiguousarray(out[..., 4]), 5)
            out = np.dstack([rgba, dist]).astype(np.float32)
        return out, P

    def cirrus(self, sun, L, **kw):
        c = self.cam
        return skylib.render_cirrus(self.cw, self.ch, c.basis(), c.tanx, c.tany, sun, L['sunHigh'],
                                    L['zen'] * 1.6 + L['hor'] * 0.4, **kw)

    def compose(self, grad, L, clouds=None, cirrus=None, sun_disc=True, stars=0.0, star_ang=0.0, t=0.0,
                moon=None, haze_fade=65.0, milky=0.0):
        """monta a placa do ceu em resolucao cheia. retorna (img HDR, alpha_nuvens)."""
        c = self.cam
        W, H = c.W, c.H
        img = cv2.resize(grad, (W, H), interpolation=cv2.INTER_CUBIC)
        np.maximum(img, 0, out=img)
        sun = L['sun']
        if stars > 0.001:
            img += STARS.render(c, star_ang, t, gain=stars)
            if milky > 0.001:
                img += STARS.milky(c, star_ang, gain=stars * milky)
        info = {}
        if moon is not None:
            mx, my, mz = c.project_dir(moon['dir'])
            if mz > 0:
                r = moon.get('size', 1.0) * np.radians(0.26) * c.f
                tex, a = moon_texture()
                n = max(int(2 * r + 2), 4)
                tx = cv2.resize(tex, (n, n), interpolation=cv2.INTER_AREA)
                ta = cv2.resize(a, (n, n), interpolation=cv2.INTER_AREA)
                col = np.asarray(moon['col'], dtype=np.float32)
                x0, y0 = int(round(mx - n / 2)), int(round(my - n / 2))
                from core import over
                over(img, tx[..., None] * col * ta[..., None], ta, x0, y0)
                info['moon_px'] = (mx, my, r)
        if sun_disc and L['el'] > -1.2:
            sx, sy, sz = c.project_dir(sun)
            if sz > 0:
                r = np.radians(0.266) * c.f
                disc = L['sunCol'] * 55.0 + 1e-4
                if -3 * r < sx < W + 3 * r and -3 * r < sy < H + 3 * r:
                    splat_disc(img, sx, sy, r, disc, soft=max(1.2, 0.18 * r))
                info['sun_px'] = (sx, sy, r)
        calpha = np.zeros((H, W, 1), np.float32)
        if cirrus is not None:
            ci = cv2.resize(cirrus, (W, H), interpolation=cv2.INTER_CUBIC)
            a = np.clip(ci[..., 3:4], 0, 1)
            img *= (1 - a)
            img += np.maximum(ci[..., :3], 0)
            calpha = a * 0.6
        if clouds is not None:
            cl = cv2.resize(clouds, (W, H), interpolation=cv2.INTER_CUBIC)
            a = np.clip(cl[..., 3:4], 0, 1)
            f = np.exp(-np.maximum(cl[..., 4:5], 0) / haze_fade)
            crgb = np.maximum(cl[..., :3], 0) * f + img * (a * (1 - f))
            img *= (1 - a)
            img += crgb
            calpha = 1 - (1 - calpha) * (1 - a)
        return img, calpha, info

    def horizon_colors(self, grad, el_deg=0.4):
        """cor do ceu logo acima do horizonte, por coluna (W,3)."""
        c = self.cam
        y = c.elev_to_y(el_deg) / c.H * self.sh
        y = int(np.clip(round(y), 0, self.sh - 1))
        row = grad[y]
        xs = (np.arange(c.W) + 0.5) / c.W * self.sw - 0.5
        xi = np.clip(xs, 0, self.sw - 1.001)
        i0 = xi.astype(int)
        fr = (xi - i0)[:, None]
        return (row[i0] * (1 - fr) + row[i0 + 1] * fr).astype(np.float32)
