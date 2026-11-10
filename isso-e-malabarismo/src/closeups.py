# closeups.py — closes: mao erguendo a bola contra o sol; bola caida na relva sob chuva
import numpy as np
import cv2
import sky as skylib
import fx
from core import *
from scene import World
from figure import ball_sprite
from shots import finish_frame
from atmos import exposure_for

SKIN = np.array([0.33, 0.185, 0.11], np.float32)


def hull_poly(circles, n=18):
    pts = []
    for (c, r) in circles:
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts.append(np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)], 1))
    return cv2.convexHull(np.concatenate(pts).astype(np.float32)).reshape(-1, 2)


def hand_polys(C, u, open_=0.0, flip=1.0, lift=(0.0, 0.0), mirror=1.0):
    """mao segurando a bola pelas pontas dos dedos, por baixo (flip=1) ou por cima (flip=-1).
    C = centro da bola, u = raio da bola (proporcoes reais: palma ~2.4u de largura)."""
    C = np.asarray(C, float)

    def pol(ang_deg, dist):
        a = np.radians(ang_deg)
        return C + np.array([np.cos(a) * mirror, np.sin(a) * flip]) * dist * u

    polys = []
    palm_c = C + np.array([0.18 * mirror, 3.05 * flip]) * u
    wrist = C + np.array([0.42 * mirror, 4.15 * flip]) * u
    arm = C + np.array([1.55 * mirror, 9.5 * flip]) * u
    polys.append(hull_poly([(wrist, 0.82 * u), (arm, 1.02 * u)]))
    fingers = [  # (ang ponta, ang no, dist no, raios, abertura)
        (172.0, 148.0, 2.75, (0.33, 0.29, 0.24), -26.0),
        (136.0, 119.0, 2.10, (0.27, 0.245, 0.205), -12.0),
        (104.0, 97.0, 2.22, (0.275, 0.25, 0.21), -2.0),
        (72.0, 76.0, 2.12, (0.265, 0.24, 0.20), 9.0),
        (42.0, 57.0, 1.98, (0.235, 0.21, 0.175), 22.0),
    ]
    mcps = []
    for (at, ak, dk, rad, spread) in fingers:
        T = pol(at + spread * open_, 0.86 + 0.75 * open_)
        K = pol(ak + spread * open_ * 0.3, dk)
        mid = 0.5 * (K + T)
        out = mid - C
        out = out / (np.linalg.norm(out) + 1e-6)
        Mj = mid + out * (0.30 + 0.10 * open_) * u
        polys.append(hull_poly([(K, rad[0] * u), (Mj, rad[1] * u)]))
        polys.append(hull_poly([(Mj, rad[1] * u), (T, rad[2] * u)]))
        mcps.append((K, rad[0] * u * 1.05))
    polys.append(hull_poly(mcps[1:] + [(palm_c, 1.02 * u), (wrist, 0.80 * u)]))
    polys.append(hull_poly([mcps[0], (palm_c + np.array([-0.55 * mirror, 0.0]) * u, 0.62 * u)]))
    return polys


def raster(polys, W, H):
    m = np.zeros((H, W), np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.round(p * 16).astype(np.int32).reshape(-1, 1, 2)], 255, lineType=cv2.LINE_AA, shift=4)
    return m.astype(np.float32) / 255.0


def ball_layer(C, Rb, lc, sunCol, amb, sunvis, rimk=3.0, glow=0.0, seam_rot=0.0, wet=0.0, rng=None):
    """retorna (rgb premult, alfa, x0, y0)."""
    n = int(2 * Rb + 6)
    spr, a = ball_sprite(np.asarray(lc, float), sunCol, amb, sunvis, 256, rimk, glow)
    spr = cv2.resize(spr, (n, n), interpolation=cv2.INTER_AREA)
    a = cv2.resize(a, (n, n), interpolation=cv2.INTER_AREA)
    # costuras do saquinho: duas curvas
    lay = np.zeros((n, n), np.uint8)
    for k, (ax_, ph) in enumerate(((0.55, 0.0), (0.42, 1.9))):
        th = np.linspace(0, 2 * np.pi, 80)
        ex = np.cos(th) * Rb * 0.98
        ey = np.sin(th) * Rb * ax_ * 0.98
        rot = seam_rot + ph
        x = n / 2 + ex * np.cos(rot) - ey * np.sin(rot)
        y = n / 2 + ex * np.sin(rot) + ey * np.cos(rot)
        front = np.sin(th + 0.4 * k) > -0.2
        pts = np.stack([x, y], 1)
        for i in range(len(th) - 1):
            if front[i]:
                cv2.line(lay, (int(pts[i, 0] * 16), int(pts[i, 1] * 16)), (int(pts[i + 1, 0] * 16), int(pts[i + 1, 1] * 16)),
                         255, max(1, int(Rb * 0.022)), cv2.LINE_AA, shift=4)
    seam = cv2.GaussianBlur(lay.astype(np.float32) / 255.0, (0, 0), max(0.6, Rb * 0.006))
    spr = spr * (1 - 0.38 * seam[..., None])
    return spr * a[..., None], a, int(round(C[0] - n / 2)), int(round(C[1] - n / 2))


class HandBallShot:
    def __init__(self, name='bola', dur=5.0, hour=6.42, seed=14):
        self.name, self.dur, self.hour, self.seed = name, dur, hour, seed
        self.n = int(round(dur * FPS))
        self.rng = np.random.default_rng(seed)

    def setup(self):
        W, H, S = CFG.W, CFG.H, CFG.S
        sun = skylib.sun_dir(self.hour)
        az = np.arctan2(sun[0], sun[2])
        el = np.arcsin(sun[1])
        cam = Camera((0.0, 0.3, 0.0), yaw=az - np.radians(1.6), pitch=el + np.radians(2.3), vfov=16.0)
        Wd = World(cam, with_meadow=False, cloud_div=2)
        L = Wd.lighting(sun)
        at = Wd.atmos
        grad = at.sky_gradient(sun)
        cl, _ = at.clouds(sun, L, dict(coverage=0.22, sigma=55.0, tmax=95.0, wwx=0.05, wwz=0.55), N=72)
        img, calpha, info = at.compose(grad, L, cl, None, sun_disc=False)
        Wd.ridges.render(img, L, at.horizon_colors(grad), fog=0.9, vis_km=60.0)
        self.bg = cv2.GaussianBlur(img, (0, 0), 30 * S)
        self.L = L
        sx, sy, _ = cam.project_dir(sun)
        self.Ps = np.array([float(sx), float(sy)])
        self.expo = exposure_for(L['el']) * 1.15
        self.Rb = 0.098 * H
        return self

    def state(self, t):
        W, H = CFG.W, CFG.H
        Ps, Rb = self.Ps, self.Rb
        start = np.array([Ps[0] - 0.10 * W, 1.22 * H])
        u = ease_io(np.clip(t / 1.9, 0, 1), 2.2)
        C = start + (Ps - start) * u
        C = C + np.array([np.sin(t * 1.3) * 0.004 * W, np.sin(t * 1.9 + 1) * 0.004 * H]) * float(smooth(1.5, 2.2, t))
        open_, lift = 0.0, np.zeros(2)
        t_dip, t_rel = 3.05, 3.42
        hand_C = C.copy()
        ballC = C.copy()
        rot = 0.25 * t
        if t >= t_dip:
            # mergulho curto e lancamento
            d = float(smooth(t_dip, t_dip + 0.22, t)) * (1 - float(smooth(t_dip + 0.22, t_rel, t)))
            rise = float(smooth(t_dip + 0.20, t_rel, t))
            hand_y = 0.032 * H * d - 0.20 * H * rise          # mergulho curto: a bola nao descobre o sol antes de subir
            hand_C = C + np.array([0.0, hand_y])
            ballC = hand_C.copy()
        if t >= t_rel:
            dt = t - t_rel
            v0 = -2.9 * H
            ballC = C + np.array([0.004 * W * dt, -0.20 * H + v0 * dt + 0.5 * 2.2 * H * dt * dt])
            rot += 5.0 * dt
            open_ = float(smooth(0.0, 0.12, dt))
            fall = float(smooth(0.12, 0.95, dt))
            hand_C = C + np.array([0.05 * W * fall, -0.20 * H + 0.03 * H * (1 - np.exp(-dt * 9)) + 1.7 * H * fall ** 1.6])
        return ballC, hand_C, open_, rot

    def frame(self, i):
        t = i / FPS
        W, H, S = CFG.W, CFG.H, CFG.S
        L, Ps, Rb = self.L, self.Ps, self.Rb
        ballC, handC, open_, rot = self.state(t)
        img = self.bg.copy()
        polys = hand_polys(handC, Rb, open_)
        hm = raster(polys, W, H)
        # mascara da bola
        bm = np.zeros((H, W), np.uint8)
        cv2.circle(bm, (int(ballC[0] * 16), int(ballC[1] * 16)), int(Rb * 16), 255, -1, cv2.LINE_AA, shift=4)
        bm = bm.astype(np.float32) / 255.0
        occ = np.maximum(hm, bm)
        # sol (disco desfocado) atras
        rs = 0.050 * H
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt((xx - Ps[0]) ** 2 + (yy - Ps[1]) ** 2)
        disc = np.clip((rs - d) / (0.35 * rs) + 0.5, 0, 1)
        sunL = L['sunCol'] * 14.0
        vis = float((disc * (1 - occ)).sum() / max(disc.sum(), 1e-6))
        img += (disc * (1 - occ))[..., None] * sunL[None, None]
        # halo e veu: so da parte visivel do sol
        fx.sun_glow(img, Ps, L['sunCol'] * 1.8, vis=vis, size=1.5, strength=1.25)
        # coroa junto a borda da bola quando ela cobre o sol
        dcen = float(np.linalg.norm(ballC - Ps))
        cover = float(smooth(Rb * 1.25, Rb * 0.35, dcen))
        if cover > 0.01:
            db = np.sqrt((xx - ballC[0]) ** 2 + (yy - ballC[1]) ** 2)
            ring = np.exp(-np.clip(db - Rb, 0, None) / (0.10 * Rb)) * (db > Rb - 1.5) * 0.9 + np.exp(-np.clip(db - Rb, 0, None) / (0.9 * Rb)) * 0.16
            toward = np.clip(1.0 - np.sqrt((xx - Ps[0]) ** 2 + (yy - Ps[1]) ** 2) / (Rb * 2.6), 0, 1)
            img += (ring * toward * cover * (1 - hm))[..., None] * (L['sunCol'] * 1.5)[None, None]
        # bola: luz por tras, na direcao do sol
        dl = (Ps - ballC) / (Rb * 2.2)
        lc = np.array([np.clip(dl[0], -1, 1), -np.clip(dl[1], -1, 1), -1.0])
        lc /= np.linalg.norm(lc)
        near = float(smooth(Rb * 3.2, Rb * 0.6, dcen))
        rgb, a, x0, y0 = ball_layer(ballC, Rb, lc, L['sunCol'], L['skyAvg'] * (0.42 - 0.20 * near), 1.0, rimk=2.6 + 3.0 * near, glow=0.0, seam_rot=rot)
        over(img, rgb, a, x0, y0)
        # mao: silhueta com contorno quente e translucidez nas bordas
        amb = L['skyAvg']
        hcol = SKIN * (amb * 0.55 + L['sunCol'] / np.pi * 0.02)
        img = img * (1 - hm[..., None]) + hcol[None, None] * hm[..., None]
        er = cv2.GaussianBlur(hm, (0, 0), 7 * S)
        edge = np.clip(hm - er, 0, 1) * hm
        prox = np.exp(-(d / (0.40 * H)) ** 2)
        rimc = np.array([1.0, 0.36, 0.12], np.float32) * (L['sunCol'].max() / np.pi)
        img += (edge * prox * 1.5 * (0.35 + 0.65 * (1 - cover * 0.5)))[..., None] * rimc[None, None]
        thin = cv2.GaussianBlur(hm, (0, 0), 16 * S)
        sss = np.clip(hm - thin, 0, 1) * hm * prox
        img += sss[..., None] * (np.array([0.9, 0.16, 0.05], np.float32) * (L['sunCol'].max() / np.pi * 0.7))[None, None]
        # pos
        fx.lens_flare(img, Ps, L['sunCol'], vis=vis, strength=1.3)
        img = bloom(img, 0.14, wide=1.6)
        vignette(img, 0.34)
        ex = self.expo * (1.0 + 1.6 * float(smooth(4.25, 5.0, t)))
        disp = tonemap(img, ex)
        grain(disp, 0.018, self.rng)
        return finish_frame(disp, t, self.dur, 0.0, 0.0, 1.6, self.seed, 1.0 + 0.03 * t / self.dur)


def blade_layer(W, H, n, y_base, hgt, rng, width, lean=0.25, dark=(0.25, 0.9)):
    """camada de folhas de capim (alfa, valor)."""
    A = np.zeros((H, W), np.uint8)
    V = np.zeros((H, W), np.uint8)
    for i in range(n):
        x0 = rng.uniform(-0.05 * W, 1.05 * W)
        h = hgt * rng.uniform(0.45, 1.0)
        th = rng.normal(0, lean)
        w = width * rng.uniform(0.6, 1.3)
        us = np.linspace(0, 1, 6)
        px = x0 + h * np.sin(th) * us ** 1.7
        py = y_base(x0) - h * (us - 0.14 * us ** 2)
        wt = w * (1 - us ** 1.4) + 0.4
        Lp = np.stack([px - wt / 2, py], 1)
        Rp = np.stack([px + wt / 2, py], 1)[::-1]
        poly = np.round(np.concatenate([Lp, Rp]) * 16).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(A, [poly], 255, lineType=cv2.LINE_AA, shift=4)
        cv2.fillPoly(V, [poly], int(255 * rng.uniform(*dark)), lineType=cv2.LINE_8, shift=4)
    return A.astype(np.float32) / 255.0, V.astype(np.float32) / 255.0


class GroundBallShot:
    def __init__(self, name='bola_chao', dur=4.0, seed=15):
        self.name, self.dur, self.seed = name, dur, seed
        self.n = int(round(dur * FPS))
        self.rng = np.random.default_rng(seed)

    def setup(self):
        W, H, S = CFG.W, CFG.H, CFG.S
        rng = self.rng
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        g = yy / H
        # fundo: crepusculo azulado, relva distante desfocada
        top = np.array([0.060, 0.070, 0.115], np.float32)
        mid = np.array([0.030, 0.045, 0.050], np.float32)
        bot = np.array([0.010, 0.020, 0.014], np.float32)
        bg = top[None, None] * (1 - smooth(0.0, 0.42, g))[..., None] + mid[None, None] * (smooth(0.0, 0.42, g) * (1 - smooth(0.42, 1.0, g)))[..., None] \
            + bot[None, None] * smooth(0.42, 1.0, g)[..., None]
        n = skylib.uniformize(skylib.fbm_nd((256, 256), 2.6, 77, lo=2.0))
        nb = cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)
        bg = bg * (0.65 + 0.7 * nb[..., None])
        # brilho quente residual no horizonte a esquerda
        bg += (np.exp(-((g - 0.30) / 0.10) ** 2) * np.exp(-((xx / W - 0.12) / 0.38) ** 2))[..., None] * np.array([0.22, 0.085, 0.045], np.float32)
        # bokeh de flores
        for k in range(46):
            x, y = rng.uniform(0, W), rng.uniform(0.34 * H, 0.9 * H)
            r = rng.uniform(10, 34) * S
            col = np.array([[0.20, 0.15, 0.02], [0.16, 0.16, 0.18], [0.10, 0.05, 0.16]][rng.integers(0, 3)], np.float32) * rng.uniform(0.25, 0.8)
            splat_disc(bg, x, y, r, col, alpha=0.5, soft=r * 0.35, additive=True)
        self.bg = cv2.GaussianBlur(bg.astype(np.float32), (0, 0), 6 * S)
        self.C = np.array([0.46 * W, 0.66 * H])
        self.Rb = 0.135 * H
        dn = skylib.uniformize(skylib.fbm_nd((128, 128), 2.2, 91, lo=2.0))
        self.dirt = dn
        base_back = lambda x: 0.80 * H + 0.03 * H * np.sin(x / W * 5)
        base_front = lambda x: 1.04 * H
        Ab, Vb = blade_layer(W, H, 520, base_back, 0.50 * H, rng, 20 * S, 0.30)
        self.back = (cv2.GaussianBlur(Ab, (0, 0), 5.0 * S), cv2.GaussianBlur(Vb * Ab, (0, 0), 5.0 * S))
        Am, Vm = blade_layer(W, H, 260, lambda x: 0.93 * H, 0.42 * H, rng, 24 * S, 0.32)
        self.mid = (cv2.GaussianBlur(Am, (0, 0), 1.4 * S), cv2.GaussianBlur(Vm * Am, (0, 0), 1.4 * S))
        Af, Vf = blade_layer(W, H, 120, base_front, 0.55 * H, rng, 54 * S, 0.36)
        self.front = (cv2.GaussianBlur(Af, (0, 0), 13.0 * S), cv2.GaussianBlur(Vf * Af, (0, 0), 13.0 * S))
        self.amb = np.array([0.105, 0.135, 0.215], np.float32)
        self.key = np.array([1.0, 0.42, 0.20], np.float32) * 0.55
        # respingos de chuva sobre a bola
        self.splash = [(rng.uniform(0.2, 3.9), rng.uniform(-0.75, 0.75), rng.uniform(-0.9, -0.25)) for _ in range(34)]
        self.beads = [(rng.uniform(-0.8, 0.8), rng.uniform(-0.85, 0.3), rng.uniform(0.012, 0.03)) for _ in range(60)]
        return self

    def frame(self, i):
        t = i / FPS
        W, H, S = CFG.W, CFG.H, CFG.S
        rng = np.random.default_rng(i * 13 + self.seed)
        img = self.bg.copy()
        C0, Rb = self.C, self.Rb
        # mao desce, fecha, leva a bola
        t_in, t_grab, t_out = 1.45, 2.35, 2.55
        open_ = 1.0 - float(smooth(t_in + 0.45, t_grab, t))
        appr = 1.0 - ease_io(np.clip((t - t_in) / (t_grab - t_in), 0, 1), 2.0)
        liftb = ease_io(np.clip((t - t_out) / 0.85, 0, 1), 2.4) * 1.5 * H
        C = C0 + np.array([0.012 * W * (liftb / (1.5 * H)), -liftb])
        handC = C + np.array([0.06 * W, -0.80 * H]) * appr
        grass_col = np.array([0.050, 0.095, 0.035], np.float32)

        def put(layer, k):
            A, VA = layer
            col = grass_col[None, None] * (self.amb[None, None] * 1.0) * (VA[..., None] * k)
            img[:] = img * (1 - A[..., None]) + col
        put(self.back, 0.8)
        # bola
        lc = np.array([-0.55, 0.50, 0.67])
        lc /= np.linalg.norm(lc)
        rgb, a, x0, y0 = ball_layer(C, Rb, lc, self.key * np.pi, self.amb * 1.25, 1.0, rimk=0.0, glow=0.0, seam_rot=0.5)
        # sombra/umidade na base e gotas
        nn = a.shape[0]
        by, bx = np.mgrid[-1:1:nn * 1j, -1:1:nn * 1j].astype(np.float32)
        rgb = rgb * (1 - 0.55 * smooth(0.25, 1.0, by))[..., None]
        drt = cv2.resize(self.dirt, (nn, nn), interpolation=cv2.INTER_CUBIC)
        dm = (smooth(0.50, 0.78, drt) * (0.35 + 0.65 * smooth(-0.2, 0.9, by)))[..., None]
        rgb = rgb * (1 - dm * np.array([0.42, 0.52, 0.62], np.float32))
        spec = np.exp(-(((bx + 0.36) / 0.10) ** 2 + ((by + 0.52) / 0.07) ** 2)) * 1.2 + np.exp(-(((bx + 0.22) / 0.03) ** 2 + ((by + 0.30) / 0.03) ** 2)) * 0.8
        rgb += (spec * a)[..., None] * np.array([0.5, 0.6, 0.8], np.float32) * 0.35
        for (bxk, byk, br) in self.beads:
            if bxk * bxk + byk * byk < 0.8:
                dd = ((bx - bxk) ** 2 + (by - byk) ** 2) / (br * br)
                rgb += (np.exp(-dd) * a)[..., None] * np.array([0.5, 0.6, 0.85], np.float32) * 0.10
                rgb *= (1 - 0.35 * np.exp(-((bx - bxk) ** 2 + (by - byk - br * 0.9) ** 2) / (br * br)) * a)[..., None]
        for (ts, sx, sy) in self.splash:
            dt = t - ts
            if 0 <= dt < 0.22 and liftb < 5:
                rr = (0.02 + 0.22 * dt) * 1.0
                dd = np.abs(np.sqrt((bx - sx) ** 2 + ((by - sy) / 0.55) ** 2) - rr)
                rgb += (np.exp(-(dd / 0.012) ** 2) * a * (1 - dt / 0.22))[..., None] * np.array([0.6, 0.7, 0.9], np.float32) * 0.30
        over(img, rgb, a, x0, y0)
        put(self.mid, 1.1)
        # mao
        if t > t_in - 0.05:
            polys = hand_polys(handC, Rb, open_ * 0.9, flip=-1.0)
            hm = raster(polys, W, H)
            hm = cv2.GaussianBlur(hm, (0, 0), 2.6 * S)
            yy = (np.arange(H, dtype=np.float32) / H)[:, None]
            hcol = SKIN[None, None] * (self.amb * 0.42)[None, None] * (0.55 + 0.75 * yy)[..., None]
            img = img * (1 - hm[..., None]) + hcol * hm[..., None]
            Ms = np.float32([[1, 0, 10 * S], [0, 1, -8 * S]])
            edge = np.clip(hm - cv2.warpAffine(hm, Ms, (W, H)), 0, 1)
            img += cv2.GaussianBlur(edge, (0, 0), 2.0 * S)[..., None] * (self.key * 0.10 + self.amb * 0.25)[None, None]
        put(self.front, 0.7)
        # chuva
        for (n, ln, th, al, bl) in ((300, 34, 1, 0.030, 0.9), (60, 95, 2, 0.050, 2.6)):
            lay = np.zeros((H // 2, W // 2), np.uint8)
            x = rng.uniform(-0.05 * W, 1.05 * W, n) / 2
            y = rng.uniform(-0.1 * H, 1.0 * H, n) / 2
            l = ln * S * rng.uniform(0.6, 1.3, n) / 2
            for k in range(n):
                cv2.line(lay, (int(x[k] * 16), int(y[k] * 16)), (int((x[k] - l[k] * 0.13) * 16), int((y[k] + l[k]) * 16)),
                         int(255 * rng.uniform(0.4, 1.0)), th, cv2.LINE_AA, shift=4)
            lf = cv2.resize(cv2.GaussianBlur(lay.astype(np.float32) / 255.0, (0, 0), bl * max(S, 0.5)), (W, H))
            img += lf[..., None] * (np.array([0.55, 0.65, 0.9], np.float32) * al)
        img = bloom(img, 0.10)
        vignette(img, 0.40)
        disp = tonemap(img, 3.4, sat=0.95)
        grain(disp, 0.022, self.rng)
        return finish_frame(disp, t, self.dur, 0.0, 0.0, 1.4, self.seed, 1.0 + 0.035 * t / self.dur)
