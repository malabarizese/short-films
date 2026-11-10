# figure.py — esqueleto procedural do malabarista (IK) e desenho 2D a partir das juntas 3D
import numpy as np
import cv2
from core import CFG, f32, smooth, over, add, splat_disc, hexlin

# proporcoes (m)
HIP_H = 0.94
HIP_W = 0.095
SH_W = 0.205
UARM, FARM, HAND = 0.30, 0.265, 0.085
THIGH, SHIN = 0.445, 0.44
ANKLE_H = 0.07

COL = dict(skin=np.array([0.33, 0.185, 0.11]), hair=np.array([0.010, 0.008, 0.007]),
           shirt=np.array([0.42, 0.055, 0.035]), pants=np.array([0.022, 0.027, 0.045]),
           bag=np.array([0.20, 0.15, 0.065]), sole=np.array([0.05, 0.03, 0.02]))


def ik2(S, T, a, b, pole):
    """IK de dois ossos. retorna a junta do meio."""
    d = T - S
    L = np.linalg.norm(d)
    L = float(np.clip(L, abs(a - b) + 1e-3, a + b - 1e-3))
    dirn = d / (np.linalg.norm(d) + 1e-9)
    x = (a * a - b * b + L * L) / (2 * L)
    h = np.sqrt(max(a * a - x * x, 0.0))
    p = pole - dirn * (pole @ dirn)
    n = np.linalg.norm(p)
    p = p / n if n > 1e-6 else np.array([0.0, -1.0, 0.0])
    return S + dirn * x + p * h, S + dirn * L


class Pose:
    def __init__(self, origin=(0, 0, 0), yaw=0.0):
        self.origin = np.asarray(origin, float)   # pes (mundo)
        self.yaw = yaw                            # 0 = olhando para +z
        self.root = np.zeros(2)                   # deslocamento local (x, z)
        self.squat = 0.0
        self.bend = 0.0                           # 0..1 inclinacao para a frente
        self.lean = 0.0                           # rad, para tras
        self.head = 0.0                           # graus para cima
        self.hands = {0: None, 1: None}           # alvos locais (centro da bola no slot 0) ou None
        self.feet = {0: None, 1: None}            # alvos locais dos tornozelos
        self.twist = 0.0
        self.shrug = 0.0
        self.breath = 0.0
        self.bag = False
        self.arm_swing = None                     # (esq, dir) deslocamento z do punho ao caminhar
        self.hand_free = {0: 0.0, 1: 0.0}         # 0 = mao no alvo ; 1 = braco solto (transicao suave)
        self.ground = None                        # funcao g(x,z) do mundo

    def solve(self):
        J = {}
        r = self.root
        hipy = HIP_H - self.squat - self.bend * 0.30
        pel = np.array([r[0], hipy + 0.03, r[1] - self.bend * 0.16])
        phi = self.bend * np.radians(74.0) - self.lean
        sd = np.array([0.0, np.cos(phi), np.sin(phi)])
        xh = np.array([np.cos(self.twist), 0.0, -np.sin(self.twist)])
        chest = pel + sd * (0.34 + 0.006 * self.breath)
        neck = pel + sd * 0.515
        hp = np.radians(self.head)
        hd = np.array([0.0, np.cos(phi - hp * 0.55), np.sin(phi - hp * 0.55)])
        head = neck + hd * 0.135
        J['pel'], J['chest'], J['neck'], J['head'] = pel, chest, neck, head
        J['head_dir'] = hd
        J['waist'] = pel + sd * 0.16
        for h, sg in ((0, -1.0), (1, 1.0)):
            sh = neck - sd * 0.074 + xh * (sg * SH_W) + np.array([0, 0.012 * self.shrug + 0.004 * self.breath, 0])
            J['sh%d' % h] = sh
            tgt = self.hands[h]
            sw = 0.0 if self.arm_swing is None else self.arm_swing[h]
            # braco pendente acompanha a inclinacao do tronco
            down = np.array([sg * 0.07, -np.cos(phi * 0.6), np.sin(phi * 0.6) * 0.5 + sw / 0.56])
            down /= np.linalg.norm(down)
            wr_hang = sh + down * (UARM + FARM) * 0.985
            wf = float(self.hand_free.get(h, 0.0))
            if tgt is None:
                wr = wr_hang
                ball = None
                wf = 1.0
            else:
                ball = np.asarray(tgt, float) + np.array([r[0], 0.0, r[1]])
                wr = (ball + np.array([sg * 0.012, -0.050, -0.062])) * (1 - wf) + wr_hang * wf
            pole = np.array([sg * 0.75, -0.55, -0.65])
            el, wr2 = ik2(sh, wr, UARM, FARM, pole)
            J['el%d' % h], J['wr%d' % h] = el, wr2
            fdir = (wr2 - el) / (np.linalg.norm(wr2 - el) + 1e-9)
            hand_free = wr2 + fdir * HAND
            if ball is not None and wf <= 0.0 and np.linalg.norm(wr - wr2) < 0.02:
                J['hand%d' % h] = ball + np.array([0, -0.032, -0.006])
            elif ball is not None and wf < 1.0 and np.linalg.norm(wr - wr2) < 0.02:
                J['hand%d' % h] = (wr2 + np.array([-sg * 0.012, 0.018, 0.056])) * (1 - wf) + hand_free * wf
            else:
                J['hand%d' % h] = hand_free
            hip = pel + xh * (sg * HIP_W) - np.array([0, 0.03, 0])
            J['hip%d' % h] = hip
            ft = self.feet[h]
            if ft is None:
                ft = np.array([sg * 0.13 + r[0], ANKLE_H, r[1] - 0.02])
            else:
                ft = np.asarray(ft, float)
            kn, an = ik2(hip, ft, THIGH, SHIN, np.array([sg * 0.12, 0.0, 1.0]))
            J['kn%d' % h], J['an%d' % h] = kn, an
            J['toe%d' % h] = an + np.array([sg * 0.035, -ANKLE_H + 0.03, 0.17])
            J['heel%d' % h] = an + np.array([0.0, -ANKLE_H + 0.035, -0.055])
        # mundo
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        Rm = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        out = {}
        for k, v in J.items():
            if k == 'head_dir':
                out[k] = Rm @ v
            else:
                out[k] = Rm @ v + self.origin
        out['_R'] = Rm
        out['_sd'] = Rm @ sd
        out['_xh'] = Rm @ xh
        return out

    def to_world(self, p_local):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        Rm = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return Rm @ np.asarray(p_local, float) + self.origin


# ---------------- desenho ----------------
def _circle_pts(c, r, n=14):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)], 1)


def _hull(pts):
    return cv2.convexHull(f32(pts)).reshape(-1, 2)


class Canvas:
    """tela supersampleada com canais: albedo (gamma 2), luz, alfa."""

    def __init__(self, x0, y0, w, h, ss):
        self.x0, self.y0, self.w, self.h, self.ss = x0, y0, w, h, ss
        self.alb = np.zeros((h * ss, w * ss, 3), np.uint8)
        self.lit = np.zeros((h * ss, w * ss), np.uint8)
        self.a = np.zeros((h * ss, w * ss), np.uint8)

    def _pts(self, pts):
        p = (np.asarray(pts, float) - [self.x0, self.y0]) * self.ss
        return np.round(p * 16).astype(np.int32)

    def poly(self, pts, col, lit):
        p = self._pts(pts).reshape(-1, 1, 2)
        c8 = tuple(int(v) for v in np.clip(np.sqrt(col) * 255, 0, 255))
        cv2.fillPoly(self.alb, [p], c8, lineType=cv2.LINE_AA, shift=4)
        cv2.fillPoly(self.lit, [p], int(lit), lineType=cv2.LINE_AA, shift=4)
        cv2.fillPoly(self.a, [p], 255, lineType=cv2.LINE_AA, shift=4)

    def lit_poly(self, pts, lit):
        p = self._pts(pts).reshape(-1, 1, 2)
        cv2.fillPoly(self.lit, [p], int(lit), lineType=cv2.LINE_AA, shift=4)

    def result(self):
        ss = self.ss
        if ss > 1:
            alb = cv2.resize(self.alb, (self.w, self.h), interpolation=cv2.INTER_AREA)
            lit = cv2.resize(self.lit, (self.w, self.h), interpolation=cv2.INTER_AREA)
            a = cv2.resize(self.a, (self.w, self.h), interpolation=cv2.INTER_AREA)
        else:
            alb, lit, a = self.alb, self.lit, self.a
        alb = (alb.astype(np.float32) / 255.0) ** 2
        return alb, lit.astype(np.float32) / 255.0, a.astype(np.float32) / 255.0


def draw_figure(img, cam, J, L, style=None, alpha=1.0, silhouette=0.0, rim=1.0, tint=None, min_ss=2, sun_scale=1.0,
                bag=False, return_mask=False):
    """desenha o malabarista. L = luz (dict). silhouette 0..1 escurece (contraluz forte)."""
    style = style or COL
    names = [k for k in J if not k.startswith('_') and k != 'head_dir']
    P3 = np.array([J[k] for k in names])
    x, y, z = cam.project(P3)
    if z.min() < 0.3:
        return None
    P2 = {k: np.array([x[i], y[i]]) for i, k in enumerate(names)}
    Z = {k: z[i] for i, k in enumerate(names)}
    k = cam.f / float(np.mean(z))                     # px por metro
    pad = 0.22 * k + 4
    x0, y0 = int(np.floor(x.min() - pad)), int(np.floor(y.min() - pad))
    x1, y1 = int(np.ceil(x.max() + pad)), int(np.ceil(y.max() + pad))
    w, h = x1 - x0, y1 - y0
    if w < 2 or h < 2 or x1 < 0 or y1 < 0 or x0 > cam.W or y0 > cam.H:
        return None
    ss = min_ss if k > 140 else (3 if k > 60 else 4)
    cv = Canvas(x0, y0, w, h, ss)
    s = L['sun']
    lc = np.array([s @ cam.right, s @ cam.up, -(s @ cam.fwd)])
    front = max(lc[2], 0.0) ** 0.8
    side = np.array([lc[0], -lc[1]])
    sn = np.linalg.norm(side)
    side = side / sn if sn > 1e-6 else np.zeros(2)
    off_amt = 0.42 * min(sn * 1.3, 1.0)
    lit_hi = 255 * np.clip(0.22 + 0.78 * (front * 0.9 + 0.35 * sn), 0, 1)
    lit_lo = 255 * np.clip(0.10 + 0.50 * front, 0, 1)

    parts = []   # (profundidade, funcao de desenho)

    def capsule(a, b, ra, rb, col, depth=None, lit=True):
        pa, pb = P2[a] if isinstance(a, str) else a, P2[b] if isinstance(b, str) else b
        da = Z[a] if isinstance(a, str) else depth
        db = Z[b] if isinstance(b, str) else depth
        d = 0.5 * (da + db) if depth is None else depth
        Ra, Rb = ra * k, rb * k

        def fn():
            pts = _hull(np.concatenate([_circle_pts(pa, Ra), _circle_pts(pb, Rb)]))
            cv.poly(pts, col, lit_lo)
            if lit:
                o = side * off_amt
                pts2 = _hull(np.concatenate([_circle_pts(pa + o * Ra, Ra * 0.80), _circle_pts(pb + o * Rb, Rb * 0.80)]))
                cv.lit_poly(pts2, lit_hi)
        parts.append((d, fn))

    def blob(pts_r, col, depth, lit=True):
        def fn():
            pts = _hull(np.concatenate([_circle_pts(p, r * k) for p, r in pts_r]))
            cv.poly(pts, col, lit_lo)
            if lit:
                o = side * off_amt
                pts2 = _hull(np.concatenate([_circle_pts(p + o * r * k, r * k * 0.82) for p, r in pts_r]))
                cv.lit_poly(pts2, lit_hi)
        parts.append((depth, fn))

    # largura aparente do tronco: elipse (largura x profundidade) vista do angulo da camera
    xh, sd = J['_xh'], J['_sd']
    view = cam.fwd
    cosv = abs(xh @ cam.right)                        # 1 = de frente/costas, 0 = de perfil

    def tw(wd, dp):
        return np.sqrt((wd * cosv) ** 2 + (dp * np.sqrt(max(1 - cosv * cosv, 0))) ** 2)

    def pt3(p):
        px, py, pz = cam.project(p)
        return np.array([float(px), float(py)]), float(pz)

    # --- pernas (calca solta, afunilada)
    for hnd in (0, 1):
        zl = Z['pel'] + 2.0 + 0.5 * (Z['hip%d' % hnd] + Z['kn%d' % hnd]) * 0.1
        capsule('hip%d' % hnd, 'kn%d' % hnd, 0.086, 0.066, style['pants'], depth=zl)
        capsule('kn%d' % hnd, 'an%d' % hnd, 0.064, 0.046, style['pants'], depth=zl + 0.001)
        capsule('heel%d' % hnd, 'toe%d' % hnd, 0.036, 0.027, style['skin'], depth=zl + 0.002)
    # --- tronco
    pel, wa, ch, nk = J['pel'], J['waist'], J['chest'], J['neck']
    rt = cam.right

    def side_pts(c3, half, rad):
        a, da = pt3(c3 - rt * half)
        b, _ = pt3(c3 + rt * half)
        return [(a, rad), (b, rad)], da
    hp_, dh = side_pts(pel - sd * 0.035, tw(0.100, 0.070), 0.072)
    wp_, _ = side_pts(wa, tw(0.090, 0.060), 0.060)
    cp_, dc = side_pts(ch, tw(0.118, 0.070), 0.062)
    sh_c = nk - sd * 0.085
    sp_, _ = side_pts(sh_c, tw(0.150, 0.060), 0.048)
    dz_t = Z['chest']
    blob(hp_ + wp_, style['pants'], Z['pel'] + 1.9, lit=True)
    nkp, _ = pt3(nk + sd * 0.010)
    hem, _ = side_pts(pel - sd * 0.005, tw(0.112, 0.080), 0.058)

    def torso_fn():
        groups = [wp_ + cp_, cp_ + sp_, sp_ + [(nkp, 0.050)], wp_ + hem]
        for gpts in groups:
            pts = _hull(np.concatenate([_circle_pts(p, r * k) for p, r in gpts]))
            cv.poly(pts, style['shirt'], lit_lo)
        o = side * off_amt
        allp = hem + wp_ + cp_ + sp_ + [(nkp, 0.050)]
        ptsl = _hull(np.concatenate([_circle_pts(p + o * r * k, r * k * 0.80) for p, r in allp]))
        # so ilumina dentro do tronco: recorta pelo alfa ja desenhado nesta etapa
        tmp = np.zeros_like(cv.lit)
        pp = cv._pts(ptsl).reshape(-1, 1, 2)
        cv2.fillPoly(tmp, [pp], int(lit_hi), lineType=cv2.LINE_AA, shift=4)
        tm = np.zeros_like(cv.lit)
        for gpts in groups:
            pts = _hull(np.concatenate([_circle_pts(p, r * k * 0.86) for p, r in gpts]))
            cv2.fillPoly(tm, [cv._pts(pts).reshape(-1, 1, 2)], 255, lineType=cv2.LINE_AA, shift=4)
        sel = (tm > 0) & (tmp > cv.lit)
        cv.lit[sel] = np.minimum(tmp[sel], np.maximum(cv.lit[sel], (tmp[sel].astype(np.float32) * (tm[sel] / 255.0)).astype(np.uint8)))
    parts.append((dz_t, torso_fn))
    # --- pescoco e cabeca
    capsule('neck', 'head', 0.046, 0.050, style['skin'], depth=Z['neck'] + 0.002)
    hc, dhd = pt3(J['head'] + J['head_dir'] * 0.012)
    fdir3 = J['_R'] @ np.array([0, 0, 1.0])
    hair_c, _ = pt3(J['head'] + J['head_dir'] * 0.034 - fdir3 * 0.020)
    up2, _ = pt3(J['head'] + J['head_dir'] * 0.1)
    ang = np.degrees(np.arctan2(up2[1] - hc[1], up2[0] - hc[0])) + 90.0
    ear_l, _ = pt3(J['head'] - J['_xh'] * 0.078 - J['head_dir'] * 0.012)
    ear_r, _ = pt3(J['head'] + J['_xh'] * 0.078 - J['head_dir'] * 0.012)
    rng_h = np.random.default_rng(11)
    curls = [(rng_h.uniform(0, 2 * np.pi), rng_h.uniform(0.016, 0.028)) for _ in range(22)]

    def head_fn():
        ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        Rr = np.array([[ca, -sa], [sa, ca]])
        th = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        e = np.stack([0.080 * k * np.cos(th), 0.106 * k * np.sin(th)], 1) @ Rr.T
        cv.poly(e + hc, style['skin'], lit_lo)
        cv.lit_poly(e * 0.84 + hc + side * off_amt * 0.09 * k, lit_hi)
        for ep in (ear_l, ear_r):
            cv.poly(_circle_pts(ep, 0.020 * k), style['skin'], lit_lo)
        # cabelo: volume principal + cachos na borda superior
        e2 = np.stack([0.088 * k * np.cos(th), 0.092 * k * np.sin(th)], 1) @ Rr.T
        cv.poly(e2 + hair_c, style['hair'], lit_lo * 0.6)
        for (aa_, rr_) in curls:
            if np.sin(aa_) > 0.35:
                continue          # so a metade de cima
            pc = np.array([0.090 * k * np.cos(aa_), 0.094 * k * np.sin(aa_)]) @ Rr.T + hair_c
            cv.poly(_circle_pts(pc, rr_ * k, 10), style['hair'], lit_lo * 0.6)
    parts.append((dhd, head_fn))
    # --- bracos: deltoide + manga + pele
    for hnd in (0, 1):
        S, E, Wr, Hd = 'sh%d' % hnd, 'el%d' % hnd, 'wr%d' % hnd, 'hand%d' % hnd
        capsule(S, E, 0.043, 0.037, style['skin'])
        mid = P2[S] * 0.46 + P2[E] * 0.54
        capsule(S, mid, 0.046, 0.045, style['shirt'], depth=0.5 * (Z[S] + Z[E]) - 0.004)
        capsule(E, Wr, 0.038, 0.027, style['skin'])
        hv = P2[Hd] - P2[Wr]
        tip = P2[Hd] + hv * 0.75
        capsule(Wr, Hd, 0.029, 0.036, style['skin'])
        capsule(Hd, tip, 0.036, 0.024, style['skin'], depth=Z[Hd])
    # --- mochila
    if bag:
        back = -(J['_R'] @ np.array([0, 0, 1.0]))
        xr = J['_xh']
        pts_b = []
        for (du, dx_) in ((0.07, -0.105), (0.07, 0.105), (-0.20, -0.115), (-0.20, 0.115)):
            c_, d1 = pt3(ch + back * 0.150 + sd * du + xr * dx_)
            pts_b.append((c_, 0.060))
        blob(pts_b, style['bag'], d1 - 0.08)
        # alcas
        for sgn in (-1.0, 1.0):
            a1, da = pt3(nk - sd * 0.06 + xr * sgn * 0.10 + back * 0.05)
            a2, _ = pt3(ch + back * 0.10 + sd * 0.05 + xr * sgn * 0.10)
            capsule(a1, a2, 0.016, 0.016, style['bag'] * 0.7, depth=da - 0.07)
    for d, fn in sorted(parts, key=lambda e: -e[0]):
        fn()
    alb, lit, a = cv.result()
    lit = cv2.GaussianBlur(lit, (0, 0), max(0.6, 0.012 * k))
    sunvis = float(smooth(-0.8, 1.0, L['el'])) * sun_scale
    sunCol = L['sunCol'] / np.pi
    amb = L['skyAvg']
    dark = 1.0 - 0.72 * silhouette
    rgb = alb * (sunCol[None, None] * (lit[..., None] * sunvis) + amb[None, None] * 0.85) * dark
    # luz de contorno (contraluz): borda voltada para o sol
    back = max(-lc[2], 0.0)
    if rim > 0 and (back > 0.05 or sn > 0.3) and sunvis > 0.01:
        rw = max(0.8, 0.0085 * k)
        sh = side * rw * 1.5
        M = np.float32([[1, 0, -sh[0]], [0, 1, -sh[1]]])
        a_sh = cv2.warpAffine(a, M, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR)
        edge = np.clip(a - a_sh, 0, 1)
        # contorno geral fino (luz envolvente)
        er = cv2.erode(a, np.ones((3, 3), np.uint8), iterations=max(1, int(rw)))
        halo = np.clip(a - er, 0, 1) * 0.22
        rimm = cv2.GaussianBlur(np.maximum(edge, halo * back), (0, 0), max(0.5, rw * 0.45))
        rk = rim * (0.42 * back + 0.30 * sn) * sunvis
        rgb += rimm[..., None] * (sunCol[None, None] * rk) * (0.35 + 0.65 * np.sqrt(alb))
    if tint is not None:
        rgb *= np.asarray(tint, np.float32)
    aa = a * alpha
    over(img, rgb * aa[..., None], aa, x0, y0)
    if return_mask:
        return (x0, y0, aa)
    return (x0, y0, w, h)


# ---------------- bolas ----------------
_ball_cache = {}


def ball_sprite(lc, sunCol, amb, sunvis, n=48, rimk=0.0, glow=0.0, albedo=0.90):
    """esfera branca sombreada. lc = direcao da luz em espaco de camera (x dir, y cima, z para a camera)."""
    key = (tuple(np.round(lc, 2)), tuple(np.round(sunCol, 3)), tuple(np.round(amb, 3)), round(sunvis, 2), n,
           round(rimk, 2), round(glow, 3))
    if key in _ball_cache:
        return _ball_cache[key]
    if len(_ball_cache) > 400:
        _ball_cache.clear()
    yy, xx = np.mgrid[-1:1:n * 1j, -1:1:n * 1j].astype(np.float32)
    r2 = xx * xx + yy * yy
    a = np.clip((1.0 - np.sqrt(r2)) * n * 0.5, 0, 1)
    nz = np.sqrt(np.clip(1 - r2, 0, 1))
    lam = np.clip(xx * lc[0] - yy * lc[1] + nz * lc[2], 0, 1)
    wrap = np.clip((xx * lc[0] - yy * lc[1] + nz * lc[2]) * 0.5 + 0.5, 0, 1) ** 2
    up = 0.80 + 0.45 * (-yy)
    col = albedo * ((sunCol / np.pi)[None, None] * ((0.8 * lam + 0.2 * wrap) * sunvis)[..., None] + amb[None, None] * up[..., None])
    if rimk > 0:
        fres = (1 - nz) ** 2.2
        back = max(-lc[2], 0.0)
        col += (sunCol / np.pi)[None, None] * (fres * rimk * back * sunvis * (0.5 + 0.5 * np.clip(xx * lc[0] - yy * lc[1], -1, 1)))[..., None]
    if glow > 0:
        col += glow * (amb[None, None] * 1.2 + (sunCol / np.pi)[None, None] * (0.25 * sunvis))
    out = (col.astype(np.float32), a.astype(np.float32))
    _ball_cache[key] = out
    return out


def draw_ball(img, cam, pts, radius, L, sun_scale=1.0, rimk=1.2, glow=0.0, alpha=1.0, lit_override=None, halo=0.0,
              boost=1.0):
    """pts: lista de posicoes 3D (subamostras de movimento). Desenha com desfoque de movimento."""
    pts = np.atleast_2d(np.asarray(pts, float))
    x, y, z = cam.project(pts)
    if np.any(z < 0.2):
        return
    r = radius * cam.f / z
    s = L['sun']
    lc = np.array([s @ cam.right, s @ cam.up, -(s @ cam.fwd)])
    sunvis = float(smooth(-0.8, 1.0, L['el'])) * sun_scale
    if lit_override is not None:
        sunvis = lit_override
    n = len(pts)
    rm = float(np.mean(r))
    if halo > 0 and sunvis > 0.02:
        # halo luminoso (bola acesa pelo sol): disco aditivo suave
        hc = (L['sunCol'] / np.pi) * (halo * sunvis * 0.55)
        for i in range(0, n, max(1, n // 3)):
            splat_disc(img, x[i], y[i], r[i] * 3.4, hc, alpha=1.0 / len(range(0, n, max(1, n // 3))), soft=r[i] * 3.0, additive=True)
    if boost != 1.0:
        L = dict(L)
        L['sunCol'] = L['sunCol'] * np.float32(boost)
    if rm < 2.2:
        # bola minuscula: discos simples
        spr, _ = ball_sprite(lc, L['sunCol'], L['skyAvg'], sunvis, 24, rimk, glow)
        col = spr[8:16, 8:16].mean((0, 1))
        for i in range(n):
            splat_disc(img, x[i], y[i], max(r[i], 0.7), col, alpha=alpha / n * min(1.0, (r[i] / 0.7) ** 2) if r[i] < 0.7 else alpha / n,
                       soft=1.0, additive=False)
        return
    ns = int(np.clip(2 * rm + 2, 8, 220))
    spr, a = ball_sprite(lc, L['sunCol'], L['skyAvg'], sunvis, 96 if rm > 20 else 48, rimk, glow)
    spr_s = cv2.resize(spr, (ns, ns), interpolation=cv2.INTER_AREA)
    a_s = cv2.resize(a, (ns, ns), interpolation=cv2.INTER_AREA)
    if n == 1:
        xi, yi = int(np.floor(x[0] - ns / 2)), int(np.floor(y[0] - ns / 2))
        fx, fy = x[0] - ns / 2 - xi, y[0] - ns / 2 - yi
        M = np.float32([[1, 0, fx], [0, 1, fy]])
        sp = cv2.warpAffine(spr_s * a_s[..., None], M, (ns + 1, ns + 1))
        aa = cv2.warpAffine(a_s, M, (ns + 1, ns + 1))
        over(img, sp * alpha, aa * alpha, xi, yi)
        return
    bx0, by0 = int(np.floor(x.min() - ns / 2 - 1)), int(np.floor(y.min() - ns / 2 - 1))
    bx1, by1 = int(np.ceil(x.max() + ns / 2 + 2)), int(np.ceil(y.max() + ns / 2 + 2))
    acc = np.zeros((by1 - by0, bx1 - bx0, 4), np.float32)
    pre = np.dstack([spr_s * a_s[..., None], a_s])
    for i in range(n):
        px, py = x[i] - ns / 2 - bx0, y[i] - ns / 2 - by0
        xi, yi = int(np.floor(px)), int(np.floor(py))
        M = np.float32([[1, 0, px - xi], [0, 1, py - yi]])
        sp = cv2.warpAffine(pre, M, (ns + 1, ns + 1))
        acc[yi:yi + ns + 1, xi:xi + ns + 1] += sp[:acc.shape[0] - yi, :acc.shape[1] - xi]
    acc *= alpha / n
    over(img, acc[..., :3], acc[..., 3], bx0, by0)
