# tree.py — ipe-amarelo procedural: cresce com a idade, folhas, florada, vento, petalas
import numpy as np
import cv2
from core import CFG, f32, smooth, over, lerp, splat_disc
from figure import Canvas

BARK = np.array([0.054, 0.044, 0.037])
LEAF = np.array([0.060, 0.140, 0.028])
LEAF_YOUNG = np.array([0.150, 0.250, 0.040])
LEAF_OLD = np.array([0.330, 0.200, 0.035])
BLOOM = np.array([1.00, 0.70, 0.020])
BLOOM_SH = np.array([0.74, 0.40, 0.008])

LEVEL_AGE = [0.0, 0.0, 0.07, 0.20, 0.38, 0.56, 0.72]


class Ipe:
    def __init__(self, base, seed=7, max_level=6):
        self.base = np.asarray(base, float)
        rng = np.random.default_rng(seed)
        self.rng = rng
        nodes = []     # dict(parent, dir, len, r, level)

        def add(parent, d, ln, r, level):
            nodes.append(dict(parent=parent, dir=d / np.linalg.norm(d), len=ln, r=r, level=level, kids=[]))
            if parent >= 0:
                nodes[parent]['kids'].append(len(nodes) - 1)
            return len(nodes) - 1

        def rot(d, ang, az):
            # gira d por 'ang' em torno de um eixo perpendicular com azimute 'az'
            a = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
            u = np.cross(d, a); u /= np.linalg.norm(u)
            v = np.cross(d, u)
            axis = u * np.cos(az) + v * np.sin(az)
            return d * np.cos(ang) + np.cross(axis, d) * np.sin(ang) + axis * (axis @ d) * (1 - np.cos(ang))

        def grow(parent, d, ln, r, level):
            # ramo tortuoso em 2 subsegmentos
            d1 = rot(d, rng.uniform(0.05, 0.22), rng.uniform(0, 6.28))
            i1 = add(parent, d1, ln * 0.52, r, level)
            d2 = rot(d, rng.uniform(0.05, 0.25), rng.uniform(0, 6.28))
            d2[1] += 0.12
            i2 = add(i1, d2, ln * 0.48, r * 0.88, level)
            if level >= max_level:
                return
            nk = 3 if (level <= 1 or rng.random() < 0.28) else 2
            az0 = rng.uniform(0, 6.28)
            for k in range(nk):
                ang = rng.uniform(0.42, 0.85) if level >= 1 else rng.uniform(0.50, 0.80)
                dk = rot(d2 / np.linalg.norm(d2), ang, az0 + k * 6.283 / nk + rng.normal(0, 0.35))
                dk[1] += 0.20 + 0.10 * rng.random()
                dk /= np.linalg.norm(dk)
                grow(i2, dk, ln * rng.uniform(0.70, 0.86), r * 0.88 * (1.0 / nk) ** 0.42, level + 1)

        grow(-1, np.array([0.06, 1.0, 0.03]), 2.05, 0.105, 0)
        self.nodes = nodes
        n = len(nodes)
        self.parent = np.array([nd['parent'] for nd in nodes])
        self.dir = np.array([nd['dir'] for nd in nodes])
        self.len = np.array([nd['len'] for nd in nodes])
        self.r = np.array([nd['r'] for nd in nodes])
        self.level = np.array([nd['level'] for nd in nodes])
        self.has_kids = np.array([len(nd['kids']) > 0 for nd in nodes])
        self.kid_level = np.array([nodes[nd['kids'][0]]['level'] if nd['kids'] else 99 for nd in nodes])
        # cachos de folhas/flores por no terminal (ou terminal temporario)
        K = 26
        self.K = K
        off = rng.normal(0, 1.0, (n, K, 3))
        off /= np.maximum(np.linalg.norm(off, axis=2, keepdims=True), 1e-6)
        self.c_off = off * (rng.random((n, K, 1)) ** 0.45) * np.array([1.0, 0.80, 1.0])
        self.c_rad = rng.uniform(0.036, 0.074, (n, K))
        self.c_rnd = rng.random((n, K, 3))
        self.ph = rng.random(n) * 6.283

    def state(self, age):
        """posicoes dos nos para a idade (0..1). retorna p0, p1, raio, g (crescimento do segmento)."""
        s = lerp(0.135, 1.0, age ** 0.85)
        n = len(self.nodes)
        g = smooth(0.0, 0.16, age - np.array([LEVEL_AGE[min(l, 6)] for l in self.level]) + 1e-9)
        g = np.where(self.level <= 1, 1.0, g)
        p0 = np.zeros((n, 3)); p1 = np.zeros((n, 3))
        ln = self.len * s * g
        for i in range(n):
            p = self.parent[i]
            p0[i] = self.base if p < 0 else p1[p]
            p1[i] = p0[i] + self.dir[i] * ln[i]
        r = self.r * (0.22 + 0.78 * s) * np.sqrt(np.maximum(g, 0.02)) * (0.6 + 0.4 * age)
        return p0, p1, r, g, s

    def crown(self, age):
        p0, p1, r, g, s = self.state(age)
        top = p1[:, 1].max()
        c = np.array([p1[:, 0].mean(), 0.55 * top + 0.45 * self.base[1], p1[:, 2].mean()])
        rad = max(np.abs(p1 - c).max(), 0.3)
        return c, rad, top

    def width_at(self, age):
        """largura da copa (m) para a sombra, em funcao da altura."""
        c, rad, top = self.crown(age)
        h0 = self.base[1]
        H = top - h0

        def w(h):
            u = h / max(H, 1e-3)
            if u < 0 or u > 1.02:
                return 0.0
            trunk = 0.16 + 0.1 * age
            cr = 2.0 * rad * np.sqrt(max(1 - ((u - 0.66) / 0.40) ** 2, 0.0))
            return max(trunk if u < 0.4 else 0.0, cr)
        return w, H

    def draw(self, img, cam, L, age=1.0, leaves=1.0, bloom=0.0, t=0.0, wind=1.0, leaf_age=0.5, silhouette=0.0,
             rim=1.0, density=1.0, sun_h=None):
        p0, p1, r, g, s = self.state(age)
        cc, crad, top = self.crown(age)
        H = max(top - self.base[1], 0.2)
        # vento: desloca nos conforme a altura
        def sway(p, ph):
            hr = np.clip((p[:, 1] - self.base[1]) / H, 0, 1) ** 1.7
            w = (np.sin(t * 1.15 + ph * 0.15 + p[:, 0] * 0.4) * 0.6 + np.sin(t * 2.6 + ph) * 0.25 + np.sin(t * 0.47) * 0.5)
            d = np.zeros_like(p)
            d[:, 0] = 0.060 * H * hr * w * wind
            d[:, 2] = 0.025 * H * hr * np.sin(t * 1.7 + ph * 1.3) * wind
            return p + d
        q1 = sway(p1, self.ph)
        q0 = np.where((self.parent >= 0)[:, None], q1[np.maximum(self.parent, 0)], p0)   # juntas sempre ligadas
        allp = np.concatenate([q0, q1])
        x, y, z = cam.project(allp)
        if z.min() < 0.4:
            return None
        n = len(q0)
        k = cam.f / float(np.mean(z))
        pad = 0.5 * k * (0.4 + 0.6 * s) + 6
        x0, y0 = int(np.floor(x.min() - pad)), int(np.floor(y.min() - pad))
        x1, y1 = int(np.ceil(x.max() + pad)), int(np.ceil(y.max() + pad))
        x0, y0 = max(x0, -int(0.2 * cam.W)), max(y0, -int(0.3 * cam.H))
        x1, y1 = min(x1, int(1.2 * cam.W)), min(y1, int(1.1 * cam.H))
        w, h = x1 - x0, y1 - y0
        if w < 2 or h < 2:
            return None
        ss = 2 if k * s > 60 else 3
        cv = Canvas(x0, y0, w, h, ss)
        sun = L['sun']
        lc = np.array([sun @ cam.right, sun @ cam.up, -(sun @ cam.fwd)])
        front = max(lc[2], 0.0)
        back = max(-lc[2], 0.0)
        sunvis = float(smooth(-0.8, 1.0, L['el']))
        # ---- cachos
        term_w = g * np.where(self.has_kids, 1.0 - smooth(0.0, 0.16, age - np.array([LEVEL_AGE[min(l, 6)] for l in self.kid_level]) + 1e-9), 1.0)
        term_w = np.where((self.level >= 1) & (np.arange(n) % 2 == 1), term_w, 0.0)   # so pontas de ramo (2o subsegmento)
        idx = np.nonzero(term_w > 0.25)[0]
        discs = []
        if len(idx) and (leaves > 0.01 or bloom > 0.01):
            cR = (0.17 + 0.20 * s) * (1.0 + 0.75 * leaves * (1 - bloom))
            tw_ = term_w[idx][:, None, None]
            centers = q1[idx][:, None, :] + self.c_off[idx] * cR * tw_
            # folhas tambem ao longo do ramo (copa mais cheia)
            along = (self.c_rnd[idx][..., 2:3] ** 1.5) * leaves * (1 - bloom) * 0.55
            centers = centers - (q1[idx] - q0[idx])[:, None, :] * along
            flut = 0.010 * wind * np.sin(t * 5.0 + self.c_rnd[idx][..., 0] * 40.0)
            centers[..., 0] += flut
            rad = self.c_rad[idx] * (0.60 + 0.40 * s) * (0.7 + 0.3 * term_w[idx][:, None])
            rnd = self.c_rnd[idx]
            C = centers.reshape(-1, 3); Rd = rad.reshape(-1); Rn = rnd.reshape(-1, 3)
            ncl = self.c_off[idx].reshape(-1, 3)
            depth_in = np.linalg.norm(ncl, axis=1)
            ncl = ncl / np.maximum(depth_in[:, None], 1e-6)
            ncr = np.repeat(q1[idx], self.K, axis=0) - cc[None, :]
            ncr /= (np.linalg.norm(ncr, axis=1, keepdims=True) + 1e-6)
            nrm = 0.62 * ncl + 0.38 * ncr
            nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-6)
            cx, cy, cz = cam.project(C)
            ncam = np.stack([nrm @ cam.right, nrm @ cam.up, -(nrm @ cam.fwd)], 1)
            lam = np.clip(ncam @ lc, 0, 1)
            crown_r = np.linalg.norm(np.repeat(q1[idx], self.K, axis=0) - cc[None, :], axis=1) / crad
            ao = np.clip(0.35 + 0.65 * depth_in, 0, 1) * np.clip(0.55 + 0.45 * crown_r, 0, 1)
            lvb = ao * (0.16 + 0.58 * lam + 0.26 * (0.5 + 0.5 * nrm[:, 1]))
            if sun_h is not None:
                lvb = lvb * smooth(sun_h[0], sun_h[1], C[:, 1])
            is_bloom = Rn[:, 1] < bloom * density
            is_leaf = (~is_bloom) & (Rn[:, 0] < leaves * density)
            vis = (is_bloom | is_leaf) & (cz > 0.4)
            base_leaf = lerp(LEAF_YOUNG, LEAF, smooth(0.0, 0.5, leaf_age)) if leaf_age < 0.5 else lerp(LEAF, LEAF_OLD, smooth(0.5, 1.0, leaf_age))
            order = np.argsort(-cz)
            for j in order:
                if not vis[j]:
                    continue
                if is_bloom[j]:
                    lv = lvb[j] + back * 0.60 * (0.55 + 0.45 * Rn[j, 2])
                    col = lerp(BLOOM_SH, BLOOM, np.clip(0.25 + 0.75 * Rn[j, 2], 0, 1))
                    rr = Rd[j] * 0.92
                else:
                    lv = lvb[j] + back * 0.42 * (0.5 + 0.5 * Rn[j, 2])
                    col = base_leaf * (0.70 + 0.60 * Rn[j, 2])
                    rr = Rd[j] * 0.95
                discs.append((cz[j], cx[j], cy[j], rr * cam.f / cz[j], col, 255 * np.clip(lv, 0.03, 1.0)))
        zmid = float(np.median([d[0] for d in discs])) if discs else 0.0

        def draw_discs(sel):
            for (dz_, dx_, dy_, rpx, col, lv) in sel:
                c8 = tuple(int(v) for v in np.clip(np.sqrt(col) * 255, 0, 255))
                ctr = (int(round((dx_ - x0) * ss * 16)), int(round((dy_ - y0) * ss * 16)))
                rad = int(max(rpx * ss, 0.6) * 16)
                cv2.circle(cv.alb, ctr, rad, c8, -1, cv2.LINE_AA, shift=4)
                cv2.circle(cv.lit, ctr, rad, int(lv), -1, cv2.LINE_AA, shift=4)
                cv2.circle(cv.a, ctr, rad, 255, -1, cv2.LINE_AA, shift=4)

        draw_discs([d for d in discs if d[0] >= zmid])
        # ---- ramos
        X0, Y0, Z0 = x[:n], y[:n], z[:n]
        X1, Y1, Z1 = x[n:], y[n:], z[n:]
        c8 = tuple(int(v) for v in np.clip(np.sqrt(BARK) * 255, 0, 255))
        blit = int(255 * np.clip(0.10 + 0.55 * front + 0.25 * abs(lc[0]), 0, 1) * (0.0 if sun_h is not None else 1.0))
        order = np.argsort(-(Z0 + Z1))
        TW = np.array([1.0, 1.0, 1.0, 0.92, 0.78, 0.62, 0.48])      # galhos finos nas pontas
        for i in order:
            if g[i] < 0.03:
                continue
            lv_ = min(int(self.level[i]), 6)
            first = (i % 2 == 0)                                      # 1o subsegmento do ramo
            ra = r[i] * TW[lv_] * (1.10 if first else 1.0)
            rb = r[i] * TW[lv_] * (0.92 if first else 0.80)
            wa = 2 * ra * cam.f / max(Z0[i], 0.4) * ss
            wb = 2 * rb * cam.f / max(Z1[i], 0.4) * ss
            ax_, ay_ = (X0[i] - x0) * ss, (Y0[i] - y0) * ss
            bx_, by_ = (X1[i] - x0) * ss, (Y1[i] - y0) * ss
            pa = (int(round(ax_ * 16)), int(round(ay_ * 16)))
            pb = (int(round(bx_ * 16)), int(round(by_ * 16)))
            if max(wa, wb) < 2.2:
                th = max(1, int(round(0.5 * (wa + wb))))
                cv2.line(cv.alb, pa, pb, c8, th, cv2.LINE_AA, shift=4)
                cv2.line(cv.lit, pa, pb, blit, th, cv2.LINE_AA, shift=4)
                if 0.5 * (wa + wb) < 1.0:
                    tmp = np.zeros_like(cv.a)
                    cv2.line(tmp, pa, pb, int(255 * max(0.5 * (wa + wb), 0.35)), 1, cv2.LINE_AA, shift=4)
                    np.maximum(cv.a, tmp, out=cv.a)
                else:
                    cv2.line(cv.a, pa, pb, 255, th, cv2.LINE_AA, shift=4)
                continue
            # segmento afunilado: quadrilatero + juntas redondas
            dx_, dy_ = bx_ - ax_, by_ - ay_
            ln_ = max((dx_ * dx_ + dy_ * dy_) ** 0.5, 1e-6)
            nx_, ny_ = -dy_ / ln_, dx_ / ln_
            quad = np.array([[ax_ + nx_ * wa / 2, ay_ + ny_ * wa / 2], [bx_ + nx_ * wb / 2, by_ + ny_ * wb / 2],
                             [bx_ - nx_ * wb / 2, by_ - ny_ * wb / 2], [ax_ - nx_ * wa / 2, ay_ - ny_ * wa / 2]])
            q16 = np.round(quad * 16).astype(np.int32)
            for lay, colv in ((cv.alb, c8), (cv.lit, blit), (cv.a, 255)):
                cv2.fillConvexPoly(lay, q16, colv, cv2.LINE_AA, shift=4)
                cv2.circle(lay, pa, int(wa * 8), colv, -1, cv2.LINE_AA, shift=4)
                cv2.circle(lay, pb, int(wb * 8), colv, -1, cv2.LINE_AA, shift=4)
        draw_discs([d for d in discs if d[0] < zmid])
        alb, lit, a = cv.result()
        sunCol = L['sunCol'] / np.pi
        amb = L['skyAvg']
        dark = 1.0 - 0.75 * silhouette
        rgb = alb * (sunCol[None, None] * (lit[..., None] * sunvis) + amb[None, None] * 0.80) * dark
        over(img, rgb * a[..., None], a, x0, y0)
        return (x0, y0, a)

    def tip_points(self, age, n=400, seed=3):
        """pontos de emissao de petalas (extremidades)."""
        p0, p1, r, g, s = self.state(age)
        idx = np.nonzero((self.level >= 3) & (g > 0.5))[0]
        if len(idx) == 0:
            idx = np.arange(len(p1))
        rr = np.random.default_rng(seed)
        sel = rr.choice(idx, n)
        return p1[sel] + rr.normal(0, 0.22, (n, 3))


class Petals:
    """petalas/folhas caindo e tapete no chao."""

    def __init__(self, tree, age, ground, n=420, seed=5, period=9.0, carpet=900):
        r = np.random.default_rng(seed)
        self.src = tree.tip_points(age, n, seed)
        self.t0 = r.uniform(0, period, n)
        self.period = period
        self.ph = r.random((n, 3)) * 6.283
        self.sp = r.uniform(0.55, 1.05, n)
        self.ground = ground
        base = tree.base
        ang = r.uniform(0, 6.283, carpet)
        rad = np.abs(r.normal(0, 1.9, carpet)) + 0.1
        self.carpet = np.stack([base[0] + rad * np.cos(ang), np.zeros(carpet), base[2] + rad * np.sin(ang)], 1)
        self.carpet[:, 1] = ground(self.carpet[:, 0], self.carpet[:, 2]) + 0.02
        self.c_rnd = r.random(carpet)

    def positions(self, t, wind=(0.35, 0.0, -0.12), slow=1.0):
        age = np.mod(t * slow - self.t0, self.period)
        p = self.src.copy()
        w = np.asarray(wind)
        p[:, 1] -= self.sp * age * 0.85
        p[:, 0] += w[0] * age + 0.16 * np.sin(self.ph[:, 0] + age * 3.1)
        p[:, 2] += w[2] * age + 0.16 * np.sin(self.ph[:, 1] + age * 2.6)
        gy = self.ground(p[:, 0], p[:, 2]) + 0.02
        landed = p[:, 1] < gy
        p[:, 1] = np.maximum(p[:, 1], gy)
        fade = np.clip((self.period - age) / 1.0, 0, 1) * np.clip(age / 0.25, 0, 1)
        return p, fade, landed

    def draw(self, img, cam, L, t, amount=1.0, zmin=0.0, zmax=1e9, slow=1.0, col=BLOOM):
        p, fade, landed = self.positions(t, slow=slow)
        x, y, z = cam.project(p)
        sunvis = float(smooth(-0.8, 1.0, L['el']))
        c = col * (L['sunCol'] / np.pi * sunvis * 0.85 + L['skyAvg'] * 0.9)
        m = int(len(p) * min(amount, 1.0))
        H, W = img.shape[:2]
        for i in range(m):
            if not (zmin <= z[i] < zmax) or z[i] < 0.5 or not (0 <= x[i] < W and 0 <= y[i] < H):
                continue
            r = max(0.016 * cam.f / z[i], 0.6 * CFG.S)
            tw = 0.55 + 0.45 * np.sin(self.ph[i, 2] + t * 9.0 * slow) if not landed[i] else 0.8
            splat_disc(img, x[i], y[i], r, c * tw, alpha=fade[i] * 0.95, soft=1.0)

    def draw_carpet(self, img, cam, L, amount=1.0, zmin=0.0, zmax=1e9, col=BLOOM):
        x, y, z = cam.project(self.carpet)
        sunvis = float(smooth(-0.8, 1.0, L['el']))
        c = col * (L['sunCol'] / np.pi * sunvis * 0.7 + L['skyAvg'] * 0.9)
        H, W = img.shape[:2]
        sel = np.nonzero((self.c_rnd < amount) & (z >= zmin) & (z < zmax) & (z > 0.5) & (x >= 0) & (x < W) & (y >= 0) & (y < H))[0]
        for i in sel:
            r = max(0.018 * cam.f / z[i], 0.6 * CFG.S)
            splat_disc(img, x[i], y[i], r, c * (0.7 + 0.5 * self.c_rnd[i]), alpha=0.9, soft=1.0)
