# meadow.py — relva do topo da colina em faixas de profundidade (2.5D), com vento, flores e sombras
import numpy as np
import cv2
from core import CFG, f32, smooth, over, lerp


def mound(x, z):
    """altura do terreno do morro (topo em 0,0)."""
    x = np.asarray(x, float)
    z = np.asarray(z, float)
    return -(x * x / 140.0 + z * z / 320.0) + 0.035 * np.sin(x * 0.83 + 1.3) * np.sin(z * 0.61 + 0.4) \
        + 0.02 * np.sin(x * 2.1 + z * 1.7)


GRASS_SUMMER = (np.array([0.030, 0.062, 0.018]), np.array([0.105, 0.170, 0.045]))
GRASS_DRY = (np.array([0.085, 0.070, 0.032]), np.array([0.30, 0.235, 0.105]))
TRANS_SUMMER = np.array([0.36, 0.56, 0.09])
TRANS_DRY = np.array([0.95, 0.70, 0.28])
FLOWER_COLS = [np.array([0.90, 0.62, 0.02]), np.array([0.92, 0.90, 0.84]), np.array([0.42, 0.20, 0.62]),
               np.array([0.95, 0.35, 0.10])]


class Strip:
    pass


class Meadow:
    def __init__(self, cam, ground=mound, seed=1, d_near=1.15, d_far=17.6, ratio=1.21, focus=17.0, dof=0.030,
                 density=1.0, flowers=1.0, stalks=1.0, extra_depths=(), clearing=(0.0, 0.0)):
        self.cam = cam
        self.ground = ground
        self.focus = focus
        rng = np.random.default_rng(seed)
        W, H = cam.W, cam.H
        fh = np.array([cam.fwd[0], 0.0, cam.fwd[2]])
        fh /= np.linalg.norm(fh)
        rh = cam.right
        self.fh, self.rh = fh, rh
        depths = []
        d = d_near
        while d < d_far:
            depths.append(d)
            d *= ratio
        depths.append(d_far)
        depths += list(extra_depths)
        depths = sorted(set(round(v, 3) for v in depths))
        self.d_lo, self.d_hi = depths[0], depths[-1]
        self.strips = []
        M = int(0.04 * W)
        for d in depths:
            hw = d * cam.tanx * 1.10 + 0.4
            rho = float(np.clip(250.0 * (d / 3.0) ** 0.62, 130, 820)) * density
            n = int(rho * 2 * hw)
            s = rng.uniform(-hw, hw, n)
            dj = d + rng.uniform(-0.10, 0.10, n) * d * 0.9
            base = cam.pos[None, :] * np.array([1, 0, 1]) + dj[:, None] * fh[None, :] + s[:, None] * rh[None, :]
            base[:, 1] = ground(base[:, 0], base[:, 2])
            hgt = np.clip(0.21 * np.exp(0.36 * rng.standard_normal(n)), 0.07, 0.44)
            kind = np.zeros(n, int)
            st = rng.random(n) < 0.035 * stalks
            hgt[st] = rng.uniform(0.38, 0.70, st.sum())
            kind[st] = 1
            fl = (rng.random(n) < 0.022 * flowers) & ~st
            hgt[fl] = rng.uniform(0.16, 0.38, fl.sum())
            # clareira pisada onde ele treina
            rc = np.hypot(base[:, 0] - clearing[0], base[:, 2] - clearing[1])
            hgt *= lerp(0.38, 1.0, smooth(1.0, 4.2, rc))
            kind[fl] = 2 + rng.integers(0, 4, fl.sum()) * (rng.random(fl.sum()) < 0.9) * 1
            kind[fl] = np.where(rng.random(fl.sum()) < 0.62, 2, np.where(rng.random(fl.sum()) < 0.55, 3, 4))
            th = rng.normal(0, 0.26, n) + 0.10
            lean = hgt * np.sin(th)
            us = np.array([0.0, 0.36, 0.70, 1.0])
            pts = base[:, None, :] + np.array([0, 1.0, 0])[None, None, :] * (hgt[:, None, None] * (us - 0.12 * us ** 2)[None, :, None]) \
                + rh[None, None, :] * (lean[:, None, None] * (us ** 1.8)[None, :, None]) \
                + fh[None, None, :] * (rng.normal(0, 0.05, n)[:, None, None] * hgt[:, None, None] * (us ** 1.8)[None, :, None])
            px, py, pz = cam.project(pts.reshape(-1, 3))
            px, py = px.reshape(n, 4), py.reshape(n, 4)
            wpx = 0.0068 * cam.f / d * rng.uniform(0.7, 1.4, n)
            vv = np.clip(rng.normal(0.48, 0.20, n) + 0.25 * (kind == 1), 0.02, 1.0)
            inside = (px[:, 0] > -M * 2) & (px[:, 0] < W + M * 2)
            px, py, wpx, vv, kind, hgt = px[inside], py[inside], wpx[inside], vv[inside], kind[inside], hgt[inside]
            if len(px) == 0:
                continue
            x0, x1 = -M, W + M
            y0 = int(np.floor(max(py.min() - 3, -M)))
            y1 = int(np.ceil(min(py.max() + 3, H + M)))
            if y1 - y0 < 2 or y0 > H:
                continue
            w, h = x1 - x0, y1 - y0
            coc = abs(1.0 / d - 1.0 / focus) * cam.f * dof
            div = 1
            if coc > 10:
                div = 4
            elif coc > 4:
                div = 2
            A = np.zeros((h, w), np.uint8)
            V = np.zeros((h, w), np.uint8)
            FA = np.zeros((h, w), np.uint8)
            FC = np.zeros((h, w, 3), np.uint8)
            sh = 16
            ox, oy = x0, y0
            wmean = float(np.mean(wpx))
            if wmean >= 2.0:
                wt = np.array([1.0, 0.82, 0.52, 0.10])
                # normal 2D de cada segmento ~ horizontal: desloca em x
                polys = []
                for i in range(len(px)):
                    hwid = wpx[i] * 0.5 * wt
                    Lp = np.stack([px[i] - hwid - ox, py[i] - oy], 1)
                    Rp = np.stack([px[i] + hwid - ox, py[i] - oy], 1)[::-1]
                    polys.append(np.round(np.concatenate([Lp, Rp]) * sh).astype(np.int32).reshape(-1, 1, 2))
                cv2.fillPoly(A, polys, 255, lineType=cv2.LINE_AA, shift=4)
                q = np.clip((vv * 9).astype(int), 0, 9)
                for lv in range(10):
                    sel = [polys[i] for i in np.nonzero(q == lv)[0]]
                    if sel:
                        cv2.fillPoly(V, sel, int(20 + lv * 25), lineType=cv2.LINE_8, shift=4)
                        cv2.polylines(V, sel, True, int(20 + lv * 25), 1, cv2.LINE_8, shift=4)
            else:
                lines = [np.round(np.stack([px[i] - ox, py[i] - oy], 1) * sh).astype(np.int32).reshape(-1, 1, 2) for i in range(len(px))]
                thick = max(1, int(round(wmean)))
                cv2.polylines(A, lines, False, 255, thick, cv2.LINE_AA, shift=4)
                if wmean < 1.0:
                    A = (A.astype(np.float32) * max(wmean, 0.55)).astype(np.uint8)
                q = np.clip((vv * 9).astype(int), 0, 9)
                for lv in range(10):
                    sel = [lines[i] for i in np.nonzero(q == lv)[0]]
                    if sel:
                        cv2.polylines(V, sel, False, int(20 + lv * 25), thick + 1, cv2.LINE_8, shift=4)
            # espigas e flores
            for i in np.nonzero(kind >= 1)[0]:
                tx, ty = px[i, 3] - ox, py[i, 3] - oy
                if kind[i] == 1:
                    rr = max(0.011 * cam.f / d, 0.8)
                    ax = (int(max(rr * 0.75, 1) * sh), int(max(rr * 3.2, 1) * sh))
                    ang = np.degrees(np.arctan2(px[i, 3] - px[i, 2], -(py[i, 3] - py[i, 2])))
                    cv2.ellipse(A, (int(tx * sh), int((ty + rr * 1.6) * sh)), ax, ang, 0, 360, 255, -1, cv2.LINE_AA, shift=4)
                    cv2.ellipse(V, (int(tx * sh), int((ty + rr * 1.6) * sh)), ax, ang, 0, 360, 235, -1, cv2.LINE_8, shift=4)
                else:
                    rr = max(rng.uniform(0.016, 0.028) * cam.f / d, 1.0)
                    c = FLOWER_COLS[kind[i] - 2]
                    c8 = tuple(int(v) for v in np.sqrt(c * rng.uniform(0.8, 1.1)).clip(0, 1) * 255)
                    cv2.circle(FA, (int(tx * sh), int(ty * sh)), int(rr * sh), 255, -1, cv2.LINE_AA, shift=4)
                    cv2.circle(FC, (int(tx * sh), int(ty * sh)), int((rr + 0.6) * sh), c8, -1, cv2.LINE_8, shift=4)
            S = Strip()
            S.d, S.x0, S.y0, S.div = d, x0, y0, div
            a = A.astype(np.float32) / 255.0
            v = V.astype(np.float32) / 255.0
            fa = FA.astype(np.float32) / 255.0
            fc = (FC.astype(np.float32) / 255.0) ** 2
            if coc > 0.6:
                va = cv2.GaussianBlur(v * a, (0, 0), coc * 0.5)
                a = cv2.GaussianBlur(a, (0, 0), coc * 0.5)
                v = va / np.maximum(a, 1e-4)
                fcb = cv2.GaussianBlur(fc * fa[..., None], (0, 0), coc * 0.5)
                fa = cv2.GaussianBlur(fa, (0, 0), coc * 0.5)
                fc = fcb / np.maximum(fa[..., None], 1e-4)
            if div > 1:
                a = cv2.resize(a, (w // div, h // div), interpolation=cv2.INTER_AREA)
                v = cv2.resize(v, (w // div, h // div), interpolation=cv2.INTER_AREA)
                fa = cv2.resize(fa, (w // div, h // div), interpolation=cv2.INTER_AREA)
                fc = cv2.resize(fc, (w // div, h // div), interpolation=cv2.INTER_AREA)
            S.a, S.v, S.fa, S.fc = a, v, fa, fc
            S.w, S.h = w, h
            # linha do chao por coluna (tela) para altura relativa
            xs = (np.arange(a.shape[1]) + 0.5) * div + x0
            sl = (xs - W / 2) / cam.f * d
            gp = cam.pos[None, :] * np.array([1, 0, 1]) + d * fh[None, :] + sl[:, None] * rh[None, :]
            gp[:, 1] = ground(gp[:, 0], gp[:, 2])
            _, gy, _ = cam.project(gp)
            S.gy = gy.astype(np.float32)
            S.hmax = 0.50 * cam.f / d
            S.sl = sl.astype(np.float32)
            S.cache_key = None
            S.phase = rng.uniform(0, 6.28)
            S.coc = coc
            self.strips.append(S)

    def _hrel(self, S):
        yy = (np.arange(S.a.shape[0], dtype=np.float32)[:, None] + 0.5) * S.div + S.y0
        return np.clip((S.gy[None, :] - yy) / S.hmax, 0, 1)

    def shade(self, S, L, dry=0.0, shadows=(), cam=None):
        cam = cam or self.cam
        s = L['sun']
        lc = np.array([s @ cam.right, s @ cam.up, -(s @ cam.fwd)])
        sunvis = float(smooth(-0.8, 1.0, L['el']))
        front = max(lc[2], 0.0) ** 0.7
        back = max(-lc[2], 0.0)
        sidel = abs(lc[0])
        sunCol = L['sunCol'] / np.pi * sunvis
        amb = L['skyAvg']
        dk = lerp(GRASS_SUMMER[0], GRASS_DRY[0], dry)
        lt = lerp(GRASS_SUMMER[1], GRASS_DRY[1], dry)
        tr = lerp(TRANS_SUMMER, TRANS_DRY, dry)
        tp = self._hrel(S) ** 1.25
        v = S.v[..., None]
        alb = dk[None, None] + (lt - dk)[None, None] * v
        dirf = (front * (0.30 + 0.70 * tp) + 0.30 * sidel * (0.4 + 0.6 * tp) + 0.10 * max(lc[1], 0))
        trn = back * (0.10 + 0.90 * tp) * (0.55 + 0.45 * S.v)
        # sombras projetadas (arvore, malabarista): mascara 1D ao longo de x
        shm = np.ones(S.a.shape[1], np.float32)
        if sunvis > 0.02 and s[1] > 0.01:
            for (bx, bz, wd, hh, strength) in shadows:
                dirg = -np.array([s[0], s[2]]) / s[1]
                bd = (np.array([bx, bz]) - cam.pos[[0, 2]]) @ self.fh[[0, 2]]
                bl = (np.array([bx, bz]) - cam.pos[[0, 2]]) @ self.rh[[0, 2]]
                kf = dirg @ self.fh[[0, 2]]
                if abs(kf) < 1e-4:
                    continue
                hs = (S.d - bd) / kf
                if hs < -0.05 or hs > hh:
                    continue
                ls = bl + (dirg @ self.rh[[0, 2]]) * hs
                wdd = wd(hs) if callable(wd) else wd
                if wdd <= 0.01:
                    continue
                pen = 0.04 + 0.012 * abs(hs) / max(s[1], 0.05)
                m = smooth(wdd * 0.5 + pen, wdd * 0.5 - pen * 0.3, np.abs(S.sl - ls))
                shm *= (1.0 - strength * m).astype(np.float32)
        shm = shm[None, :, None]
        col = alb * (sunCol[None, None] * dirf[..., None] * shm + amb[None, None] * (0.40 + 0.60 * tp[..., None]))
        col += (tr * sunCol)[None, None] * (trn[..., None] * shm)
        a = S.a
        rgb = col * a[..., None]
        # chao opaco abaixo da linha do solo desta faixa
        yy = (np.arange(S.a.shape[0], dtype=np.float32)[:, None] + 0.5) * S.div + S.y0
        gm = np.clip((yy - S.gy[None, :]) / max(2.0, S.hmax * 0.10) + 0.2, 0, 1) * (1 - a)
        if gm.max() > 0.003:
            gcol = (dk * 0.55 + lt * 0.12)[None, None] * (sunCol[None, None] * (0.18 * front + 0.05) * shm + amb[None, None] * 0.38)
            rgb = rgb + gcol * gm[..., None]
            a = a + gm
        fa = S.fa
        if fa.max() > 0.003:
            fcol = S.fc * (sunCol[None, None] * (0.25 + 0.75 * max(front, 0.55 * back + 0.3 * sidel)) * shm + amb[None, None] * 0.9)
            rgb = rgb * (1 - fa[..., None]) + fcol * fa[..., None]
            a = a + fa - a * fa
        return rgb.astype(np.float32), a.astype(np.float32)

    def ground_plate(self, img, cam, L, dry=0.0):
        """chao opaco do morro: tudo o que fica abaixo da silhueta do terreno (sem frestas entre as faixas)."""
        W, H = cam.W, cam.H
        fh, rh = self.fh, self.rh
        cols = np.arange(-16, W + 17, 8, dtype=np.float64)
        env = np.full(len(cols), 1e9)
        base = cam.pos * np.array([1.0, 0.0, 1.0])
        for d in np.geomspace(0.30, self.d_hi, 80):
            sl = np.linspace(-1.35, 1.35, 96) * (d * cam.tanx + 0.3)
            gp = base[None, :] + d * fh[None, :] + sl[:, None] * rh[None, :]
            gp[:, 1] = self.ground(gp[:, 0], gp[:, 2])
            px, py, pz = cam.project(gp)
            ok = pz > 0.05
            if ok.sum() < 4:
                continue
            env = np.minimum(env, np.interp(cols, px[ok], py[ok], left=1e9, right=1e9))
        if not np.isfinite(env).all() or env.min() > H:
            env = np.where(env > 1e8, H + 50.0, env)
        y0 = int(max(np.floor(env.min()) - 3, 0))
        if y0 >= H:
            return
        envx = np.interp(np.arange(W), cols, env).astype(np.float32)
        ramp = max(2.0, 0.05 * cam.f / self.d_hi)
        yy = (np.arange(y0, H, dtype=np.float32)[:, None] + 0.5)
        a = np.clip((yy - envx[None, :]) / ramp + 0.2, 0.0, 1.0)
        s = L['sun']
        lc2 = -(s @ cam.fwd)
        front = max(lc2, 0.0) ** 0.7
        sunvis = float(smooth(-0.8, 1.0, L['el']))
        sunCol = L['sunCol'] / np.pi * sunvis
        dk = lerp(GRASS_SUMMER[0], GRASS_DRY[0], dry)
        lt = lerp(GRASS_SUMMER[1], GRASS_DRY[1], dry)
        gcol = ((dk * 0.55 + lt * 0.12) * (sunCol * (0.18 * front + 0.05) + L['skyAvg'] * 0.38)).astype(np.float32)
        reg = img[y0:H]
        reg *= (1.0 - a)[..., None]
        reg += a[..., None] * gcol[None, None, :]

    def items(self, cam_now, L, t=0.0, dry=0.0, shadows=(), wind=1.0, light_key=None, d_min=0.0, d_max=1e9):
        """retorna [(profundidade, funcao(img))] para o pintor."""
        out = [(1e5, lambda img: self.ground_plate(img, cam_now, L, dry))]
        cam0 = self.cam
        delta = cam_now.pos - cam0.pos
        dz = float(delta @ self.fh)
        dxy = np.array([delta @ cam0.right, delta @ cam0.up])
        for S in self.strips:
            if not (d_min <= S.d < d_max):
                continue
            dn = S.d - dz
            if dn < 0.35:
                continue

            def fn(img, S=S, dn=dn):
                key = light_key
                if key is None or S.cache_key != key:
                    S.rgb, S.al = self.shade(S, L, dry, shadows, cam0)
                    S.cache_key = key
                rgb, al = S.rgb, S.al
                hpx = S.hmax
                if wind > 0 and hpx > 26 * CFG.S and S.div == 1:
                    # balanco: deslocamento horizontal proporcional a altura relativa
                    hr = self._hrel(S)
                    xs = np.arange(rgb.shape[1], dtype=np.float32)[None, :]
                    ph = xs * (0.011 / CFG.S) + S.phase
                    wv = 0.55 * np.sin(ph - t * 1.9) + 0.30 * np.sin(ph * 2.3 + t * 2.7 + 1.0) + 0.25 * np.sin(ph * 0.31 - t * 0.8)
                    amp = 0.085 * hpx * wind
                    mx = (xs - amp * (hr ** 1.6) * wv).astype(np.float32)
                    my = (np.arange(rgb.shape[0], dtype=np.float32)[:, None] + np.zeros_like(xs)).astype(np.float32)
                    rgb = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    al = cv2.remap(al, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                elif wind > 0:
                    xs = np.arange(rgb.shape[1], dtype=np.float32)
                    sh = 1.0 + 0.10 * wind * (np.sin(xs * (0.016 * S.div / CFG.S) + S.phase - t * 1.6) * 0.6 +
                                              np.sin(xs * (0.041 * S.div / CFG.S) + t * 2.2 + S.phase * 2) * 0.4)
                    rgb = rgb * sh[None, :, None].astype(np.float32)
                sc = S.d / dn
                moved = abs(dz) > 1e-4 or np.abs(dxy).max() > 1e-4
                if S.div > 1 or moved:
                    # transforma (escala em torno do centro optico + translacao pela mudanca de camera)
                    W, H = cam0.W, cam0.H
                    w, h = S.w, S.h
                    tx = W / 2 + (S.x0 - W / 2) * sc - cam0.f * dxy[0] / dn
                    ty = H / 2 + (S.y0 - H / 2) * sc + cam0.f * dxy[1] / dn
                    ow, oh = int(np.ceil(w * sc)), int(np.ceil(h * sc))
                    ix, iy = int(np.floor(tx)), int(np.floor(ty))
                    # recorta ao quadro
                    cx0, cy0 = max(ix, 0), max(iy, 0)
                    cx1, cy1 = min(ix + ow, W), min(iy + oh, H)
                    if cx1 <= cx0 or cy1 <= cy0:
                        return
                    k = sc * S.div
                    Mx = np.float32([[k, 0, tx - cx0 + 0.5 * (k - 1)], [0, k, ty - cy0 + 0.5 * (k - 1)]])
                    r2 = cv2.warpAffine(rgb, Mx, (cx1 - cx0, cy1 - cy0), flags=cv2.INTER_LINEAR)
                    a2 = cv2.warpAffine(al, Mx, (cx1 - cx0, cy1 - cy0), flags=cv2.INTER_LINEAR)
                    over(img, r2, a2, cx0, cy0)
                else:
                    over(img, rgb, al, S.x0, S.y0)
            out.append((dn, fn))
        return out
