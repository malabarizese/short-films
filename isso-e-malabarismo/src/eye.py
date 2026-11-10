# eye.py — close extremo do olho ao nascer do sol (procedural 2D)
import numpy as np
import cv2
import sky as skylib
from core import *
from shots import finish_frame


def _noise(n, beta, seed, lo=2.0):
    return skylib.uniformize(skylib.fbm_nd((n, n), beta, seed, lo=lo))


class EyeShot:
    def __init__(self, name='olho', dur=5.0, seed=3):
        self.name, self.dur = name, dur
        self.n = int(round(dur * FPS))
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # ---------- geometria das palpebras ----------
    def lids(self, xs, blink=0.0):
        """curvas das palpebras (y para baixo, px). xs em px. retorna (yu, yl)."""
        s = (xs - self.cx) / (self.w / 2)
        sa = np.clip(np.abs(s), 0, 1)
        # canto interno (esquerda) mais baixo e arredondado; externo levemente erguido
        tilt = -0.045 * self.w * s
        up = self.hu * (1 - sa ** 2.3) ** 0.78 * (1 + 0.10 * s)
        lo = self.hl * (1 - sa ** 2.6) ** 0.70 * (1 - 0.06 * s)
        yu0 = self.cy + tilt - up
        yl = self.cy + tilt + lo
        close = yl - 0.10 * lo * blink
        yu = yu0 + (close - yu0) * blink
        yl2 = yl - 0.10 * lo * blink
        return yu, yl2

    def setup(self):
        W, H, S = CFG.W, CFG.H, CFG.S
        self.W, self.H = W, H
        self.cx, self.cy = W * 0.50, H * 0.535
        self.w = W * 0.60
        self.hu, self.hl = H * 0.215, H * 0.125
        self.R = 0.205 * self.w                 # raio da iris
        self.ix, self.iy = self.cx - 0.012 * self.w, self.cy - 0.022 * H
        rng = self.rng
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.xx, self.yy = xx, yy
        xs = np.arange(W, dtype=np.float32)
        yu, yl = self.lids(xs, 0.0)
        self.yu0, self.yl0 = yu, yl
        # ---------- pele ----------
        n1 = _noise(1024, 2.2, 5, lo=24)          # poros finos
        n2 = _noise(1024, 3.0, 6, lo=3)           # manchas largas
        tex = cv2.resize(n1, (W, H), interpolation=cv2.INTER_LINEAR)
        blot = cv2.resize(n2, (W, H), interpolation=cv2.INTER_CUBIC)
        # relevo: olho afundado na orbita, arco da sobrancelha em cima, nariz a esquerda
        dxn = (xx - self.cx) / (self.w * 0.5)
        dyn = (yy - self.cy) / (self.w * 0.5)
        socket = np.exp(-(dxn ** 2 / 1.25 + dyn ** 2 / 0.42))
        brow = np.exp(-((dyn + 0.74) / 0.20) ** 2) * (0.75 + 0.25 * np.tanh(dxn * 1.5 + 0.4))
        cheek = np.exp(-((dyn - 0.70) / 0.32) ** 2)
        nose = np.exp(-((dxn + 1.55) / 0.42) ** 2)
        hgt = -0.85 * socket + 0.70 * brow + 0.45 * cheek + 1.0 * nose
        hgt = cv2.GaussianBlur(hgt, (0, 0), 9 * CFG.S)
        # rugas finas: sob o olho e no canto externo
        sgn = (xx - self.cx) / (self.w / 2)
        wr = np.zeros_like(hgt)
        for k, (off, amp) in enumerate(((0.20, 1.0), (0.275, 0.7), (0.345, 0.45))):
            yl_k = self.yl0[None, :] + off * self.hl * 1.9 + 0.02 * H * sgn ** 2
            wr += amp * np.exp(-((yy - yl_k) / (0.0042 * H)) ** 2) * np.clip(1 - np.abs(sgn * 0.95) ** 3, 0, 1)
        for k, ang in enumerate((-0.42, -0.12, 0.20)):
            px0, py0 = self.cx + self.w * 0.53, self.cy - 0.012 * H
            dxr, dyr = xx - px0, yy - py0
            along = dxr * np.cos(ang) + dyr * np.sin(ang)
            perp = -dxr * np.sin(ang) + dyr * np.cos(ang)
            wr += 0.8 * np.exp(-(perp / (0.0045 * H)) ** 2) * smooth(0.0, 0.03 * self.w, along) * (1 - smooth(0.10 * self.w, 0.22 * self.w, along))
        self.wrinkle = wr.astype(np.float32)
        gx = cv2.Sobel(hgt, cv2.CV_32F, 1, 0, ksize=5) * (self.w * 0.0045)
        gy = cv2.Sobel(hgt, cv2.CV_32F, 0, 1, ksize=5) * (self.w * 0.0045)
        mic = cv2.GaussianBlur(tex, (0, 0), 1.2 * CFG.S)
        gx += cv2.Sobel(mic, cv2.CV_32F, 1, 0, ksize=3) * 0.55
        gy += cv2.Sobel(mic, cv2.CV_32F, 0, 1, ksize=3) * 0.55
        self.skin_n = np.dstack([-gx, -gy, np.ones_like(gx)])
        self.skin_n /= np.linalg.norm(self.skin_n, axis=2, keepdims=True)
        self.skin_ao = (0.55 + 0.45 * (1 - socket) ** 0.8).astype(np.float32)
        alb = np.array([0.36, 0.215, 0.140], np.float32)
        albm = alb[None, None] * (0.86 + 0.20 * tex[..., None] + 0.16 * (blot[..., None] - 0.5))
        # olheira/rubor leve
        albm *= (1.0 - 0.18 * np.exp(-((dyn - 0.36) / 0.16) ** 2) * np.exp(-(dxn / 0.9) ** 2))[..., None]
        albm[..., 0] *= 1.0 + 0.10 * cheek
        self.skin_alb = albm.astype(np.float32)
        self.pore = (tex - 0.5).astype(np.float32)
        # dobra da palpebra superior (linha escura acompanhando a borda)
        crease = self.yu0 - 0.118 * H * (1 - np.clip(np.abs((xs - self.cx) / (self.w / 2)), 0, 1) ** 2.2) ** 0.6 - 0.012 * H
        self.crease = crease
        self.eye_fade = np.clip(1 - np.abs((xs - self.cx) / (self.w * 0.52)) ** 8, 0, 1).astype(np.float32)
        # ---------- iris (textura polar) ----------
        n = int(2 * self.R * 1.08) // 2 * 2
        self.ir_n = n
        ay, ax = np.mgrid[-1:1:n * 1j, -1:1:n * 1j].astype(np.float32) * 1.08
        rr = np.sqrt(ax * ax + ay * ay)
        th = np.arctan2(ay, ax)
        self.ir_r, self.ir_th = rr, th
        pn = _noise(1024, 1.5, 21, lo=1)
        pn2 = _noise(1024, 2.4, 22, lo=2)
        # fibras radiais: ruido amostrado em (theta, r) com alta frequencia angular
        u = ((th / (2 * np.pi) + 0.5) * 1024 * 3.0) % 1024
        v = (rr * 90.0) % 1024
        fib = cv2.remap(pn, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        u2 = ((th / (2 * np.pi) + 0.5) * 1024 * 1.0) % 1024
        fib2 = cv2.remap(pn2, u2.astype(np.float32), (rr * 260.0 % 1024).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        wav = cv2.remap(pn2, u2.astype(np.float32), np.full_like(u2, 300.0), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        self.fib = (0.65 * fib + 0.35 * fib2).astype(np.float32)
        col_in = np.array([0.52, 0.27, 0.07], np.float32)      # ambar junto a pupila
        col_mid = np.array([0.20, 0.090, 0.028], np.float32)
        col_out = np.array([0.060, 0.030, 0.014], np.float32)
        coll = 0.40 + 0.07 * (wav - 0.5)                     # colarete ondulado
        k1 = smooth(coll - 0.10, coll + 0.16, rr)[..., None]
        k2 = smooth(0.62, 0.98, rr)[..., None]
        base = col_in[None, None] * (1 - k1) + col_mid[None, None] * k1
        base = base * (1 - k2) + col_out[None, None] * k2
        fibc = (0.50 + 1.05 * self.fib[..., None]) ** 1.5
        ir = base * fibc
        # criptas escuras
        cr = _noise(512, 2.0, 23, lo=6)
        crm = cv2.remap(cr, ((th / (2 * np.pi) + 0.5) * 512 * 2 % 512).astype(np.float32), (rr * 150 % 512).astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        ir *= (1.0 - 0.45 * smooth(0.70, 0.90, crm) * smooth(0.25, 0.45, rr) * (1 - smooth(0.75, 0.9, rr)))[..., None]
        limb = smooth(0.86, 1.0, rr)[..., None]
        ir = ir * (1 - 0.80 * limb)
        self.iris = ir.astype(np.float32)
        self.iris_a = (1 - smooth(0.975, 1.03, rr)).astype(np.float32)
        # ---------- esclera ----------
        vn = _noise(1024, 2.0, 31, lo=2)
        self.vein_tex = vn
        # ---------- cilios ----------
        self._make_lashes()
        # ---------- sobrancelha ----------
        self._make_brow()
        # reflexo do horizonte na cornea
        self._make_reflection()
        return self

    def _lash_layer(self, upper=True):
        W, H = self.W, self.H
        rng = self.rng
        lay = np.zeros((H, W), np.uint8)
        xs_all = np.linspace(self.cx - self.w * 0.47, self.cx + self.w * 0.485, 230 if upper else 90)
        yu, yl = self.lids(xs_all, 0.0)
        for i, x0 in enumerate(xs_all):
            s = (x0 - self.cx) / (self.w / 2)
            if upper:
                y0 = yu[i] + 0.004 * H
                ln = self.R * rng.uniform(0.34, 0.62) * (0.55 + 0.45 * (1 - abs(s) ** 2))
                ang = np.radians(-90 + 50 * s + rng.normal(0, 7) + 9 * np.sin(i * 0.9))       # abre em leque, em tufos
                curl = rng.uniform(0.25, 0.60)
            else:
                y0 = yl[i] - 0.002 * H
                ln = self.R * rng.uniform(0.13, 0.26) * (0.5 + 0.5 * (1 - abs(s) ** 2))
                ang = np.radians(90 - 40 * s + rng.normal(0, 10))
                curl = -rng.uniform(0.3, 0.8)
            x0j = x0 + rng.normal(0, 0.004 * self.w)
            pts = []
            for u in np.linspace(0, 1, 7):
                a = ang - curl * (u ** 1.4) * np.sign(np.cos(ang) + 1e-6) * (1 if upper else 1)
                px = x0j + ln * u * np.cos(ang) + ln * 0.22 * curl * (u ** 2) * np.sign(s + 1e-6) * 0.6
                py = y0 + ln * u * np.sin(ang) + (ln * 0.34 * curl * (u ** 2.2)) * (1 if upper else 1)
                pts.append((px, py))
            pts = np.array(pts)
            th0 = max(1.0, (5.0 if upper else 2.6) * CFG.S * rng.uniform(0.7, 1.3))
            for k in range(6):
                t = max(1, int(round(th0 * (1 - k / 6.5))))
                cv2.line(lay, (int(pts[k, 0] * 16), int(pts[k, 1] * 16)), (int(pts[k + 1, 0] * 16), int(pts[k + 1, 1] * 16)),
                         255, t, cv2.LINE_AA, shift=4)
        return lay.astype(np.float32) / 255.0

    def _make_lashes(self):
        self.lash_u = self._lash_layer(True)
        self.lash_l = self._lash_layer(False)

    def _make_brow(self):
        W, H = self.W, self.H
        rng = self.rng
        lay = np.zeros((H, W), np.uint8)
        for i in range(900):
            s = rng.uniform(-1.05, 1.15)
            x0 = self.cx + s * self.w * 0.52
            yb = self.cy - 0.415 * H - 0.05 * H * np.cos(s * 1.3) + 0.03 * H * s + rng.normal(0, 0.022 * H) * (1 - 0.4 * abs(s))
            ln = self.R * rng.uniform(0.16, 0.34)
            ang = np.radians(-28 + 34 * s + rng.normal(0, 10))
            x1, y1 = x0 + ln * np.cos(ang), yb + ln * np.sin(ang)
            cv2.line(lay, (int(x0 * 16), int(yb * 16)), (int(x1 * 16), int(y1 * 16)), 255, max(1, int(2.6 * CFG.S)), cv2.LINE_AA, shift=4)
        b = cv2.GaussianBlur(lay.astype(np.float32) / 255.0, (0, 0), 3.2 * CFG.S)
        self.brow = np.clip(b * 1.5, 0, 1)

    def _make_reflection(self):
        """imagem refletida na cornea: ceu do amanhecer, linha das serras e o sol."""
        n = self.ir_n
        ay, ax = np.mgrid[-1:1:n * 1j, -1:1:n * 1j].astype(np.float32) * 1.08
        # janela do reflexo: faixa curva no terco superior-esquerdo
        yb = ay + 0.20 + 0.20 * ax * ax          # curvatura (esfera)
        band = smooth(-0.55, -0.20, yb) * (1 - smooth(0.02, 0.10, yb))
        horizon = 0.02 - 0.05 * np.sin(ax * 7.0 + 1.0) * np.exp(-((ax + 0.2) / 0.6) ** 2) - 0.03 * np.sin(ax * 13.0)
        land = smooth(horizon - 0.015, horizon + 0.02, yb)
        skyc = np.zeros((n, n, 3), np.float32)
        g = np.clip((0.06 - yb) / 0.55, 0, 1)
        skyc[..., 0] = 1.0 - 0.55 * g
        skyc[..., 1] = 0.72 - 0.30 * g
        skyc[..., 2] = 0.50 + 0.10 * g
        refl = skyc * (band * (1 - land))[..., None]
        refl *= (1 - smooth(0.80, 1.02, np.sqrt(ax * ax + ay * ay)))[..., None]
        self.refl = cv2.GaussianBlur(refl, (0, 0), 9.0 * CFG.S + 0.5) * 0.50
        sx, sy = -0.18, -0.19
        d2 = (ax - sx) ** 2 + (ay - sy - 0.20 * ax * ax + 0.20 * sx * sx) ** 2
        self.sunspot = (np.exp(-d2 / 0.0018) * 3.0 + np.exp(-d2 / 0.02) * 0.45 + np.exp(-d2 / 0.12) * 0.10).astype(np.float32)
        self.win = (np.exp(-((ax - 0.36) / 0.10) ** 2 - ((ay + 0.30) / 0.16) ** 2) * 0.35).astype(np.float32)

    # ---------- quadro ----------
    def frame(self, i):
        t = i / FPS
        W, H, S = self.W, self.H, CFG.S
        xx, yy = self.xx, self.yy
        # --- animacao
        sunrise = float(smooth(0.3, 3.4, t))                 # luz do sol chegando
        tb = 2.95
        bl = 0.0
        if tb <= t < tb + 0.11:
            bl = ease((t - tb) / 0.11)
        elif tb + 0.11 <= t < tb + 0.15:
            bl = 1.0
        elif tb + 0.15 <= t < tb + 0.36:
            bl = 1.0 - ease((t - tb - 0.15) / 0.21)
        squint = 0.10 * sunrise
        blink = float(np.clip(bl + squint * (1 - bl), 0, 1))
        rp = float(lerp(0.44, 0.27, smooth(0.5, 3.0, t))) + 0.008 * np.sin(t * 5.1)
        gx = 0.014 * self.w * float(smooth(1.55, 1.63, t)) - 0.020 * self.w * float(smooth(3.85, 3.93, t))
        gy = -0.006 * H * float(smooth(1.55, 1.63, t)) - 0.010 * H * float(smooth(3.85, 3.93, t))
        ix, iy = self.ix + gx, self.iy + gy
        # --- luz: a frente de sol desce pelo rosto
        sun_col = np.array([1.00, 0.60, 0.30], np.float32) * 2.5
        sky_col = np.array([0.24, 0.31, 0.50], np.float32) * (0.52 + 0.22 * sunrise)
        ld = np.array([-0.46, -0.16, 0.87], np.float32)
        ld /= np.linalg.norm(ld)
        yf = float(lerp(-0.25, 1.35, smooth(0.25, 3.1, t))) * H
        sunm = smooth(yf + 0.16 * H, yf - 0.16 * H, yy + 0.10 * (xx - W / 2)).astype(np.float32) * 0.97 + 0.03
        self._sunm = sunm
        xs = np.arange(W, dtype=np.float32)
        yu, yl = self.lids(xs, blink)
        dyu = yu - self.yu0
        # --- pele (desliza com a palpebra superior)
        nrm = self.skin_n
        ndl = nrm[..., 0] * ld[0] + nrm[..., 1] * ld[1] + nrm[..., 2] * ld[2]
        lam = np.clip(ndl * 0.62 + 0.38, 0, 1) ** 1.35
        lam = lam * (1 - 0.22 * self.wrinkle)
        skin = self.skin_alb * (sun_col[None, None] * (lam * sunm * 0.62)[..., None] + sky_col[None, None] * (self.skin_ao[..., None] * 0.95))
        hv = nrm[..., 0] * (ld[0] * 0.5) + nrm[..., 1] * (ld[1] * 0.5) + nrm[..., 2] * ((ld[2] + 1) * 0.5)
        spec = np.clip(hv, 0, 1) ** 22 * (0.5 + 1.0 * (self.pore + 0.5)) * sunm
        skin += (sun_col * 0.030)[None, None] * spec[..., None]
        # dispersao sob a pele: transicao de sombra avermelhada
        skin[..., 0] *= 1.0 + 0.10 * (1 - lam) * sunm
        above = yy < (self.yu0[None, :] + 4 * S)
        if blink > 0.002:
            # mapeia: pixels acima da nova borda vem de cima (palpebra desce)
            k = np.clip((self.yu0[None, :] - (yy - dyu[None, :])) / (0.16 * H), 0, 1)     # 0 na borda, 1 longe
            src_y = yy - dyu[None, :] * (1 - k)
            skin_u = cv2.remap(skin, xx, src_y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            skin_u = skin
        # --- globo ocular
        img = np.zeros((H, W, 3), np.float32)
        mid = (yu + yl) * 0.5
        hgt_open = np.maximum(yl - yu, 1.0)
        v = (yy - mid[None, :]) / (0.5 * (self.yl0 - self.yu0)[None, :] + 1e-3)
        s = (xx - self.cx) / (self.w / 2)
        scl = np.array([0.66, 0.62, 0.60], np.float32)
        shade = (1.0 - 0.55 * np.abs(s) ** 1.5) * (1.0 - 0.34 * np.clip(-v, 0, 1) ** 1.2) * (1.0 - 0.16 * np.clip(v, 0, 1))
        # sombra da palpebra superior sobre o globo
        lidsh = np.exp(-np.clip(yy - yu[None, :], 0, None) / (0.050 * H))
        shade = shade * (1.0 - 0.70 * lidsh)
        veins = cv2.remap(self.vein_tex, (xx * (700.0 / W)) % 1024, (yy * (2600.0 / H)) % 1024, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        vmask = smooth(0.58, 0.66, veins) * smooth(0.25, 0.85, np.abs(s))
        shade = shade * (0.72 + 0.28 * np.clip(1 - np.abs(s) ** 2.2, 0, 1))
        sc = scl[None, None] * shade[..., None] * (sun_col[None, None] * (0.30 * sunm)[..., None] + sky_col[None, None] * 0.95)
        sc[..., 1] *= 1 - 0.30 * vmask
        sc[..., 2] *= 1 - 0.34 * vmask
        # caruncula (canto interno, rosado)
        car = np.exp(-(((xx - (self.cx - self.w * 0.455)) / (0.035 * self.w)) ** 2 + ((yy - (self.cy + 0.035 * H)) / (0.030 * H)) ** 2))
        sc = sc * (1 - 0.55 * car[..., None]) + (np.array([0.50, 0.17, 0.14], np.float32) * (sun_col * 0.3 * sunrise + sky_col))[None, None] * (0.55 * car[..., None])
        img[:] = sc
        # iris
        n = self.ir_n
        x0, y0 = int(round(ix - n / 2)), int(round(iy - n / 2))
        rr, th = self.ir_r, self.ir_th
        # caustica: lado oposto a luz mais claro
        caus = 0.55 + 0.95 * np.clip(0.5 + 0.5 * (np.cos(th) * 0.86 + np.sin(th) * 0.34), 0, 1) ** 1.6
        ys_i = (np.arange(n, dtype=np.float32) + y0)[:, None]
        xs_i = np.clip(np.arange(n) + x0, 0, W - 1)
        yi_c = np.clip(np.arange(n) + y0, 0, H - 1)
        sm_i = sunm[yi_c][:, xs_i]
        iris_lit = self.iris * (sun_col[None, None] * (caus * sm_i * 0.62)[..., None] + sky_col[None, None] * 0.60)
        # pupila
        pr = rp * (1 + 0.012 * np.sin(th * 5 + 1.0))
        pm = 1 - smooth(pr - 0.012, pr + 0.016, rr)
        ruff = np.exp(-((rr - pr - 0.02) / 0.018) ** 2) * 0.5
        iris_lit = iris_lit * (1 - ruff[..., None]) * (1 - pm[..., None])
        # sombra da palpebra dentro da iris
        lsh = np.exp(-np.clip(ys_i - yu[xs_i][None, :], 0, None) / (0.055 * H))
        iris_lit *= (1 - 0.72 * lsh)[..., None]
        # reflexos na cornea
        refl_k = 0.10 + 0.50 * float(smooth(0.6, 2.4, t))
        iris_lit += self.refl * (np.array([1.0, 0.82, 0.62], np.float32) * refl_k)[None, None] * (1 - 0.6 * lsh[..., None])
        iris_lit += self.sunspot[..., None] * (sun_col * (0.75 * float(smooth(1.2, 2.5, t))))[None, None]
        iris_lit += self.win[..., None] * (sky_col * 1.2)[None, None]
        ia = self.iris_a
        over(img, iris_lit * ia[..., None], ia, x0, y0)
        # linha d'agua da palpebra inferior (brilho umido)
        wl = np.exp(-((yy - (yl[None, :] - 0.006 * H)) / (0.0045 * H)) ** 2) * (1 - np.abs(s) ** 3).clip(0, 1)
        img += wl[..., None] * (sun_col * 0.16 * sunrise + sky_col * 0.35)[None, None]
        wb = smooth(yl[None, :] - 0.020 * H, yl[None, :] - 0.011 * H, yy) * self.eye_fade[None, :]
        wcol = np.array([0.46, 0.21, 0.16], np.float32) * (sun_col * 0.30 * sunrise + sky_col * 1.1)
        img = img * (1 - wb[..., None]) + wcol[None, None] * wb[..., None]
        img += (np.exp(-((yy - (yl[None, :] - 0.019 * H)) / (0.0030 * H)) ** 2) * (1 - np.abs(s) ** 3).clip(0, 1))[..., None] * (sun_col * 0.10 * sunrise + sky_col * 0.30)[None, None]
        # sombra dos cilios superiores no globo
        if blink < 0.9:
            Ms = np.float32([[1, 0, 6 * S], [0, 1, 0.050 * H]])
            lsd = cv2.warpAffine(self.lash_u, Ms, (W, H))
            if blink > 0.002:
                lsd = cv2.remap(lsd, xx, (yy - dyu[None, :]).astype(np.float32), cv2.INTER_LINEAR)
            lsd = cv2.GaussianBlur(lsd, (0, 0), 5.0 * S)
            img *= (1 - 0.42 * lsd[..., None])
        # --- palpebras/pele por cima
        open_m = np.clip((yy - yu[None, :]) / (2.0 * S) + 0.5, 0, 1) * np.clip((yl[None, :] - yy) / (2.0 * S) + 0.5, 0, 1)
        skin_c = np.where(above[..., None], skin_u, skin)
        # borda das palpebras: margem mais escura e avermelhada
        em_u = np.exp(-((yy - yu[None, :] + 0.004 * H) / (0.010 * H)) ** 2)
        em_l = np.exp(-((yy - yl[None, :] - 0.006 * H) / (0.011 * H)) ** 2)
        ef = self.eye_fade[None, :]
        skin_c = skin_c * (1 - 0.42 * (em_u * ef)[..., None]) * (1 - 0.16 * (em_l * ef)[..., None])
        skin_c[..., 0] *= 1 + 0.22 * em_l * ef
        # dobra da palpebra
        cr_y = self.crease[None, :] + dyu[None, :] * 0.55
        crs = np.exp(-((yy - cr_y) / (0.011 * H)) ** 2) * (1 - 0.9 * blink) * ef
        skin_c = skin_c * (1 - 0.50 * crs[..., None])
        # sobrancelha
        skin_c = skin_c * (1 - 0.90 * self.brow[..., None])
        # profundidade de campo: pele levemente fora de foco longe do olho
        img = img * open_m[..., None] + skin_c * (1 - open_m[..., None])
        # --- cilios
        lu = self.lash_u
        if blink > 0.002:
            lu = cv2.remap(lu, xx, (yy - dyu[None, :]).astype(np.float32), cv2.INTER_LINEAR)
        lash = np.clip(lu + self.lash_l, 0, 1)
        lcol = np.array([0.010, 0.008, 0.007], np.float32) * (sun_col * 0.25 * sunrise + sky_col)
        img = img * (1 - lash[..., None]) + lcol[None, None] * lash[..., None]
        # brilho quente nas pontas dos cilios (contraluz lateral)
        tipg = cv2.GaussianBlur(lash, (0, 0), 2.0 * S) * 0.012 * sunrise
        img += tipg[..., None] * sun_col[None, None]
        # --- pos
        soft = cv2.GaussianBlur(img, (0, 0), 3.2 * S)
        dfocus = np.clip((np.abs(yy - self.cy) / (0.40 * H)) ** 2 + (np.abs(xx - self.cx) / (0.62 * W)) ** 4, 0, 1)[..., None]
        img = img * (1 - 0.85 * dfocus) + soft * (0.85 * dfocus)
        img = bloom(img, 0.14)
        vignette(img, 0.42)
        disp = tonemap(img, 0.80 - 0.12 * sunrise, sat=1.0)
        grain(disp, 0.020, self.rng)
        zoom = 1.0 + 0.05 * (t / self.dur)
        return finish_frame(disp, t, self.dur, 0.0, 0.0, 1.4, self.seed, zoom)
