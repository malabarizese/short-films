# terrain.py — serras em camadas com neblina de vale (mar de morros) e relva do topo da colina
import numpy as np
import cv2
import sky as skylib
from core import CFG, f32, smooth, blur, over, lerp

_tex_cache = {}


def noise_tex(n, beta, seed, lo=2.0):
    key = (n, beta, seed, lo)
    if key not in _tex_cache:
        _tex_cache[key] = skylib.uniformize(skylib.fbm_nd((n, n), beta, seed, lo=lo))
    return _tex_cache[key]


def fbm1d(n, beta, rng, lo=1.0):
    k = np.fft.rfftfreq(n) * n
    amp = np.zeros_like(k)
    amp[1:] = k[1:] ** (-beta / 2)
    amp[k < lo] = 0
    ph = rng.random(len(k)) * 2 * np.pi
    f = np.fft.irfft(amp * np.exp(1j * ph), n)
    return (f - f.mean()) / f.std()


AZ = np.linspace(-80.0, 80.0, 16001)       # graus, passo 0.01


def range_profile(rng, base, amp, beta=2.3, lo=3.0, ridged=0.35):
    f = fbm1d(len(AZ), beta, rng, lo=lo)
    r = 1.0 - np.abs(fbm1d(len(AZ), beta, rng, lo=lo * 1.5)) * 0.9
    y = (1 - ridged) * f * 0.5 + ridged * (r - 0.4) * 1.4
    y = np.convolve(y, np.hanning(41) / np.hanning(41).sum(), mode='same')
    y += 0.025 * fbm1d(len(AZ), 2.4, rng, lo=30)
    return base + amp * y


def domes_profile(rng, base, n, wr, hr, k=7.0, rough=0.05):
    c = rng.uniform(AZ[0], AZ[-1], n)
    w = rng.uniform(wr[0], wr[1], n)
    h = rng.uniform(hr[0], hr[1], n)
    acc = np.zeros(len(AZ))
    kk = k / hr[1]
    for ci, wi, hi in zip(c, w, h):
        b = hi * np.exp(-((AZ - ci) / wi) ** 2)
        acc += np.exp(kk * b)
    y = np.log(acc) / kk - np.log(n) / kk * 0.0
    y = y - y.min()
    y += rough * hr[1] * fbm1d(len(AZ), 2.2, rng, lo=25)
    return base + y


def crown_bumps(rng, tree_deg, forest_frac):
    """silhueta de copas ao longo da crista (graus): copas arredondadas de tamanhos variados."""
    n = len(AZ)
    step = AZ[1] - AZ[0]
    bumps = np.zeros(n)
    x = AZ[0]
    while x < AZ[-1]:
        w = tree_deg * rng.uniform(0.6, 1.5)
        h = rng.uniform(0.30, 1.0) ** 1.3
        i0, i1 = int((x - AZ[0]) / step), int((x + w - AZ[0]) / step) + 1
        i1 = min(i1, n)
        if i1 > i0 + 1:
            u = np.linspace(0, 1, i1 - i0)
            bumps[i0:i1] = np.maximum(bumps[i0:i1], h * np.sqrt(np.clip(1 - (2 * u - 1) ** 2, 0, 1)))
        x += w * rng.uniform(0.30, 0.75)
    return tree_deg * 0.72 * (0.22 + 0.78 * bumps), None


class Ridges:
    """Camadas de serras (do fundo para a frente) em coordenadas angulares."""

    SPECS = [
        dict(D=80.0, kind='range', base=1.20, amp=0.62, beta=2.9, lo=2.0, ridged=0.45),
        dict(D=58.0, kind='range', base=0.86, amp=0.50, beta=2.8, lo=2.5, ridged=0.40),
        dict(D=42.0, kind='range', base=0.52, amp=0.44, beta=2.8, lo=3.0, ridged=0.30),
        dict(D=31.0, kind='range', base=0.20, amp=0.38, beta=2.7, lo=4.0, ridged=0.25),
        dict(D=23.0, kind='domes', base=-0.22, n=50, wr=(2.0, 6.0), hr=(0.18, 0.55)),
        dict(D=17.0, kind='domes', base=-0.48, n=48, wr=(2.0, 6.0), hr=(0.22, 0.62)),
        dict(D=12.5, kind='domes', base=-0.75, n=46, wr=(2.2, 6.5), hr=(0.26, 0.74)),
        dict(D=9.2, kind='domes', base=-1.05, n=44, wr=(2.4, 7.0), hr=(0.30, 0.86), forest=0.40, tree=0.094),
        dict(D=6.8, kind='domes', base=-1.42, n=42, wr=(2.6, 7.5), hr=(0.36, 1.00), forest=0.45, tree=0.127),
        dict(D=5.0, kind='domes', base=-1.86, n=40, wr=(2.8, 8.5), hr=(0.42, 1.15), forest=0.50, tree=0.172),
        dict(D=3.7, kind='domes', base=-2.40, n=36, wr=(3.2, 9.5), hr=(0.50, 1.35), forest=0.50, tree=0.232),
        dict(D=2.7, kind='domes', base=-3.10, n=32, wr=(3.6, 11.0), hr=(0.60, 1.60), forest=0.55, tree=0.318),
        dict(D=2.0, kind='domes', base=-4.00, n=30, wr=(4.2, 12.5), hr=(0.72, 1.90), forest=0.55, tree=0.43),
        dict(D=1.45, kind='domes', base=-5.20, n=26, wr=(5.0, 14.0), hr=(0.90, 2.30), forest=0.60, tree=0.59),
        dict(D=1.05, kind='domes', base=-6.90, n=22, wr=(6.0, 17.0), hr=(1.10, 2.90), forest=0.60, tree=0.82),
    ]

    def __init__(self, cam, seed=3, fog_alt=0.125):
        self.cam = cam
        rng = np.random.default_rng(seed)
        self.layers = []
        W, H = cam.W, cam.H
        xs = np.arange(W) + 0.5
        yy_full = np.arange(H, dtype=np.float32)[:, None] + 0.5
        yaw_deg = np.degrees(cam.yaw)
        prev_crest = None
        raw = []
        for sp in self.SPECS:
            if sp['kind'] == 'range':
                el = range_profile(rng, sp['base'], sp['amp'], sp['beta'], sp['lo'], sp['ridged'])
            else:
                el = domes_profile(rng, sp['base'], sp['n'], sp['wr'], sp['hr'])
            crown = None
            fmask = np.zeros(len(AZ))
            if sp.get('forest'):
                crown, _ = crown_bumps(rng, sp['tree'], sp['forest'])
                hscale = sp['hr'][1]
                ks = 512.0 / (hscale * 5.0)
                tex2 = noise_tex(512, 3.0, 150 + len(raw), lo=2.0)
                f2 = cv2.remap(tex2, ((AZ[None, :] * ks * 0.16 + 91) % 512).astype(np.float32),
                               np.full((1, len(AZ)), 17.0, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)[0]
                th = 1.0 - sp['forest']
                fmask = smooth(th - 0.07, th + 0.09, f2)
                soft = cv2.GaussianBlur(fmask[None, :].astype(np.float32), (0, 0), sp['tree'] * 2.2 / (AZ[1] - AZ[0]))[0]
                crown = crown * smooth(0.25, 0.95, soft)
            raw.append((sp, el, crown, fmask))
        self.raw = raw
        for li, (sp, el, crown, fmask) in enumerate(raw):
            elc = el + (crown if crown is not None else 0.0)
            # projeta a crista (az, el) na tela e interpola por coluna
            az_r, el_r = np.radians(AZ), np.radians(elc)
            d = np.stack([np.sin(az_r) * np.cos(el_r), np.sin(el_r), np.cos(az_r) * np.cos(el_r)], -1)
            px, py, pz = cam.project_dir(d)
            ok = pz > 0.2
            order = np.argsort(px[ok])
            crest = np.interp(xs, px[ok][order], py[ok][order]).astype(np.float32)
            # crista suave (sem copas) para o sombreamento
            el0 = np.radians(el)
            d0 = np.stack([np.sin(az_r) * np.cos(el0), np.sin(el0), np.cos(az_r) * np.cos(el0)], -1)
            px0, py0, _ = cam.project_dir(d0)
            crest0 = np.interp(xs, px0[ok][order], py0[ok][order]).astype(np.float32)
            fm = np.interp(xs, px[ok][order], fmask[ok][order]).astype(np.float32)
            azx = np.interp(xs, px[ok][order], AZ[ok][order]).astype(np.float32)
            self.layers.append(dict(sp=sp, crest=crest, crest0=crest0, fmask=fm, az=azx))
        # bandas de linhas e mapas estaticos
        n = len(self.layers)
        for i, Lr in enumerate(self.layers):
            r0 = int(max(np.floor(Lr['crest'].min()) - 2, 0))
            if i + 1 < n:
                r1 = int(min(np.ceil(self.layers[i + 1]['crest'].max()) + 4 + 0.06 * H, H))
            else:
                r1 = H
            r1 = max(r1, min(r0 + 4, H))
            Lr['r0'], Lr['r1'] = r0, r1
            if r0 >= H:
                Lr['skip'] = True
                continue
            Lr['skip'] = False
            yy = yy_full[r0:r1]
            alpha = np.clip(yy - Lr['crest'][None, :] + 0.5, 0, 1).astype(np.float32)
            Lr['alpha'] = alpha
            sp = Lr['sp']
            # profundidade abaixo da crista em graus
            ppd = cam.f * np.radians(1.0)
            dep = np.maximum(yy - Lr['crest0'][None, :], 0) / ppd
            hscale = sp.get('amp', None) or sp['hr'][1]
            u = np.clip(dep / (hscale * 1.1), 0, 1)
            # inclinacao lateral: derivada da crista suavizada
            sl = np.gradient(cv2.GaussianBlur(Lr['crest0'][None, :], (0, 0), 6 * CFG.S + 1)[0])
            sl = np.clip(sl * 2.2, -1, 1).astype(np.float32)
            nx = (-sl[None, :] * (1 - 0.5 * u)).astype(np.float32)
            ny = (0.85 * (1 - u) ** 1.5 + 0.12).astype(np.float32)
            nz = (0.25 + 0.75 * u).astype(np.float32)
            nl = np.sqrt(nx * nx + ny * ny + nz * nz)
            Lr['nx'], Lr['ny'], Lr['nz'] = nx / nl, ny / nl, nz / nl
            # textura de vegetacao: manchas alongadas na horizontal (vista rasante)
            tex = noise_tex(512, 2.6, 100 + i, lo=3.0)
            ks = 512.0 / (hscale * 5.0)
            mx = ((Lr['az'][None, :] * ks * 0.35 + dep * ks * 0.25) % 512).astype(np.float32)
            my = ((dep * ks * 1.6 + Lr['az'][None, :] * ks * 0.05) % 512).astype(np.float32)
            t = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            Lr['tex'] = t
            if sp.get('forest'):
                tex2 = noise_tex(512, 3.0, 150 + i, lo=2.0)
                f2 = cv2.remap(tex2, ((Lr['az'][None, :] * ks * 0.16 + dep * ks * 0.10 + 91) % 512).astype(np.float32),
                               ((dep * ks * 0.8 + 17 + 0 * mx) % 512).astype(np.float32), cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_WRAP)
                th = 1.0 - sp['forest']
                fm = smooth(th - 0.07, th + 0.09, f2)
                fm = np.clip(fm + 0.45 * smooth(0.5, 1.0, u), 0, 1)
            else:
                fm = np.zeros_like(t)
            Lr['forest'] = (fm * (0.7 + 0.3 * t)).astype(np.float32)
            Lr['u'] = u.astype(np.float32)
            # neblina na frente desta camada (entre ela e a proxima mais perto)
            if i + 1 < n:
                Df = np.sqrt(sp['D'] * self.layers[i + 1]['sp']['D'])
            else:
                Df = sp['D'] * 0.6
            el_top = -np.degrees(np.arctan(fog_alt / Df))
            wn = noise_tex(512, 2.7, 200 + i, lo=2.0)
            nline = cv2.remap(wn, ((Lr['az'][None, :] * 1.3) % 512).astype(np.float32),
                              np.full((1, W), 37.0 * i % 512, np.float32), cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_WRAP)[0]
            nline = cv2.GaussianBlur(nline[None, :], (0, 0), 14 * CFG.S + 2)[0]
            thick = max(0.10, 0.35 * abs(el_top)) + 0.05
            top_el = el_top + (nline - 0.5) * thick * 1.0
            ytop = cam.H / 2 - cam.f * np.tan(np.radians(top_el) - cam.pitch)
            Lr['fog_ytop'] = ytop.astype(np.float32)
            Lr['fog_soft'] = float(thick * ppd)
            Lr['fog_D'] = Df
        self.fog_wisp = noise_tex(1024, 2.9, 300, lo=2.0)

    def render(self, img, L, horcol, fog=0.5, vis_km=45.0, fog_t=0.0, green=1.0, cloud_shadow=None, fog_glow=0.25):
        """desenha as camadas em img. L = dict de luz; horcol (W,3) cor do horizonte."""
        cam = self.cam
        W, H = cam.W, cam.H
        s = L['sun']
        lx, ly, lz = float(s @ cam.right), float(s @ cam.up), float(-(s @ cam.fwd))
        sunCol = L['sunCol'] / np.pi
        amb = L['skyAvg']
        sunvis = float(smooth(-0.6, 1.2, L['el']))
        hc = horcol[None, :, :]
        yy_full = np.arange(H, dtype=np.float32)[:, None] + 0.5
        for i, Lr in enumerate(self.layers):
            if Lr['skip']:
                continue
            sp = Lr['sp']
            r0, r1 = Lr['r0'], Lr['r1']
            D = sp['D']
            lam = np.clip(Lr['nx'] * lx + Lr['ny'] * ly + Lr['nz'] * lz, 0, 1)
            lam = lam * 0.85 + 0.15 * max(ly, 0)
            t = Lr['tex']
            fr = Lr['forest']
            # albedo: pasto (claro) x floresta (escuro), levemente azulado com a distancia
            pa = np.array([0.085, 0.135, 0.040]) * (0.75 + 0.25 * green) + np.array([0.05, 0.03, 0.0]) * (1 - green)
            fo = np.array([0.020, 0.046, 0.016])
            alb = (pa[None, None, :] * (0.72 + 0.56 * t[..., None])) * (1 - fr[..., None]) + fo[None, None, :] * fr[..., None]
            if cloud_shadow is not None:
                lam = lam * cloud_shadow(Lr, i)
            col = alb * (sunCol[None, None, :] * (lam[..., None] * sunvis) + amb[None, None, :] * (0.55 + 0.45 * Lr['ny'][..., None]))
            # perspectiva aerea
            T = np.exp(-D / (vis_km * np.array([1.25, 1.0, 0.78])))
            col = col * T[None, None, :] + hc * (1 - T[None, None, :]) * 0.93
            a = Lr['alpha'][..., None]
            roi = img[r0:r1]
            roi *= (1 - a)
            roi += (col * a).astype(np.float32)
            # neblina de vale em frente
            if fog > 0.005 and Lr['fog_ytop'].min() < H:
                f0 = int(max(np.floor(Lr['fog_ytop'].min() - Lr['fog_soft'] * 1.5), 0))
                f1 = H if i + 1 >= len(self.layers) else int(min(self.layers[i + 1]['r1'], H))
                if f1 > f0 + 1:
                    yy = yy_full[f0:f1]
                    sh = smooth(-1.2, 2.6, (yy - Lr['fog_ytop'][None, :]) / Lr['fog_soft'])
                    wx = ((np.arange(W)[None, :] * (300.0 / (W * (0.5 + 0.1 * i))) + fog_t * (8.0 + 2 * i)) % 1024)
                    wy = ((yy * (1100.0 / H) + 60.0 * i) % 1024) + np.zeros((1, W), np.float32)
                    wis = cv2.remap(self.fog_wisp, (wx + np.zeros_like(wy)).astype(np.float32), wy.astype(np.float32),
                                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
                    dens = np.clip(fog * (0.70 + 0.6 * wis), 0, 1) * min(1.0, 0.45 + Lr['fog_D'] / 5.0)
                    fa = (sh * dens).astype(np.float32)[..., None]
                    fcol = hc * (0.95 + fog_glow) + amb[None, None, :] * 0.30
                    roi = img[f0:f1]
                    roi *= (1 - fa)
                    roi += (fcol * fa).astype(np.float32)
        return img
