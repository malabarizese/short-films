# cards.py — cartao de texto final e assinatura com o monstrinho (logo oficial em vetor)
import os
import re
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from core import *

NOITE = np.array([0x14, 0x17, 0x14], np.float32) / 255.0
VERDE = np.array([0x22, 0xA5, 0x57], np.float32) / 255.0
BRANCO = np.array([1.0, 1.0, 1.0], np.float32)


# ---------------- texto ----------------
def text_mask(txt, font_path, size, tracking=0.0):
    font = ImageFont.truetype(font_path, int(size))
    widths = [font.getlength(ch) for ch in txt]
    total = int(sum(widths) + tracking * size * (len(txt) - 1)) + 8
    asc, desc = font.getmetrics()
    im = Image.new('L', (total, asc + desc + 8), 0)
    d = ImageDraw.Draw(im)
    x = 4.0
    for ch, w in zip(txt, widths):
        d.text((x, 4), ch, font=font, fill=255)
        x += w + tracking * size
    return np.asarray(im, dtype=np.float32) / 255.0, asc


class TextCard:
    def __init__(self, name='texto', dur=11.0):
        self.name, self.dur = name, dur
        self.n = int(round(dur * FPS))

    def setup(self):
        W, H, S = CFG.W, CFG.H, CFG.S
        f300 = os.path.join(BRAND, 'titillium-300.ttf')
        f400 = os.path.join(BRAND, 'titillium-400.ttf')
        size1 = 74 * S
        parts = ['Mãos sujas,', ' corpo cansado,', ' mente tranquila.']
        masks = [text_mask(p, f300, size1, 0.035) for p in parts]
        gap = 0.085 * size1                      # respiro extra depois de cada virgula
        tw = sum(m[0].shape[1] - 8 for m in masks) + gap * (len(masks) - 1)
        x = (W - tw) / 2
        y1 = H * 0.455
        self.items = []
        tins = [0.9, 2.1, 3.3]
        for (m, asc), tin in zip(masks, tins):
            self.items.append(dict(mask=m, x=int(x), y=int(y1 - asc), tin=tin, fin=1.0, tout=8.9, fout=2.0))
            x += m.shape[1] - 8 + gap
        m2, asc2 = text_mask('Isso é malabarismo.', f400, 74 * S, 0.035)
        self.items.append(dict(mask=m2, x=int((W - m2.shape[1]) / 2), y=int(H * 0.455 + 118 * S - asc2), tin=5.3, fin=1.0,
                               tout=9.0, fout=2.0))
        return self

    def frame(self, i):
        t = i / FPS
        W, H = CFG.W, CFG.H
        img = np.zeros((H, W, 3), np.float32)
        for it in self.items:
            a = float(smooth(it['tin'], it['tin'] + it['fin'], t) * (1 - smooth(it['tout'], it['tout'] + it['fout'], t)))
            if a <= 0.001:
                continue
            m = it['mask'] * a * 0.94
            add(img, np.repeat(m[..., None], 3, 2), it['x'], it['y'])
        return to_u8(img)


# ---------------- logo em vetor ----------------
def parse_path(d, n_curve=14):
    """so M, L, C absolutos. retorna lista de subcaminhos (Nx2)."""
    toks = re.findall(r'[MLC]|-?\d+\.?\d*(?:e-?\d+)?', d)
    subs, cur, i = [], [], 0
    cmd = None
    pos = np.zeros(2)
    while i < len(toks):
        tk = toks[i]
        if tk in 'MLC':
            cmd = tk
            i += 1
            continue
        if cmd == 'M':
            if cur:
                subs.append(np.array(cur))
            pos = np.array([float(toks[i]), float(toks[i + 1])]); i += 2
            cur = [pos.copy()]
            cmd = 'L'
        elif cmd == 'L':
            pos = np.array([float(toks[i]), float(toks[i + 1])]); i += 2
            cur.append(pos.copy())
        elif cmd == 'C':
            c1 = np.array([float(toks[i]), float(toks[i + 1])])
            c2 = np.array([float(toks[i + 2]), float(toks[i + 3])])
            p1 = np.array([float(toks[i + 4]), float(toks[i + 5])]); i += 6
            for u in np.linspace(0, 1, n_curve)[1:]:
                cur.append((1 - u) ** 3 * pos + 3 * (1 - u) ** 2 * u * c1 + 3 * (1 - u) * u * u * c2 + u ** 3 * p1)
            pos = p1
    if cur:
        subs.append(np.array(cur))
    return subs


def load_logo():
    s = open(os.path.join(BRAND, 'logo-vertical-noite.svg')).read()
    groups = re.findall(r'<g transform="([^"]+)">(.*?)</g>', s, flags=re.S)
    out = []
    for tr, body in groups:
        ops = re.findall(r'(translate|scale)\(([^)]+)\)', tr)

        def apply(P):
            for name, args in reversed(ops):
                v = [float(a) for a in re.split(r'[ ,]+', args.strip())]
                if name == 'translate':
                    P = P + np.array([v[0], v[1] if len(v) > 1 else 0.0])
                else:
                    P = P * (v[0] if len(v) == 1 else np.array(v))
            return P
        paths = []
        for fill, d in re.findall(r'<path fill="([^"]+)" d="([^"]+)"', body):
            subs = [apply(p) for p in parse_path(d)]
            paths.append((fill.upper(), subs))
        out.append(paths)
    vb = [float(v) for v in re.search(r'viewBox="([^"]+)"', s).group(1).split()]
    return out[0], out[1], vb


class LogoCard:
    def __init__(self, name='logo', dur=12.0):          # 7,5 s da vinheta + 4,5 s de tela final parada (para a tela final do YouTube)
        self.name, self.dur = name, dur
        self.n = int(round(dur * FPS))

    def setup(self):
        self.masc, self.word, self.vb = load_logo()
        W, H = CFG.W, CFG.H
        self.k = 0.50 * H / self.vb[3]                       # px por unidade
        self.ox = W / 2 - (self.vb[0] + self.vb[2] / 2) * self.k
        self.oy = H / 2 - (self.vb[1] + self.vb[3] / 2) * self.k
        allp = np.concatenate([p for _, subs in self.masc for p in subs])
        self.mb = (allp[:, 0].min(), allp[:, 1].min(), allp[:, 0].max(), allp[:, 1].max())
        self.mw, self.mh = self.mb[2] - self.mb[0], self.mb[3] - self.mb[1]
        self.base = np.array([(self.mb[0] + self.mb[2]) / 2, self.mb[3]])
        eye = self.masc[4][1][0]
        self.eye_c = np.array([(eye[:, 0].min() + eye[:, 0].max()) / 2, (eye[:, 1].min() + eye[:, 1].max()) / 2])
        self.eye_h = (eye[:, 1].max() - eye[:, 1].min()) / 2
        self.eye_w = (eye[:, 0].max() - eye[:, 0].min()) / 2
        self.head_top = np.array([self.base[0], self.mb[1]])
        self.ball_r = 0.105 * self.mh
        return self

    # --- coreografia da piada: a bola quica na cabeca, quase cai, ele recupera e joga de volta
    def state(self, t):
        mw, mh = self.mw, self.mh
        st = dict(x=0.0, y=0.0, sx=1.0, sy=1.0, rot=0.0, blink=0.0, px=0.0, py=0.0, word=0.0, ball=None, pop=1.0,
                  bg=float(smooth(0.0, 0.35, t)))
        # entrada
        u = np.clip((t - 0.35) / 0.5, 0, 1)
        st['pop'] = float(1 - (1 - u) ** 3 + 0.10 * np.sin(u * np.pi) * (1 - u) * 4 * u) if t >= 0.35 else 0.0
        g = 9.0 * mh
        t_hit1, t_hit2, t_toss = 1.55, 2.20, 3.22
        # deslocamento do monstrinho para recuperar a bola
        dx_catch = 0.46 * mw
        st['x'] = dx_catch * float(smooth(1.62, 2.08, t)) * (1 - float(smooth(3.35, 3.95, t)))
        yb = None
        if 1.02 <= t < t_hit1:
            dtb = t_hit1 - t
            xb = 0.10 * mw
            yb = -self.ball_r - (g * 0.5 * dtb * dtb + 0.9 * mh * dtb * 0)     # cai acelerando ate a cabeca
            yb = -self.ball_r - 0.5 * g * dtb * dtb - 1.2 * mh * dtb
        elif t_hit1 <= t < t_hit2:
            T = t_hit2 - t_hit1
            u = (t - t_hit1) / T
            xb = 0.10 * mw + (dx_catch - 0.10 * mw) * u
            yb = -self.ball_r - 0.5 * g * T * T * u * (1 - u)
        elif t_hit2 <= t < t_toss:
            xb = st['x']
            yb = -self.ball_r
        elif t_toss <= t < t_toss + 0.75:
            dtb = t - t_toss
            xb = st['x']
            yb = -self.ball_r - (5.2 * mh * dtb - 0.5 * g * 0.35 * dtb * dtb)
        if yb is not None:
            st['ball'] = (xb, yb)
        # amassa/estica
        def bump(tc, w, amp):
            return amp * np.exp(-((t - tc) / w) ** 2)
        sq = bump(t_hit1 + 0.03, 0.07, 0.16) + bump(t_hit2 + 0.03, 0.06, 0.07) + bump(t_toss - 0.12, 0.08, 0.12) \
            - bump(t_toss + 0.05, 0.07, 0.10) - bump(0.62, 0.08, 0.06) - 0.05 * float(smooth(1.75, 2.0, t)) * (1 - float(smooth(2.1, 2.25, t)))
        st['sy'] = 1.0 - sq
        st['sx'] = 1.0 + 0.6 * sq
        st['rot'] = np.radians(5.0) * np.exp(-max(t - t_hit2, 0) * 3.2) * np.sin((t - t_hit2) * 16.0) * (1 if t > t_hit2 else 0) \
            + np.radians(-6.0) * float(smooth(1.65, 1.95, t)) * (1 - float(smooth(2.1, 2.3, t)))
        # olho segue a bola
        if st['ball'] is not None and t < t_toss + 0.5:
            bx, by = st['ball']
            ex = st['x'] + (self.eye_c[0] - self.base[0])
            ey = -(self.base[1] - self.eye_c[1])
            d = np.array([bx - ex, by - (ey - 0.0)])
            d = d / (np.linalg.norm(d) + 1e-6)
            k = float(smooth(1.05, 1.3, t)) * (1 - float(smooth(2.45, 2.75, t)) * (1 - float(smooth(2.95, 3.1, t))))
            st['px'], st['py'] = d[0] * self.eye_w * 0.55 * k, d[1] * self.eye_h * 0.42 * k
        for tb in (2.62, 5.45, 9.1):
            if tb <= t < tb + 0.20:
                st['blink'] = float(np.sin((t - tb) / 0.20 * np.pi))
        st['word'] = float(smooth(3.75, 4.55, t))
        return st

    def frame(self, i):
        t = i / FPS
        W, H = CFG.W, CFG.H
        st = self.state(t)
        k, ox, oy = self.k, self.ox, self.oy
        bg = NOITE * st['bg']
        can = np.empty((H, W, 3), np.uint8)
        can[:] = to_u8(bg)

        def col8(c):
            return tuple(int(v) for v in to_u8(np.asarray(c, np.float32)))

        def fill(subs, col):
            pts = [np.round(np.stack([p[:, 0] * k + ox, p[:, 1] * k + oy], 1) * 16).astype(np.int32).reshape(-1, 1, 2) for p in subs]
            cv2.fillPoly(can, pts, col8(col), lineType=cv2.LINE_AA, shift=4)
        # transformacao do monstrinho em torno da base
        cr, sr = np.cos(st['rot']), np.sin(st['rot'])
        pop = st['pop']

        def xf(P):
            Q = (P - self.base) * np.array([st['sx'] * pop, st['sy'] * pop])
            Q = np.stack([Q[:, 0] * cr - Q[:, 1] * sr, Q[:, 0] * sr + Q[:, 1] * cr], 1)
            return Q + self.base + np.array([st['x'], st['y']])
        fade = float(1 - smooth(self.dur - 0.5, self.dur - 0.02, t))
        if pop > 0.01:
            for idx, (fillc, subs) in enumerate(self.masc):
                c = VERDE if fillc == '#22A557' else BRANCO
                ss = subs
                if idx == 4:          # branco do olho: pisca
                    ss = [np.stack([p[:, 0], self.eye_c[1] + (p[:, 1] - self.eye_c[1]) * (1 - 0.94 * st['blink'])], 1) for p in subs]
                if idx in (5, 6):     # pupila e brilho
                    if st['blink'] > 0.55:
                        continue
                    ss = [p + np.array([st['px'], st['py']]) for p in subs]
                fill([xf(p) for p in ss], c * fade + bg * (1 - fade) if fade < 1 else c)
        # bola
        if st['ball'] is not None and pop > 0.5:
            bx, by = st['ball']
            top = xf(self.head_top[None, :])[0] if (2.20 <= t < 3.22) else None
            if top is not None:
                c = top + np.array([-sr, -cr]) * self.ball_r * 0.98
            else:
                c = self.base + np.array([bx, -self.mh + by])
            cv2.circle(can, (int(round((c[0] * k + ox) * 16)), int(round((c[1] * k + oy) * 16))), int(self.ball_r * k * 16),
                       col8(BRANCO), -1, cv2.LINE_AA, shift=4)
        # wordmark
        if st['word'] > 0.003:
            wa = st['word'] * fade
            lay = np.zeros((H, W), np.uint8)
            dy = (1 - st['word']) * 10.0 / k * CFG.S
            for fillc, subs in self.word:
                pts = [np.round(np.stack([p[:, 0] * k + ox, (p[:, 1] + dy) * k + oy], 1) * 16).astype(np.int32).reshape(-1, 1, 2) for p in subs]
                cv2.fillPoly(lay, pts, 255, lineType=cv2.LINE_AA, shift=4)
            a = (lay.astype(np.float32) / 255.0 * wa)[..., None]
            can = (can.astype(np.float32) * (1 - a) + 255.0 * a + 0.5).astype(np.uint8)
        if fade < 0.999:
            can = (can.astype(np.float32) * fade + 0.5).astype(np.uint8)
        return can


class TrailerCard(LogoCard):
    """cartao final do teaser vertical: monstrinho, wordmark, "Assista completo no Malabarize-se" e o endereco do canal.
    Tudo dentro da zona segura do Reels (y 250-1500 em 1920; longe dos 140 px da direita e dos 420 px de baixo)."""

    def __init__(self, name='cartao', dur=7.5):
        LogoCard.__init__(self, name, dur)

    def setup(self):
        LogoCard.setup(self)
        W, H, S = CFG.W, CFG.H, CFG.S
        u = H / 1920.0
        cx = 510 * u                                   # centro da zona segura (80 a 940 px de 1080)
        self.k = 0.60 * W / self.vb[2]
        self.ox = cx - (self.vb[0] + self.vb[2] / 2) * self.k
        self.oy = H * 0.385 - (self.vb[1] + self.vb[3] / 2) * self.k
        f400 = os.path.join(BRAND, 'titillium-400.ttf')
        f700 = os.path.join(BRAND, 'titillium-700.ttf')
        m1, a1 = text_mask('Assista completo no Malabarize-se', f400, 50 * u, 0.02)
        m2, a2 = text_mask('youtube.com/malabarismo', f700, 62 * u, 0.02)
        y0 = self.oy + (self.vb[1] + self.vb[3]) * self.k + 60 * u
        self.lines = [dict(mask=m1, x=int(cx - m1.shape[1] / 2), y=int(y0), tin=1.9, col=BRANCO * 0.93),
                      dict(mask=m2, x=int(cx - m2.shape[1] / 2), y=int(y0 + 96 * u), tin=2.5, col=VERDE)]
        return self

    def state(self, t):
        st = dict(x=0.0, y=0.0, sx=1.0, sy=1.0, rot=0.0, blink=0.0, px=0.0, py=0.0, word=0.0, ball=None, pop=1.0,
                  bg=float(smooth(0.0, 0.3, t)))
        u = np.clip((t - 0.3) / 0.5, 0, 1)
        st['pop'] = float(1 - (1 - u) ** 3 + 0.10 * np.sin(u * np.pi) * (1 - u) * 4 * u) if t >= 0.3 else 0.0
        sq = -0.06 * np.exp(-((t - 0.58) / 0.08) ** 2)
        st['sy'] = 1.0 - sq; st['sx'] = 1.0 + 0.6 * sq
        # olha para cima, para as bolas que ficaram no ar... e depois para quem assiste
        up = 1.0 - float(smooth(3.9, 4.4, t))
        st['py'] = -self.eye_h * 0.44 * up
        st['px'] = self.eye_w * 0.10 * np.sin(t * 1.3) * up
        for tb in (3.35, 5.7):
            if tb <= t < tb + 0.20:
                st['blink'] = float(np.sin((t - tb) / 0.20 * np.pi))
        st['word'] = float(smooth(1.1, 1.7, t))
        return st

    def frame(self, i):
        t = i / FPS
        can = LogoCard.frame(self, i).astype(np.float32) / 255.0
        fade = float(1 - smooth(self.dur - 0.5, self.dur - 0.02, t))
        for ln in self.lines:
            a = float(smooth(ln['tin'], ln['tin'] + 0.6, t)) * fade
            if a <= 0.003:
                continue
            m = ln['mask'] * a
            x, y = ln['x'], ln['y'] + int(round((1 - smooth(ln['tin'], ln['tin'] + 0.6, t)) * 10 * CFG.H / 1920.0))
            h, w = m.shape
            reg = can[y:y + h, x:x + w]
            reg[:] = reg * (1 - m[..., None]) + ln['col'][None, None, :] * m[..., None]
        return to_u8(can)
