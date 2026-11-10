# trailer.py — teaser vertical (1080x1920) composto nativamente: mesma colina, mesma protagonista, cameras proprias.
# Rodar com FILM_H=1920 FILM_ASPECT=9:16 FILM_TIMELINE=trailer
import numpy as np
import film
from core import *
from shots import orbit_cam, HillShot
from scene import mound

BPM = 120.0
# (nome, plano de origem, t inicial no plano, duracao no teaser, velocidade)
SEGS = [
    ('t_bola', 'bola', 0.92, 3.5, 1.0),          # a mao ja entra no 1o quadro, ergue a bola contra o sol e lanca (aos 2,5 s)
    ('t_tres', 'tres', 1.6, 4.5, 1.0),           # tres bolas ao nascer do sol
    ('t_estacoes', 'tl2', 4.0, 6.0, 2.0),        # outono, inverno, primavera: o ipe cresce e floresce
    ('t_queda', 'sete_falha', 4.3, 3.0, 1.0),    # sete bolas... e caem
    ('t_chuva', 'tl3', 1.05, 3.0, 1.0),          # treino na tempestade (raio aos 18,0 s)
    ('t_cansaco', 'cansaco', 2.5, 2.0, 1.0),     # ele se levanta
    ('t_torre', 'torre', 1.1, 7.5, 1.0),         # db97531: as bolas sobem em camera lenta... corta antes de voltarem
    ('t_preto', None, 0.0, 0.5, 1.0),
    ('t_cartao', 'cartao', 0.0, 7.5, 1.0),
]
ORDER = [s[0] for s in SEGS]


def vcam(phi, dist, vfov, feet=0.765, height=1.0):
    """camera vertical em orbita atras do malabarista, com os pes dele na fracao 'feet' da altura do quadro
    (acima dos 420 px de baixo, que a interface do Reels cobre)."""
    ph = np.radians(phi)
    x, z = dist * np.sin(ph), -dist * np.cos(ph)
    hc = height + float(mound(x, z)) - float(mound(0.0, 0.0))
    f = 0.5 / np.tan(np.radians(vfov) / 2)                      # em alturas de quadro
    p = np.arctan((feet - 0.5) / f) - np.arctan(hc / dist)
    target_h = hc + dist * np.tan(p)
    return orbit_cam(phi, dist, height=height, target_h=target_h, vfov=vfov, off=(0.0, 0.0))


def vertical_shot(src):
    """o plano do filme com camera e enquadramento refeitos para a vertical."""
    if src == 'bola':
        from closeups import HandBallShot
        return HandBallShot()
    if src == 'cartao':
        from cards import TrailerCard
        return TrailerCard()
    sh = film.SHOTS[src]()
    bk = dict(sh.ball_kw or {})
    if src == 'tres':
        sh.cam0 = vcam(9.0, 8.2, 36.0, feet=0.775, height=0.9)
        sh.meadow_kw = dict(d_far=9.5, focus=8.2, d_near=0.9)
        sh.dolly = lambda t: (0.0, 0.0, 0.04 * t)
        bk['radius'] = 0.048
    elif src == 'tl2':
        sh.cam0 = vcam(-30.0, 8.6, 46.0, feet=0.775)

        def expo_mul(t):
            m = 1.0
            for b in (2 * film.SEG, 3 * film.SEG):
                m *= 1.0 - 0.93 * np.exp(-((t - b) / 0.22) ** 2)
            return m
        sh.expo_mul = expo_mul
        bk['radius'] = 0.052
    elif src == 'sete_falha':
        sh.cam0 = vcam(-14.0, 8.8, 40.0, feet=0.775, height=0.95)
        sh.meadow_kw = dict(d_far=10.2, focus=8.8, d_near=1.0)
        sh.expo_mul = 1.75
        bk['radius'] = 0.050
    elif src == 'tl3':
        sh.cam0 = vcam(-26.0, 9.6, 46.0, feet=0.775)
        bk['radius'] = 0.055
    elif src == 'cansaco':
        sh.cam0 = vcam(20.0, 6.4, 32.0, feet=0.80, height=0.9)
        sh.meadow_kw = dict(d_far=7.7, focus=6.4, d_near=0.9)
    elif src == 'torre':
        sh.cam0 = vcam(-14.0, 14.9, 46.0, feet=0.775)
        sh.moon = film.moon_fn(0.25, az=6.0, el=13.0)
        sh.shake = 0.6
        bk['radius'] = 0.062
    sh.ball_kw = bk
    return sh


class Seg:
    def __init__(self, name, src, t0, dur, speed):
        self.name, self.src, self.t0, self.dur, self.speed = name, src, t0, dur, speed
        self.n = int(round(dur * FPS))
        self.shot = None

    def setup(self):
        if self.src is not None:
            self.shot = vertical_shot(self.src)
            self.shot.setup()
        return self

    def frame(self, i):
        if self.shot is None:
            return np.zeros((CFG.H, CFG.W, 3), np.uint8)
        fr = self.shot.frame((self.t0 + self.speed * i / FPS) * FPS)
        if self.name == 't_torre':                      # corte seco para o preto no auge, com 4 quadros de queda de luz
            k = (self.n - i) / 4.0
            if k < 1.0:
                fr = (fr.astype(np.float32) * max(k, 0.0)).astype(np.uint8)
        return fr


def make(name):
    for s in SEGS:
        if s[0] == name:
            return Seg(*s)
    raise KeyError(name)


def durations():
    return {s[0]: s[3] for s in SEGS}


def starts():
    t = 0.0
    out = {}
    for s in SEGS:
        out[s[0]] = t
        t += s[3]
    out['_fim'] = t
    return out


if __name__ == '__main__':
    st = starts()
    for s in SEGS:
        print('%-11s %6.2f  dur %4.1f  (plano %s a partir de %.2f s, x%.1f)' % (s[0], st[s[0]], s[3], s[1], s[2], s[4]))
    print('total %.1f s' % st['_fim'])
