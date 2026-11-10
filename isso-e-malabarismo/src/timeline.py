# timeline.py — ordem final dos planos e fabrica unica de planos
import film
from core import FPS

ORDER = ['abertura', 'subida', 'chegada', 'olho', 'orvalho', 'neblina', 'bola', 'tres', 'tl1', 'cinco', 'tl2',
         'sete_falha', 'bola_chao', 'tl3', 'cansaco', 'sete_ok', 'torre', 'recolhe', 'torre_b', 'final', 'texto', 'logo']


class SubShot:
    """um trecho (quadros f0..f1) de outro plano, tratado como plano proprio na montagem."""

    def __init__(self, shot, f0, f1, name):
        self.shot, self.f0, self.f1, self.name = shot, f0, f1, name
        self.n = f1 - f0
        self.dur = self.n / FPS

    def setup(self):
        self.shot.setup()
        return self

    def frame(self, i):
        return self.shot.frame(i + self.f0)


def make(name):
    if name == 'olho':
        from eye import EyeShot
        return EyeShot()
    if name == 'bola':
        from closeups import HandBallShot
        return HandBallShot()
    if name == 'bola_chao':
        from closeups import GroundBallShot
        return GroundBallShot()
    if name == 'texto':
        from cards import TextCard
        return TextCard()
    if name == 'logo':
        from cards import LogoCard
        return LogoCard()
    if name == 'torre':            # plano geral ate a descida das bolas
        return SubShot(film.SHOTS['torre'](), 0, int(round(film.RECOLHE_T0 * FPS)), 'torre')
    if name == 'torre_b':          # volta ao plano geral depois da ultima bola
        sh = film.SHOTS['torre']()
        return SubShot(sh, int(round(film.RECOLHE_T1 * FPS)), sh.n, 'torre_b')
    return film.SHOTS[name]()


_DUR = {}


def durations():
    """duracao (s) de cada plano, sem montar o mundo."""
    if not _DUR:
        for n in ORDER:
            _DUR[n] = make(n).dur
    return dict(_DUR)


def starts():
    d = durations()
    t = 0.0
    out = {}
    for n in ORDER:
        out[n] = t
        t += d[n]
    out['_fim'] = t
    return out


if __name__ == '__main__':
    s = starts(); d = durations()
    for n in ORDER:
        print('%-11s inicio %7.2f  dur %5.2f  quadros %4d' % (n, s[n], d[n], int(round(d[n] * FPS))))
    print('total %.2f s' % s['_fim'])
