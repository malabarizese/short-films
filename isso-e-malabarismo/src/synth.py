# synth.py — sampler (amostras VSCO-2 CE, CC0), sintese de efeitos, reverberacao e mixagem
import os, re, json
from fractions import Fraction
import numpy as np
from scipy import signal
import smp
from smp import SR

SCAN_PATH = '/home/claude/filme/audio/scan.json'
f32 = np.float32


# ---------------------------------------------------------------- instrumentos
class Sample:
    def __init__(self, path, midi, layer, kind):
        self.path, self.midi, self.layer, self.kind = path, midi, layer, kind
        self.x = None

    def load(self):
        if self.x is not None:
            return self.x
        x, sr = smp.read_wav(os.path.join(smp.ROOT, self.path))
        self.sr = sr
        env = np.abs(x).max(1)
        pk = float(env.max()) + 1e-9
        thr = 0.03 if self.kind == 'perc' else 0.015
        i0 = int(np.argmax(env > thr * pk))
        i0 = max(i0 - int(0.003 * sr), 0)
        x = x[i0:]
        # nivel de referencia
        if self.kind == 'perc':
            seg = x[:int(0.5 * sr)]
        else:
            seg = x[int(0.3 * sr):int(2.3 * sr)]
        rms = float(np.sqrt(np.mean(seg ** 2)) + 1e-9)
        self.x = (x / rms * 0.1).astype(f32)
        return self.x


class Inst:
    """instrumento amostrado. kind: 'perc' (decai sozinho) ou 'sus' (sustentado)."""

    def __init__(self, name, kind, samples, layer_edges=(), rel=0.35, att=0.004, gain=1.0):
        self.name, self.kind, self.samples = name, kind, samples
        self.layer_edges = layer_edges           # limites de velocidade entre camadas
        self.rel, self.att, self.gain = rel, att, gain
        self.layers = sorted(set(s.layer for s in samples))
        self._rr = 0

    def pick(self, midi, vel):
        li = 0
        for e in self.layer_edges:
            if vel > e:
                li += 1
        li = min(li, len(self.layers) - 1)
        lay = self.layers[li]
        cands = [s for s in self.samples if s.layer == lay]
        best = min(abs(s.midi - midi) for s in cands)
        cands = [s for s in cands if abs(s.midi - midi) <= best + 0.3]
        self._rr += 1
        return cands[self._rr % len(cands)]

    def note(self, midi, dur, vel=0.6, att=None, rel=None, tune=0.0, vib=None):
        """retorna (n,2) float32 a 48 kHz, ja com envelope. dur = tempo ate o inicio da soltura."""
        s = self.pick(midi, vel)
        x = s.load()
        rel = self.rel if rel is None else rel
        att = self.att if att is None else att
        shift = midi + tune - s.midi
        step = (s.sr / SR) * 2.0 ** (shift / 12.0)
        n_out = int((dur + rel) * SR) + 8
        n_in = min(len(x), int(n_out * step) + 64)
        fr = Fraction(1.0 / step).limit_denominator(480)
        y = signal.resample_poly(x[:n_in], fr.numerator, fr.denominator, axis=0).astype(f32)
        if len(y) < n_out and self.kind == 'sus' and len(x) <= n_in:
            # nota mais longa que a amostra: emenda com outra arcada (crossfade)
            xf = int(1.2 * SR)
            parts = [y]
            total = len(y)
            while total < n_out:
                s2 = self.pick(midi, vel)
                x2 = s2.load()
                st2 = (s2.sr / SR) * 2.0 ** ((midi + tune - s2.midi) / 12.0)
                f2 = Fraction(1.0 / st2).limit_denominator(480)
                y2 = signal.resample_poly(x2, f2.numerator, f2.denominator, axis=0).astype(f32)
                parts.append(y2)
                total += len(y2) - xf
            out = np.zeros((total + xf, 2), f32)
            pos = 0
            for k, p in enumerate(parts):
                p = p.copy()
                if k > 0:
                    p[:xf] *= np.sin(np.linspace(0, np.pi / 2, xf))[:, None].astype(f32) ** 2
                if k < len(parts) - 1:
                    p[-xf:] *= np.cos(np.linspace(0, np.pi / 2, xf))[:, None].astype(f32) ** 2
                out[pos:pos + len(p)] += p
                pos += len(p) - xf
            y = out
        y = y[:n_out].copy()
        n = len(y)
        e = np.ones(n, f32)
        na = int(att * SR)
        if na > 1:
            e[:na] = np.sin(np.linspace(0, np.pi / 2, na)) ** 2
        nd = int(dur * SR)
        if nd < n:
            k = np.arange(n - nd) / max(rel * SR, 1.0)
            e[nd:] *= np.exp(-5.0 * k).astype(f32) * (1 - k.clip(0, 1) ** 4)
        y *= e[:, None]
        v = float(vel)
        return y * (self.gain * (0.12 + 0.88 * v ** 1.5))


def _parse(tok_re, name):
    m = re.search(tok_re, name)
    return m


def load_instruments():
    scan = json.load(open(SCAN_PATH))
    by_dir = {}
    for rel, nm, det, conf, pk, dur in scan:
        by_dir.setdefault(os.path.dirname(rel), []).append((rel, nm, det, conf, pk, dur))

    def mk(dirname, kind, off, layer_fn, skip=None):
        out = []
        for rel, nm, det, conf, pk, dur in by_dir.get(dirname, []):
            b = os.path.basename(rel)
            if nm is None or (skip and skip(b)):
                continue
            midi = nm + off
            if det == det and abs(det - midi) < 0.6:
                midi = det                       # afinacao fina medida
            out.append(Sample(rel, float(midi), layer_fn(b), kind))
        return out

    I = {}
    dyn = {'pp': 0, 'mf': 1, 'f': 2}
    I['piano'] = Inst('piano', 'perc', mk('Keys/Upright Nr1', 'perc', 0, lambda b: dyn[b.split('_')[2]]),
                      layer_edges=(0.34, 0.74), rel=0.45, att=0.001)
    felt = []
    for rel, nm, det, conf, pk, dur in by_dir.get('Keys/Upright Piano', []):
        b = os.path.basename(rel)
        m = re.match(r'Player_dyn(\d)_rr1_(\d+)\.wav', b)
        if not m:
            continue
        midi = 21 + 2 * int(m.group(2))
        if det == det and abs(det - midi) < 0.6:
            midi = det
        felt.append(Sample(rel, float(midi), int(m.group(1)) - 1, 'perc'))
    I['felt'] = Inst('felt', 'perc', felt, layer_edges=(0.5,), rel=0.5, att=0.001)
    I['harp'] = Inst('harp', 'perc', mk('Strings/Harp', 'perc', 0, lambda b: 0, skip=lambda b: 'F7' in b or 'G1' in b),
                     rel=0.8, att=0.001)
    I['glock'] = Inst('glock', 'perc', mk('Percussion/Glock', 'perc', 12, lambda b: 0), rel=1.5, att=0.001)
    v12 = lambda b: 0 if '_v1' in b else 1
    I['vln'] = Inst('vln', 'sus', mk('Strings/Violin Section/susVib', 'sus', 12, v12), layer_edges=(0.55,), rel=0.45, att=0.03)
    I['trem'] = Inst('trem', 'sus', mk('Strings/Violin Section/Trem', 'sus', 12, v12), layer_edges=(0.55,), rel=0.35, att=0.03)
    I['vla'] = Inst('vla', 'sus', mk('Strings/Viola Section/susvib', 'sus', 12, v12), layer_edges=(0.55,), rel=0.45, att=0.03)
    I['vc'] = Inst('vc', 'sus', mk('Strings/Cello Section/susvib', 'sus', 12, v12), layer_edges=(0.55,), rel=0.5, att=0.03)
    I['pizz'] = Inst('pizz', 'perc', mk('Strings/Cello Section/pizzT', 'perc', 12, v12,
                                        skip=lambda b: 'F2_v1' in b or 'C1_v2_RR1' in b or 'D4' in b),
                     layer_edges=(0.55,), rel=0.6, att=0.001)
    I['cb'] = Inst('cb', 'sus', mk('Strings/Solo Contrabass/SusVib', 'sus', 12, v12), layer_edges=(0.55,), rel=0.5, att=0.04)
    I['solo'] = Inst('solo', 'sus', mk('Strings/Solo Violin/Arco Vib', 'sus', 0, lambda b: 0), rel=0.4, att=0.03)
    I['horn'] = Inst('horn', 'sus', mk('Brass/F Horn/sus', 'sus', 12, v12, skip=lambda b: 'A0' in b),
                     layer_edges=(0.6,), rel=0.35, att=0.03)
    return I


def raw(path):
    """amostra crua (sem altura): timpano, bumbo. retorna (n,2) a 48 kHz, aparada no ataque."""
    x, sr = smp.read_wav(os.path.join(smp.ROOT, path))
    env = np.abs(x).max(1)
    i0 = max(int(np.argmax(env > 0.03 * env.max())) - int(0.003 * sr), 0)
    x = x[i0:]
    fr = Fraction(SR / sr).limit_denominator(480)
    return signal.resample_poly(x, fr.numerator, fr.denominator, axis=0).astype(f32)


def repitch(x, semis):
    fr = Fraction(2.0 ** (-semis / 12.0)).limit_denominator(480)
    return signal.resample_poly(x, fr.numerator, fr.denominator, axis=0).astype(f32)


# ---------------------------------------------------------------- utilitarios de sintese
def rng_for(seed):
    return np.random.default_rng(seed)


def white(n, rng, ch=2):
    return rng.standard_normal((n, ch)).astype(f32)


def pink(n, rng, ch=2):
    X = np.fft.rfft(rng.standard_normal((n, ch)), axis=0)
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    X /= np.sqrt(f)[:, None]
    y = np.fft.irfft(X, n, axis=0)
    return (y / (np.std(y) + 1e-9)).astype(f32)


def filt(x, kind, fc, order=2):
    sos = signal.butter(order, fc, btype=kind, fs=SR, output='sos')
    return signal.sosfilt(sos, x, axis=0).astype(f32)


def bp(x, lo, hi, order=2):
    return filt(x, 'band', [lo, hi], order)


def lp(x, fc, order=2):
    return filt(x, 'low', fc, order)


def hp(x, fc, order=2):
    return filt(x, 'high', fc, order)


def ad(n, a, d, curve=1.0):
    """envelope ataque (s) + decaimento exponencial (constante d, s)."""
    t = np.arange(n) / SR
    e = np.exp(-t / d)
    na = max(int(a * SR), 1)
    e[:na] *= (np.arange(na) / na) ** curve
    return e.astype(f32)


def bell(n, power=2.0):
    return (np.sin(np.linspace(0, np.pi, n)) ** power).astype(f32)


def sweep(f0, f1, dur, shape='exp'):
    n = int(dur * SR)
    u = np.arange(n) / max(n - 1, 1)
    f = f0 * (f1 / f0) ** u if shape == 'exp' else f0 + (f1 - f0) * u
    return np.sin(2 * np.pi * np.cumsum(f) / SR).astype(f32), f


def st(x):
    """mono -> estereo."""
    x = np.asarray(x, f32)
    return np.stack([x, x], 1) if x.ndim == 1 else x


# ---------------------------------------------------------------- efeitos sonoros
def sfx_catch(rng, vel=0.6, bright=1.0):
    """bola de malabares caindo na mao: batida curta e macia."""
    n = int(0.16 * SR)
    nz = white(n, rng)
    body = bp(nz, 170, 1500 * bright) * ad(n, 0.0012, 0.022)[:, None]
    slap = bp(nz, 1800, 5200) * ad(n, 0.0008, 0.007)[:, None] * 0.35 * bright
    t = np.arange(n) / SR
    th = np.sin(2 * np.pi * (150 * t - 160 * t * t).clip(0)) * np.exp(-t / 0.028)
    y = body * 0.9 + slap + st(th) * 0.55
    return y * (0.35 + 0.65 * vel)


def sfx_throw(rng, vel=0.5):
    n = int(0.12 * SR)
    y = bp(white(n, rng), 500, 2600) * (bell(n, 1.5) * ad(n, 0.01, 0.045))[:, None]
    return y * 0.5 * (0.3 + 0.7 * vel)


def sfx_thud(rng, vel=0.7):
    """bola caindo na relva."""
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    low = np.sin(2 * np.pi * (92 * t - 60 * t * t).clip(0)) * np.exp(-t / 0.060)
    nz = white(n, rng)
    mid = bp(nz, 180, 900) * ad(n, 0.002, 0.045)[:, None]
    rus = bp(nz, 1800, 6500) * (ad(n, 0.004, 0.09) * (0.6 + 0.4 * np.sin(2 * np.pi * 23 * t) ** 2))[:, None] * 0.22
    return (st(low) * 1.0 + mid * 0.8 + rus) * (0.3 + 0.7 * vel)


def sfx_rustle(rng, dur=0.25, vel=0.5, lo=1200, hi=6000):
    n = int(dur * SR)
    t = np.arange(n) / SR
    e = bell(n, 1.2) * (0.55 + 0.45 * np.sin(2 * np.pi * rng.uniform(14, 26) * t + rng.uniform(0, 6)) ** 2)
    return bp(white(n, rng), lo, hi) * e[:, None] * 0.5 * vel


def sfx_step(rng, vel=0.5):
    """passo na relva: calcanhar + roçar."""
    n = int(0.30 * SR)
    t = np.arange(n) / SR
    nz = white(n, rng)
    a = bp(nz, 700, 4200) * ad(n, 0.006, 0.050)[:, None]
    b = np.roll(bp(nz, 1500, 6500) * ad(n, 0.02, 0.075)[:, None], int(0.07 * SR), axis=0) * 0.6
    b[:int(0.07 * SR)] = 0
    low = st(np.sin(2 * np.pi * 70 * t) * np.exp(-t / 0.035)) * 0.35
    return (a + b + low) * 0.5 * vel


def wind(dur, rng, gust=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    base = pink(n, rng)
    low = bp(base, 90, 700)
    mid = bp(base, 500, 2200)
    l1 = 0.55 + 0.45 * np.sin(2 * np.pi * 0.07 * t + rng.uniform(0, 6)) * np.sin(2 * np.pi * 0.023 * t + 1.0)
    l2 = np.clip(np.sin(2 * np.pi * 0.11 * t + rng.uniform(0, 6)) * 0.6 + np.sin(2 * np.pi * 0.31 * t + 2.0) * 0.4, 0, 1) ** 2
    y = low * l1[:, None] + mid * (0.20 + 0.55 * gust * l2)[:, None]
    return (y / (np.std(y) + 1e-9) * 0.1).astype(f32)


def bird_song(rng, kind=0):
    """um canto curto. retorna mono."""
    out = []
    if kind == 0:        # assobios descendentes
        for k in range(rng.integers(2, 5)):
            d = rng.uniform(0.09, 0.16)
            f0 = rng.uniform(3600, 4600)
            s, f = sweep(f0, f0 * rng.uniform(0.62, 0.80), d)
            out.append(s * bell(len(s), 1.3))
            out.append(np.zeros(int(rng.uniform(0.05, 0.11) * SR), f32))
    elif kind == 1:      # trinado rapido
        f0 = rng.uniform(4800, 6400)
        for k in range(rng.integers(7, 14)):
            d = 0.032
            s, f = sweep(f0 * 0.92, f0 * 1.10, d)
            out.append(s * bell(len(s), 1.0))
            out.append(np.zeros(int(0.022 * SR), f32))
    elif kind == 2:      # duas notas (chamado)
        f0 = rng.uniform(2500, 3300)
        for mul, d in ((1.0, 0.16), (1.26, 0.10), (1.0, 0.22)):
            s, f = sweep(f0 * mul, f0 * mul * 0.97, d)
            vib = 1 + 0.25 * np.sin(2 * np.pi * 28 * np.arange(len(s)) / SR)
            out.append(s * bell(len(s), 1.5) * vib)
            out.append(np.zeros(int(0.045 * SR), f32))
    else:                # pios ascendentes
        for k in range(rng.integers(3, 7)):
            f0 = rng.uniform(3000, 3800) * (1 + 0.04 * k)
            s, f = sweep(f0, f0 * 1.35, 0.055)
            out.append(s * bell(len(s), 1.0))
            out.append(np.zeros(int(rng.uniform(0.06, 0.09) * SR), f32))
    y = np.concatenate(out).astype(f32)
    ph = np.cumsum(np.ones(len(y))) / SR
    y2 = y * (1.0 + 0.0 * ph)
    y = y + 0.22 * np.sign(y) * y * y                      # harmonicos pares suaves
    nz = rng.standard_normal(len(y)).astype(f32) * np.abs(y) * 0.12
    return (y + nz).astype(f32)


def rain(dur, rng, amount=1.0):
    n = int(dur * SR)
    nz = white(n, rng)
    hiss = hp(nz, 2500) * 0.55
    body = bp(nz, 350, 1800) * 0.50
    t = np.arange(n) / SR
    mod = 0.85 + 0.15 * np.sin(2 * np.pi * 0.21 * t + 1.3)
    y = (hiss + body) * mod[:, None]
    # gotas proximas
    nd = int(dur * 55 * amount)
    drops = np.zeros((n, 2), f32)
    for i in range(nd):
        p = rng.integers(0, n - 2000)
        m = int(rng.uniform(0.004, 0.012) * SR)
        g = rng.uniform(0.2, 1.0)
        pan = rng.uniform(0.2, 0.8)
        d = rng.standard_normal(m).astype(f32) * np.exp(-np.arange(m) / (m * 0.3))
        drops[p:p + m, 0] += d * g * (1 - pan)
        drops[p:p + m, 1] += d * g * pan
    y = y + bp(drops, 1500, 7000) * 2.2
    y = lp(y, 7500, 2)
    return (y / (np.std(y) + 1e-9) * 0.1).astype(f32)


def thunder(rng, dur=5.0, crack=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    nz = white(n, rng)
    rumble = lp(nz, 130, 4)
    e = (1 - np.exp(-t / 0.04)) * np.exp(-t / (dur * 0.30))
    roll = 0.55 + 0.45 * np.abs(lp(white(n, rng)[:, :1], 9, 2)[:, 0]) / 0.02
    roll = np.clip(roll, 0.3, 2.2)
    y = rumble * (e * roll)[:, None] * 9.0
    mid = bp(nz, 150, 900) * ((1 - np.exp(-t / 0.02)) * np.exp(-t / (dur * 0.12)))[:, None] * 0.9
    ck = bp(nz, 700, 5000) * (np.exp(-t / 0.05) * (1 - np.exp(-t / 0.003)))[:, None] * 0.55 * crack
    y = y + mid + ck
    return (y / (np.abs(y).max() + 1e-9)).astype(f32)


def whoosh(dur, rng, f0=300, f1=3200, power=2.0):
    n = int(dur * SR)
    nz = white(n, rng)
    # varredura por mistura de bandas
    u = np.linspace(0, 1, n)
    bands = [(200, 600), (500, 1400), (1100, 3000), (2400, 6500)]
    cs = np.log2(np.array([np.sqrt(a * b) for a, b in bands]))
    fc = np.log2(f0) + (np.log2(f1) - np.log2(f0)) * u
    y = np.zeros((n, 2), f32)
    for (a, b), c in zip(bands, cs):
        w = np.exp(-((fc - c) / 0.9) ** 2)
        y += bp(nz, a, b) * w[:, None].astype(f32)
    return y * bell(n, power)[:, None] * 0.3


def heartbeat(rng, vel=1.0):
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    def th(t0, f, d, g):
        tt = np.clip(t - t0, 0, None)
        return np.where(t >= t0, np.sin(2 * np.pi * (f * tt - 40 * tt * tt).clip(0)) * np.exp(-tt / d) * g, 0.0)
    y = th(0.0, 62, 0.060, 1.0) + th(0.24, 74, 0.045, 0.72)
    return st(y.astype(f32)) * vel


# ---------------------------------------------------------------- mixagem
class Bus:
    def __init__(self, dur):
        self.n = int(dur * SR) + SR
        self.dry = np.zeros((self.n, 2), f32)
        self.send = np.zeros((self.n, 2), f32)

    def add(self, x, t, gain=1.0, pan=0.0, send=0.25):
        x = st(x)
        i = int(round(t * SR))
        if i < 0:
            x = x[-i:]; i = 0
        m = min(len(x), self.n - i)
        if m <= 0:
            return
        gl = gain * float(np.cos((pan + 1) * np.pi / 4)) * 1.4142
        gr = gain * float(np.sin((pan + 1) * np.pi / 4)) * 1.4142
        seg = x[:m] * np.array([gl, gr], f32)[None, :]
        self.dry[i:i + m] += seg
        if send > 0:
            self.send[i:i + m] += seg * send


def make_ir(rt=(3.0, 2.5, 1.3), pre=0.022, length=3.6, seed=5):
    n = int(length * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    ir = np.zeros((n, 2))
    for (lo, hi, r) in ((70, 400, rt[0]), (400, 2500, rt[1]), (2500, 9500, rt[2])):
        nz = rng.standard_normal((n, 2))
        sos = signal.butter(2, [lo, hi], btype='band', fs=SR, output='sos')
        ir += signal.sosfilt(sos, nz, axis=0) * np.exp(-6.91 * t / r)[:, None]
    ir *= (1 - np.exp(-t / 0.018))[:, None]
    # primeiras reflexoes esparsas
    for k in range(9):
        p = int(rng.uniform(0.006, 0.070) * SR)
        ir[p, rng.integers(0, 2)] += rng.uniform(0.4, 1.0) * (1 - k / 14) * np.std(ir[:2000]) * 9
    npre = int(pre * SR)
    ir = np.concatenate([np.zeros((npre, 2)), ir])[:n]
    ir /= np.sqrt((ir ** 2).sum(0))[None, :]
    return ir.astype(f32)


def reverb(x, ir):
    out = np.stack([signal.fftconvolve(x[:, c], ir[:, c]) for c in (0, 1)], 1)
    return out[:len(x)].astype(f32)


def limiter(x, ceil_db=-2.0, look=0.005, rel=0.16, hop=8, stats=False):
    """limitador de pico com antecipacao: o ganho desce antes do pico (janela 'look') e volta devagar ('rel')."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    c = 10 ** (ceil_db / 20)
    pk = np.abs(x).max(1)
    nb = len(pk) // hop + 1
    pad = np.zeros(nb * hop, np.float32); pad[:len(pk)] = pk
    pkb = pad.reshape(nb, hop).max(1)
    need = np.minimum(1.0, c / np.maximum(pkb, 1e-9))
    L = max(int(look * SR / hop), 1)
    g = minimum_filter1d(need, size=L + 1, origin=-(L // 2), mode='nearest')      # menor ganho na janela a frente
    a = np.exp(-hop / (rel * SR))
    out = np.empty(nb)
    cur = 1.0
    for i in range(nb):                                                           # soltura exponencial com retencao
        v = g[i]
        cur = v if v < cur else a * cur + (1 - a) * v
        out[i] = cur
    out = uniform_filter1d(out, size=L, origin=(L // 2) - (1 if L > 1 else 0), mode='nearest')   # ataque em rampa
    out = np.minimum(out, minimum_filter1d(need, size=3, mode='nearest'))
    gi = np.repeat(out, hop)[:len(pk)]
    y = (x * gi[:, None]).astype(f32)
    np.clip(y, -c, c, out=y)
    if stats:
        gr = -20 * np.log10(np.maximum(gi, 1e-6))
        return y, dict(max_db=float(gr.max()), frac_1db=float((gr > 1.0).mean()), frac_3db=float((gr > 3.0).mean()))
    return y


def write_wav(path, x, sr=SR):
    import wave
    y = np.clip(x, -1.0, 1.0)
    i = (y * 32767.0).round().astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(i.tobytes())


def write_wav24(path, x, sr=SR):
    import wave
    y = np.clip(x, -1.0, 1.0)
    v = (y * 8388607.0).round().astype(np.int32)
    b = np.empty((v.size, 3), np.uint8)
    f = v.reshape(-1)
    b[:, 0] = f & 0xFF; b[:, 1] = (f >> 8) & 0xFF; b[:, 2] = (f >> 16) & 0xFF
    with wave.open(path, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(3); w.setframerate(sr)
        w.writeframes(b.tobytes())
