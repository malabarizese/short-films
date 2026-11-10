# smp.py — leitor de amostras (VSCO-2 CE, CC0) e sampler simples
import os, re, wave, glob
import numpy as np

SR = 48000
ROOT = '/home/claude/dl/vsco'
NOTE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def read_wav(path):
    """le wav PCM 16/24 bits -> float32 (n, 2), taxa."""
    with wave.open(path, 'rb') as w:
        ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, '<i2').astype(np.float32) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        x = v.astype(np.float32) / float(1 << 23)
    else:
        raise ValueError('formato nao suportado: %s' % path)
    x = x.reshape(-1, ch)
    if ch == 1:
        x = np.repeat(x, 2, axis=1)
    return x[:, :2], sr


def name_midi(tok):
    m = re.match(r'^([A-G])(#?)(-?\d)$', tok)
    if not m:
        return None
    return 12 * (int(m.group(3)) + 1) + NOTE[m.group(1)] + (1 if m.group(2) else 0)


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def detect_f0(x, sr, lo=25.0, hi=4500.0, t0=0.25, dur=1.2):
    """f0 por autocorrelacao (via FFT) num trecho depois do ataque."""
    a = int(t0 * sr); b = min(len(x), a + int(dur * sr))
    if b - a < int(0.2 * sr):
        a = 0; b = len(x)
    s = x[a:b].mean(1).astype(np.float64)
    s = s - s.mean()
    s *= np.hanning(len(s))
    n = 1 << int(np.ceil(np.log2(2 * len(s))))
    F = np.fft.rfft(s, n)
    ac = np.fft.irfft(np.abs(F) ** 2)[:n // 2]
    ac /= ac[0] + 1e-20
    kmin, kmax = int(sr / hi), min(int(sr / lo), len(ac) - 2)
    seg = ac[kmin:kmax]
    # primeiro pico forte (evita sub-harmonicos): menor atraso com valor >= 0.9 * maximo
    mx = seg.max()
    cand = np.nonzero((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:]) & (seg[1:-1] >= 0.88 * mx))[0] + 1
    if len(cand) == 0:
        return 0.0, 0.0
    k = cand[0] + kmin
    y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
    d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-20)
    return sr / (k + d), float(mx)
