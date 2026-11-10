# balance.py — energia por instrumento em janelas do filme + cromagrama (confere tons e equilibrio)
import sys, os, json
sys.path.insert(0, '/home/claude/filme/src')
import numpy as np
import score, synth
from synth import SR
tally = {}
orig = score.mus.add
cur = {'inst': None}
_play = score.play
def play(inst, note, t, dur, vel, **kw):
    cur['inst'] = inst
    _play(inst, note, t, dur, vel, **kw)
score.play = play
def add(x, t, gain=1.0, pan=0.0, send=0.25):
    x = synth.st(x)
    e = float((x.astype(np.float64) ** 2).sum()) * gain * gain
    tally.setdefault(cur['inst'], []).append((t, e, len(x) / SR))
    orig(x, t, gain, pan, send)
score.mus.add = add
score.music()
wins = [('intro 8-24', 8, 24), ('tres 41-49', 41, 49), ('cinco 63-75', 63, 75), ('tl2 75-91', 75, 91), ('tempestade 103-111', 103, 111),
        ('sete_ok 117-129', 117, 129), ('suspenso 129-141', 129.2, 141.5), ('chegada 141-151', 141.5, 151), ('texto', 153, 164)]
insts = sorted(tally)
print('%-20s' % 'janela' + ''.join('%7s' % i for i in insts))
for name, a, b in wins:
    row = []
    for i in insts:
        e = sum(en for (t, en, d) in tally[i] if a <= t < b)
        row.append(10 * np.log10(e / ((b - a) * SR) + 1e-12))
    print('%-20s' % name + ''.join('%7.1f' % v for v in row))
print('notas por instrumento:', {i: len(tally[i]) for i in insts})
# cromagrama da musica seca
x = score.mus.dry.mean(1)
hop = SR; nwin = 2 * SR
names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
f = np.fft.rfftfreq(nwin, 1 / SR)
midi = 69 + 12 * np.log2(np.maximum(f, 1e-6) / 440.0)
ok = (f > 60) & (f < 4000)
pc = np.round(midi).astype(int) % 12
dev = np.abs(midi - np.round(midi))
out = []
for k in range(0, int(score.DUR) - 1):
    seg = x[k * hop:k * hop + nwin] * np.hanning(nwin)
    S = np.abs(np.fft.rfft(seg)) ** 2
    ch = np.array([S[ok & (pc == p) & (dev < 0.25)].sum() for p in range(12)])
    off = S[ok & (dev >= 0.35)].sum()
    out.append((ch / (ch.sum() + 1e-20), off / (S[ok].sum() + 1e-20)))
C = np.array([o[0] for o in out])
print('fora da escala de re maior (C, D#, F, G#, A#) por trecho de 10 s (%):')
bad = C[:, [0, 3, 5, 8, 10]].sum(1)
print(' '.join('%d:%.0f' % (k * 10, 100 * bad[k * 10:k * 10 + 10].mean()) for k in range(len(bad) // 10 + 1)))
print('energia entre semitons (desafinacao/ruido) media: %.1f%%' % (100 * np.mean([o[1] for o in out])))
for k in (9, 20, 44, 65, 77, 81, 85, 89, 100, 102, 104, 106, 108, 110, 119, 131, 143, 146, 148, 150, 152, 160):
    top = np.argsort(-C[k])[:4]
    print('  t=%3d s: %s' % (k, '  '.join('%s %.0f%%' % (names[p], 100 * C[k][p]) for p in top)))
