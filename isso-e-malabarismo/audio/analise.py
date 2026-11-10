# analise.py — niveis por secao (musica x efeitos), para equilibrar a mixagem
import sys, os, json
sys.path.insert(0, '/home/claude/filme/src')
import numpy as np
SR = 48000
AUD = '/home/claude/filme/audio'
EV = json.load(open(AUD + '/events.json'))
ST = EV['_info']['starts']
m = np.load(AUD + '/stem_mus.npy'); s = np.load(AUD + '/stem_sfx.npy')
names = [k for k in ST if k != '_fim']
names.sort(key=lambda k: ST[k])
def db(x):
    return 20 * np.log10(max(float(x), 1e-9))
print('%-11s %7s %7s | %8s %8s | %8s %8s' % ('plano', 'inicio', 'dur', 'mus rms', 'mus pico', 'sfx rms', 'sfx pico'))
for i, k in enumerate(names):
    a = int(ST[k] * SR); b = int((ST[names[i + 1]] if i + 1 < len(names) else ST['_fim']) * SR)
    mm, ss = m[a:b], s[a:b]
    print('%-11s %7.2f %7.2f | %8.1f %8.1f | %8.1f %8.1f' % (k, ST[k], (b - a) / SR, db(np.sqrt(np.mean(mm ** 2))), db(np.abs(mm).max()),
                                                           db(np.sqrt(np.mean(ss ** 2))), db(np.abs(ss).max())))
# os 12 maiores picos da mistura
x = m + s
pk = np.abs(x).max(1)
win = SR // 2
blocks = pk[:len(pk) // win * win].reshape(-1, win).max(1)
top = np.argsort(-blocks)[:14]
print('maiores picos (t, dB):', ', '.join('%.1fs %.1f' % (i * 0.5, db(blocks[i])) for i in sorted(top)))
