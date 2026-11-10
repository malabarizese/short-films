import sys, os, json, wave
sys.path.insert(0, '/home/claude/filme/src')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
import smp
AUD = '/home/claude/filme/audio'
path = sys.argv[1]
x, sr = smp.read_wav(path)
EV = json.load(open(AUD + '/events.json')); ST = EV['_info']['starts']
m = np.load(AUD + '/stem_mus.npy'); s = np.load(AUD + '/stem_sfx.npy')
def rms_db(y, win=0.4):
    n = int(win * sr); k = len(y) // n
    r = np.sqrt((y[:k * n] ** 2).reshape(k, n, 2).mean((1, 2)))
    return np.arange(k) * win, 20 * np.log10(np.maximum(r, 1e-6))
fig, ax = plt.subplots(3, 1, figsize=(20, 11), sharex=True)
t, d = rms_db(x); ax[0].plot(t, d, 'k', lw=1.0, label='master')
g = 10 ** ((np.median(d) - np.median(rms_db(m + s)[1])) / 20)
t2, d2 = rms_db(m); t3, d3 = rms_db(s)
off = np.median(d - rms_db((m + s)[:len(x)])[1][:len(d)])
ax[0].plot(t2, d2 + off, 'C0', lw=0.8, label='musica'); ax[0].plot(t3, d3 + off, 'C1', lw=0.8, label='efeitos')
ax[0].set_ylim(-60, -5); ax[0].legend(loc='lower left'); ax[0].grid(alpha=0.3); ax[0].set_ylabel('RMS 0,4 s (dBFS)')
pk = np.abs(x).max(1); n = int(0.05 * sr); k = len(pk) // n
ax[1].fill_between(np.arange(k) * 0.05, pk[:k * n].reshape(k, n).max(1), color='0.3'); ax[1].set_ylim(0, 1); ax[1].set_ylabel('pico')
f, tt, S = signal.spectrogram(x.mean(1), sr, nperseg=4096, noverlap=2048)
ax[2].pcolormesh(tt, f, 10 * np.log10(S + 1e-12), vmin=-110, vmax=-35, cmap='magma', shading='auto')
ax[2].set_yscale('log'); ax[2].set_ylim(30, 16000); ax[2].set_ylabel('Hz')
for a in ax:
    for kname, v in ST.items():
        if kname != '_fim':
            a.axvline(v, color='c', lw=0.6, alpha=0.8)
for kname, v in ST.items():
    if kname != '_fim':
        ax[0].text(v + 0.2, -8, kname, fontsize=8, rotation=90, va='top')
ax[2].set_xlim(0, ST['_fim']); ax[2].set_xticks(np.arange(0, ST['_fim'], 5))
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=70)
