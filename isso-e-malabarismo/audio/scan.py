# varre a biblioteca: nota no nome x altura medida
import sys, os, glob, re
sys.path.insert(0, '/home/claude/filme/src')
import numpy as np
import smp
out = []
for f in sorted(glob.glob(smp.ROOT + '/**/*.wav', recursive=True)):
    rel = os.path.relpath(f, smp.ROOT)
    base = os.path.basename(f)[:-4]
    nm = None
    for tok in re.split(r'[_ ]', base):
        m = smp.name_midi(tok)
        if m is not None:
            nm = m; break
    x, sr = smp.read_wav(f)
    pk = float(np.abs(x).max()); dur = len(x) / sr
    f0, conf = smp.detect_f0(x, sr)
    det = 69 + 12 * np.log2(f0 / 440.0) if f0 > 0 else float('nan')
    out.append((rel, nm, det, conf, pk, dur))
import json
json.dump(out, open('/home/claude/filme/audio/scan.json', 'w'))
cur = None
for rel, nm, det, conf, pk, dur in out:
    d = os.path.dirname(rel)
    if d != cur:
        cur = d; print('==', d)
    print('  %-34s nome %s  medido %6.2f  dif %+6.2f  conf %.2f  pico %.2f  %.1fs' % (os.path.basename(rel), nm, det, (det - nm) if nm is not None else float('nan'), conf, pk, dur))
