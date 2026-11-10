# confere afinacao e niveis do sampler: toca notas isoladas e mede a altura
import sys
sys.path.insert(0, '/home/claude/filme/src')
import numpy as np
import smp, synth
from smp import SR
I = synth.load_instruments()
for name, notes in (('piano', [38, 50, 62, 66, 69, 74, 81, 86]), ('felt', [50, 62, 69, 78]), ('harp', [50, 62, 66, 69, 74, 81, 90]),
                    ('glock', [86, 93, 98]), ('vln', [62, 69, 74, 81, 86]), ('vla', [50, 57, 62, 66]), ('vc', [38, 45, 50, 57]),
                    ('cb', [26, 33, 38]), ('solo', [69, 74, 81, 86, 90]), ('horn', [50, 57, 62, 66]), ('pizz', [38, 45, 50, 57]),
                    ('trem', [62, 69, 74, 81])):
    ins = I[name]
    row = []
    for m in notes:
        for vel in (0.3, 0.8):
            y = ins.note(m, 1.6, vel)
            f0, conf = smp.detect_f0(y, SR, t0=0.15, dur=1.0)
            det = 69 + 12 * np.log2(f0 / 440.0) if f0 > 0 else float('nan')
            rms = np.sqrt(np.mean(y[:int(0.8 * SR)] ** 2))
            row.append('%d:%+.2f(%.3f)' % (m, det - m, rms))
    print('%-6s %d amostras | %s' % (name, len(ins.samples), ' '.join(row)))
