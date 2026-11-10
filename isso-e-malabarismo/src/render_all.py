# render_all.py — renderiza uma lista de planos em 2560x1440, em trechos de ate 3 s (um mp4 por trecho).
# Trechos ja prontos sao pulados: se o processo for interrompido, basta rodar de novo.
import sys, os, time, traceback
os.environ['FILM_H'] = os.environ.get('FILM_H', '1440')
sys.path.insert(0, '/home/claude/filme/src')
import numpy as np
import cv2
import importlib
timeline = importlib.import_module(os.environ.get('FILM_TIMELINE', 'timeline'))     # 'trailer' para o teaser vertical
from core import CFG, FPS
from shots import ffmpeg_writer

CH = 90


def chunks_of(n):
    cs = [(a, min(a + CH, n)) for a in range(0, n, CH)]
    if len(cs) > 1 and cs[-1][1] - cs[-1][0] < 30:          # evita um resto curto demais
        a, b = cs.pop()
        cs[-1] = (cs[-1][0], b)
    return cs


def chunk_path(out, name, k):
    return os.path.join(out, '%s_%03d.mp4' % (name, k))


if __name__ == '__main__':
    names = sys.argv[1].split(',')
    tag = sys.argv[2] if len(sys.argv) > 2 else 'a'
    out = sys.argv[3] if len(sys.argv) > 3 else '/home/claude/filme/out'
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, 'log_%s.txt' % tag), 'a')

    def log(m):
        logf.write(time.strftime('%H:%M:%S ') + m + '\n'); logf.flush()

    log('--- inicio ' + ','.join(names))
    for n in names:
        try:
            sh = timeline.make(n)
            cs = chunks_of(sh.n)
            todo = [k for k in range(len(cs)) if not os.path.exists(chunk_path(out, n, k))]
            if not todo:
                log('pula ' + n); continue
            sh.setup()
            W, H = CFG.W, CFG.H
            t00 = time.time()
            for k in todo:
                a, b = cs[k]
                final = chunk_path(out, n, k)
                tmp = final[:-4] + '.tmp.mp4'
                p = ffmpeg_writer(tmp, W, H, crf=17, preset='medium')
                t0 = time.time()
                for i in range(a, b):
                    fr = sh.frame(i)
                    if fr.shape[0] != H or fr.shape[1] != W:
                        fr = cv2.resize(fr, (W, H))
                    p.stdin.write(np.ascontiguousarray(fr).tobytes())
                p.stdin.close()
                if p.wait() != 0:
                    raise RuntimeError('ffmpeg falhou em %s trecho %d' % (n, k))
                os.replace(tmp, final)
                log('%s trecho %d/%d (quadros %d-%d)  %.2fs/quadro' % (n, k + 1, len(cs), a, b - 1, (time.time() - t0) / (b - a)))
            log('FEITO %s em %.1f min' % (n, (time.time() - t00) / 60))
        except Exception:
            log('ERRO em %s\n%s' % (n, traceback.format_exc()))
    log('FIM')
