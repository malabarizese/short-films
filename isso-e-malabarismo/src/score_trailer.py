# score_trailer.py — musica nova para o teaser vertical: mesmos instrumentos do filme, pulso de 120 BPM,
# cortes na batida, e o acorde fica suspenso (sem a terca) quando as bolas sobem: a resolucao so existe no filme.
import sys, os
sys.path.insert(0, '/home/claude/filme/src')
import numpy as np
import synth
from synth import SR, f32
import score as sc
from score import n, play, pno, chord, ostinato, env_pts

DURT = 37.5
sc.mus = synth.Bus(DURT + 3)
sc.sfx = synth.Bus(DURT + 3)
EV = sc.EV
B = 0.5                                   # 120 BPM

# (inicio no teaser, plano do filme, t inicial no plano)
SEG = dict(bola=(0.0, 0.92), tres=(3.5, 1.6), estacoes=(8.0, 4.0), queda=(14.0, 4.3), chuva=(17.0, 1.05), cansaco=(20.0, 2.5),
           torre=(22.0, 1.1), preto=(29.5, 0.0), cartao=(30.0, 0.0))
T_CUT = 29.5


def ev(shot, seg, lo, hi):
    """eventos da coreografia (tempo no teaser, tipo, intensidade) com tempo local do plano entre lo e hi."""
    t0, s0 = SEG[seg]
    return [(t0 + (e[3] - s0), e[1], e[2]) for e in EV[shot] if lo <= e[3] < hi]


def pulse(t0, t1, notes, vel=0.55):
    """baixo em pizzicato a cada tempo (o pulso de 120 BPM)."""
    k = 0
    t = t0
    while t < t1 - 1e-6:
        play('pizz', notes[k % len(notes)], t, 0.9, vel if k % 2 == 0 else vel - 0.12)
        t += B; k += 1


def music():
    # ---- 0-4: a mao ergue a bola contra o sol
    chord('vc', ['D2', 'A2'], 0.0, 3.65, 0.36, att=1.0, rel=0.5, swell=(0.6, 1.3))
    chord('vla', ['F#3', 'A3'], 0.6, 3.65, 0.32, att=0.9, rel=0.5, swell=(0.6, 1.3), gain=0.8)
    for nt in ('D3', 'A3'):
        pno(nt, 0.0, 0.46, 3.5)
    for nt, t in (('F#4', 0.0), ('A4', 1.0), ('D5', 2.0)):
        pno(nt, t, 0.50, 2.5)
    for k, nt in enumerate(['D4', 'F#4', 'A4', 'D5', 'F#5', 'A5', 'D6', 'F#6', 'A6']):       # a bola sobe (3,0 s)
        play('harp', nt, 2.5 + 0.05 * k, 1.6, 0.42 + 0.04 * k)
    play('glock', 'D6', 3.0, 2.0, 0.34)
    play('pizz', 'A2', 3.0, 0.8, 0.5)
    # ---- 4-8: tres bolas (3 notas por segundo)
    ostinato(3.5, 8.0, 3, [(0, 'D'), (6.0, 'G')], vel=0.50, vel1=0.58)
    pulse(3.5, 8.0, ['D2', 'A2', 'D2', 'A2', 'D2', 'G1', 'D2', 'G1', 'D2'])
    chord('vla', ['F#3', 'A3'], 3.5, 6.1, 0.36, att=0.2, gain=0.7); chord('vla', ['G3', 'B3'], 6.0, 8.1, 0.38, att=0.2, gain=0.7)
    for nt, t, d in (('F#5', 3.5, 1.5), ('A5', 5.0, 1.0), ('B5', 6.0, 1.0), ('D6', 7.0, 1.0)):
        play('vln', nt, t, d, 0.52, att=0.08, rel=0.35, gain=1.4)
    # ---- 8-14: tres estacoes (5 notas por segundo)
    cds = [(0, 'Bm'), (10.0, 'Em'), (12.0, 'A')]
    ostinato(8.0, 10.0, 5, cds, vel=0.52, vel1=0.52)
    ostinato(10.0, 12.0, 5, cds, vel=0.36, vel1=0.38, insts=('harp',), gains=(1.0,))
    ostinato(12.0, 14.0, 5, cds, vel=0.52, vel1=0.70)
    ostinato(12.0, 14.0, 5, cds, vel=0.36, vel1=0.50, insts=('harp',), octave=1, gains=(0.6,))
    pulse(8.0, 10.0, ['B1', 'F#2']); pulse(10.0, 12.0, ['E2', 'B2'], vel=0.40); pulse(12.0, 14.0, ['A1', 'E2'], vel=0.60)
    chord('vc', ['B1', 'F#2'], 8.0, 10.1, 0.46, att=0.15); chord('vla', ['D3', 'F#3'], 8.0, 10.1, 0.42, att=0.15, gain=0.8)
    chord('vc', ['E2', 'B2'], 10.0, 12.1, 0.34, att=0.3); chord('trem', ['G4', 'B4'], 10.0, 12.1, 0.32, att=0.4, rel=0.4, gain=0.55)
    chord('vc', ['A1', 'E2'], 12.0, 14.1, 0.50, att=0.15, swell=(0.8, 1.4)); chord('vla', ['E3', 'A3', 'C#4'], 12.0, 14.1, 0.48, att=0.15, swell=(0.8, 1.4), gain=0.8)
    for nt, t, d in (('D5', 8.0, 1.0), ('C#5', 9.0, 1.0), ('B4', 10.0, 2.0), ('E5', 12.0, 0.5), ('F#5', 12.5, 0.5), ('G5', 13.0, 0.5), ('A5', 13.5, 0.5)):
        play('vln', nt, t, d, 0.55 if t < 10 or t >= 12 else 0.36, att=0.07, rel=0.3, gain=1.5)
    for k, (tt, nt) in enumerate(((12.25, 'A5'), (12.75, 'C#6'), (13.25, 'E6'), (13.75, 'A6'))):
        play('glock', nt, tt, 1.2, 0.26 + 0.03 * k)
    play('horn', 'E4', 12.0, 2.0, 0.5, att=0.3, gain=0.65, env=lambda u: 0.7 + 0.6 * u)
    # ---- 14-17: sete bolas... e caem
    thuds = [e[0] for e in ev('sete_falha', 'queda', 4.3, 7.3) if e[1] == 'thud']
    t_th = thuds[0] if thuds else 15.6
    ostinato(14.0, t_th - 0.05, 7, [(0, 'Em7'), (14.8, 'Asus')], vel=0.52, vel1=0.70)
    chord('vc', ['A1', 'E2'], 14.0, t_th + 0.1, 0.52, att=0.15, rel=0.25, swell=(0.9, 1.4))
    chord('vla', ['G3', 'A3', 'D4'], 14.0, t_th + 0.1, 0.50, att=0.15, rel=0.25, swell=(0.9, 1.4), gain=0.8)
    chord('trem', ['E5', 'A5'], 14.3, t_th + 0.1, 0.45, att=0.4, rel=0.2, swell=(0.4, 1.3), gain=0.6)
    pulse(14.0, t_th - 0.2, ['E2', 'B2', 'A1', 'E2'], vel=0.6)
    fall = ['A5', 'G5', 'E5', 'D5', 'B4', 'A4', 'G4', 'E4', 'D4', 'B3', 'A3', 'G3', 'E3', 'D3']
    tt = t_th
    for k, nt in enumerate(fall):
        pno(nt, tt, 0.55 - 0.02 * k, 1.0, gain=0.9)
        tt += 0.055 + 0.010 * k
    chord('vc', ['B1', 'F#2'], t_th + 0.3, 17.05, 0.40, att=0.4, rel=0.3, gain=0.9)
    chord('vla', ['D3', 'F#3'], t_th + 0.3, 17.05, 0.36, att=0.5, rel=0.3, gain=0.8)
    pno('B1', t_th + 0.35, 0.44, 1.2)
    # ---- 17-20: tempestade
    cds = [(0, 'Bm'), (18.0, 'G'), (19.0, 'F#')]
    ostinato(17.0, 20.0, 7, cds, vel=0.66, vel1=0.82, octave=-1, insts=('piano',), gains=(0.58,))
    ostinato(17.0, 20.0, 7, cds, vel=0.50, vel1=0.64, insts=('harp',), gains=(0.8,))
    for a, t in (('B1', 17.0), ('G1', 18.0), ('F#1', 19.0)):
        chord('vc', [a, n(a) + 7], t, t + 1.08, 0.72, att=0.06, rel=0.25)
        chord('cb', [sc.low(a)], t, t + 1.08, 0.7, att=0.06, rel=0.25)
        pno(a, t, 0.8, 1.2, gain=0.7)
    for nts, t in ((['B4', 'D5', 'F#5'], 17.0), (['B4', 'D5', 'G5'], 18.0), (['A#4', 'C#5', 'F#5'], 19.0)):
        chord('trem', nts, t, t + 1.06, 0.72, att=0.06, rel=0.2, gain=0.7, swell=(0.9, 1.2))
    for nt, t, d in (('B4', 17.0, 0.5), ('D5', 17.5, 0.5), ('G5', 18.0, 1.0), ('F#5', 19.0, 1.0)):
        play('vln', nt, t, d, 0.80, att=0.05, rel=0.25, gain=1.4); play('vln', n(nt) + 12, t, d, 0.74, att=0.05, rel=0.25, gain=1.0)
        play('horn', n(nt) - 12, t, d, 0.75, att=0.05, rel=0.25, gain=1.0)
    # ---- 20-22: ele se levanta
    chord('vc', ['A1', 'E2'], 20.05, 22.05, 0.34, att=0.5, rel=0.2, swell=(0.6, 1.6))
    chord('vla', ['A3', 'D4', 'E4'], 20.05, 22.05, 0.30, att=0.5, rel=0.2, swell=(0.6, 1.7), gain=0.8)
    chord('vln', ['E5'], 20.3, 22.05, 0.30, att=0.6, rel=0.2, swell=(0.5, 1.7), gain=0.7)
    for nt, t, v in (('E5', 20.0, .40), ('A4', 20.5, .42), ('C#5', 21.0, .48), ('E5', 21.5, .54)):
        pno(nt, t, v, min(1.6, 22.0 - t), rel=0.25)
    pno('A2', 20.0, 0.40, 1.9, rel=0.25)
    chord('trem', ['E5', 'A5', 'C#6'], 21.0, 23.0, 0.66, att=0.6, rel=0.12, swell=(0.25, 1.5), gain=0.7)
    # ---- 22-23: a cascata de 7 (7 notas por segundo) -> o lancamento
    T_thr = [e[0] for e in ev('torre', 'torre', 2.0, 2.3) if e[1] == 'throw'][0]
    ostinato(22.0, T_thr - 0.02, 7, [(0, 'A')], vel=0.66, vel1=0.82, gains=(0.8, 0.65))
    chord('vc', ['A1', 'E2'], 22.0, T_thr + 0.05, 0.72, att=0.08, rel=0.2, swell=(1.0, 1.3))
    chord('cb', [sc.low('A0')], 22.0, T_thr + 0.05, 0.7, att=0.08, rel=0.2)
    chord('vla', ['A3', 'C#4', 'E4', 'G4'], 22.0, T_thr + 0.05, 0.70, att=0.08, rel=0.2, swell=(1.0, 1.3), gain=0.8)
    play('vln', 'A5', 22.0, T_thr - 22.0, 0.80, att=0.06, rel=0.25, gain=1.6)
    play('horn', 'E4', 22.0, T_thr - 22.0, 0.75, att=0.08, rel=0.2, gain=0.8)
    pulse(22.0, T_thr - 0.1, ['A1', 'E2'], vel=0.66)
    # ---- o truque em camera lenta: re-mi-la, suspenso
    throws = [e[0] for e in ev('torre', 'torre', 2.0, 8.6) if e[1] == 'throw']
    up = ['D4', 'A4', 'D5', 'E5', 'A5', 'D6', 'E6']
    for k, (t, nt) in enumerate(zip(throws, up)):
        pno(nt, t, 0.52 + 0.02 * k, min(5.0, T_CUT - 0.03 - t), send=0.55, rel=0.1)
        if k >= 4:
            play('glock', nt, t, 2.0, 0.20, gain=0.8)
    chord('vln', ['A5', 'E6'], T_thr + 0.5, T_CUT, 0.26, att=2.0, rel=0.06, gain=0.55)
    chord('vla', ['D4', 'E4', 'A4'], T_thr + 1.2, T_CUT, 0.24, att=2.0, rel=0.06, gain=0.55)
    chord('vc', ['D2', 'A2'], T_thr + 0.3, T_CUT, 0.26, att=1.6, rel=0.06, gain=0.7)
    chord('trem', ['A4', 'D5', 'E5', 'G5'], 27.4, T_CUT, 0.60, att=1.6, rel=0.05, swell=(0.12, 1.5), gain=0.7)
    # ---- 30-37.5: cartao — o monstrinho olha para cima; acorde aberto, sem resolver
    T0 = 30.0
    play('pizz', 'D3', T0 + 0.32, 0.8, 0.5)
    for k, nt in enumerate(['D5', 'E5', 'A5', 'D6']):
        play('glock', nt, T0 + 1.10 + 0.13 * k, 2.2, 0.38 + 0.03 * k)
        play('harp', nt, T0 + 1.10 + 0.13 * k, 2.2, 0.52)
    tc = T0 + 1.75
    for nt in ('D2', 'A2', 'D3'):
        play('pizz', nt, tc, 1.5, 0.66)
    for nt in ('D4', 'A4', 'E5'):
        play('harp', nt, tc, 3.0, 0.58)
    play('glock', 'A5', tc, 3.0, 0.42); play('glock', 'E6', tc + 0.02, 3.0, 0.36)
    for nt in ('D3', 'A3', 'E4'):
        pno(nt, tc, 0.46, 4.0)
    pno('A4', T0 + 2.5, 0.36, 3.5, send=0.45); pno('D5', T0 + 3.1, 0.34, 3.5, send=0.45); pno('E5', T0 + 4.3, 0.30, 3.0, send=0.5)
    chord('vc', ['D2', 'A2'], tc, T0 + 6.6, 0.22, att=1.0, rel=1.2, swell=(1.0, 0.3), gain=0.6)


def effects():
    r = np.random.default_rng(31)
    sfx = sc.sfx
    w = synth.wind(DURT + 1, r, gust=1.0)
    g = env_pts([(0, 0.0), (0.6, 0.8), (8, 0.6), (14, 0.7), (16.9, 0.9), (17.1, 1.6), (19.9, 1.6), (20.1, 0.6), (22, 0.5),
                 (23.1, 0.35), (T_CUT - 0.05, 0.35), (T_CUT + 0.05, 0.0), (DURT + 1, 0.0)], len(w))
    sfx.add(w * g[:, None], 0.0, gain=0.30, send=0.05)
    # bola
    sfx.add(synth.sfx_rustle(r, 0.20, 0.5, 900, 4000), 0.05, gain=0.25, send=0.1)
    sfx.add(synth.sfx_throw(r, 0.9), 2.5, gain=0.5, send=0.15)
    sfx.add(synth.whoosh(0.7, r, 400, 2600), 2.43, gain=0.35, send=0.3)

    def juggle(evs, g_catch, g_throw, pan=0.0):
        for (t, typ, inten) in evs:
            if typ == 'catch':
                sfx.add(synth.sfx_catch(r, inten), t, gain=g_catch, pan=pan + float(r.uniform(-0.12, 0.12)), send=0.10)
            elif typ == 'throw':
                sfx.add(synth.sfx_throw(r, inten), t, gain=g_throw, pan=pan, send=0.10)
            elif typ == 'thud':
                sfx.add(synth.sfx_thud(r, inten), t, gain=g_catch * 1.4, pan=pan + float(r.uniform(-0.3, 0.3)), send=0.12)
    juggle(ev('tres', 'tres', 1.6, 6.1), 0.30, 0.10)
    juggle(ev('sete_falha', 'queda', 4.3, 7.3), 0.26, 0.07)
    juggle(ev('torre', 'torre', 1.1, 2.1), 0.20, 0.06)
    for (t, typ, inten) in ev('torre', 'torre', 2.1, 8.6):                  # camera lenta
        if typ == 'throw':
            sfx.add(synth.repitch(synth.sfx_throw(r, 1.0), -7), t, gain=0.20, send=0.5)
        elif typ == 'catch':
            sfx.add(synth.repitch(synth.sfx_catch(r, 1.0), -5), t, gain=0.5, send=0.45)
    # viradas de estacao
    for t in (8.0, 10.0, 12.0, 17.0):
        sfx.add(synth.whoosh(0.6, r, 300, 3600), t - 0.28, gain=0.28, send=0.35)
    # chuva e raio
    rn = synth.rain(3.6, r)
    g = env_pts([(16.7, 0.0), (17.0, 1.2), (19.95, 1.3), (20.05, 0.0)], len(rn), 16.7)
    sfx.add(rn * g[:, None], 16.7, gain=0.40, send=0.05)
    sfx.add(synth.thunder(r, 3.2, crack=0.6), 18.12, gain=0.55, pan=-0.1, send=0.2)
    # respiracao
    for k in range(2):
        nb = int(0.5 * SR)
        y = synth.bp(synth.white(nb, r), 500 if k % 2 == 0 else 700, 2400) * synth.bell(nb, 1.4)[:, None]
        sfx.add(y, 20.15 + k * 0.95, gain=0.035, pan=0.2, send=0.15)
    # coracao enquanto as bolas sobem; a ultima batida ja no preto
    for t in (24.25, 25.3, 26.35, 27.35, 28.25, 29.05, 29.62):
        sfx.add(synth.heartbeat(r), t, gain=0.30 if t < 28 else 0.42, send=0.08)
    # cartao
    s, f = synth.sweep(320, 980, 0.07)
    sfx.add(synth.st(s * synth.bell(len(s), 0.8)), 30.31, gain=0.18, send=0.2)


def build(out_wav):
    music(); effects()
    ir = synth.make_ir()
    nout = int(DURT * SR)
    m = sc.mus.dry * 0.86 + synth.reverb(sc.mus.send, ir) * 0.62
    s = sc.sfx.dry + synth.reverb(sc.sfx.send, ir) * 0.55
    # equilibrio por trecho (dB) e corte seco no preto
    pts = [(0, 3.0), (3.4, 3.0), (3.6, 1.5), (7.9, 1.5), (8.1, 0.5), (13.9, 0.5), (14.1, 0.5), (16.9, 1.0), (17.1, -2.0),
           (19.9, -2.0), (20.1, 3.0), (21.9, 1.0), (22.1, -0.5), (23.0, -0.5), (23.2, 3.0), (T_CUT - 0.02, 3.0), (T_CUT + 0.03, -60.0),
           (29.95, -60.0), (30.0, 1.0), (DURT + 5, 1.0)]
    ga = env_pts([(t, 10 ** (d / 20)) for t, d in pts], len(m))
    # no preto fica so a cauda da sala: silencia o som direto, deixa a reverberacao
    dry_m = sc.mus.dry * 0.86 * ga[:, None]
    wet_m = synth.reverb(sc.mus.send * ga[:, None], ir) * 0.62
    gs = env_pts([(0, 1.0), (T_CUT - 0.02, 1.0), (T_CUT + 0.03, 0.0), (29.5, 0.0), (29.55, 1.0), (DURT + 5, 1.0)], len(s))
    mix = dry_m + wet_m + s * gs[:, None]
    mix = synth.hp(mix, 28, 2)
    mix = mix + 0.45 * synth.hp(mix, 5200, 1)
    mix = mix[:nout]
    nf = int(0.5 * SR)
    mix[-nf:] *= np.linspace(1, 0, nf)[:, None].astype(f32) ** 1.5
    return mix


if __name__ == '__main__':
    out = sys.argv[1]
    mix = build(out)
    print('pico bruto %.3f' % np.abs(mix).max())
    print(sc.master(mix, out))
