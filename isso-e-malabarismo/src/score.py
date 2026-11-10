# score.py — trilha original e efeitos sonoros do filme, sincronizados com os cortes e com as coreografias
# 60 BPM (1 tempo = 1 s). Ostinato com 3, 5 e 7 notas por tempo = numero de bolas em cena.
import sys, os, json, subprocess
sys.path.insert(0, '/home/claude/filme/src')
import numpy as np
import synth
from synth import SR, f32

AUD = '/home/claude/filme/audio'
EV = json.load(open(os.path.join(AUD, 'events.json')))
ST = EV['_info']['starts']
DUR = ST['_fim']
I = synth.load_instruments()
I['piano'].layer_edges = (-1.0, 0.72)        # camada pp tem ruido de fundo: usa mf filtrado nas notas suaves

mus = synth.Bus(DUR + 3)
sfx = synth.Bus(DUR + 3)
R = np.random.default_rng(2026)

NN = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def n(s):
    """'F#4' -> midi."""
    nm, octv = s[:-1], int(s[-1])
    return 12 * (octv + 1) + NN[nm[0]] + (1 if '#' in nm else 0) - (1 if 'b' in nm else 0)


PAN = dict(piano=-0.08, harp=0.38, glock=0.22, vln=-0.38, trem=-0.30, vla=0.12, vc=0.30, cb=0.42, solo=-0.18, horn=-0.24,
           pizz=0.28, felt=0.0)
SEND = dict(piano=0.30, harp=0.34, glock=0.45, vln=0.42, trem=0.42, vla=0.40, vc=0.36, cb=0.25, solo=0.50, horn=0.50,
            pizz=0.30, felt=0.3)
GAIN = dict(piano=1.0, harp=0.85, glock=0.42, vln=0.80, trem=0.60, vla=0.75, vc=0.85, cb=0.9, solo=0.75, horn=0.70,
            pizz=0.9, felt=1.0)


def hum(t, v, jt=0.006, jv=0.07):
    return t + float(R.normal(0, jt)), float(np.clip(v * (1 + R.normal(0, jv)), 0.03, 1.0))


def play(inst, note, t, dur, vel, gain=1.0, pan=None, send=None, env=None, human=True, **kw):
    m = n(note) if isinstance(note, str) else note
    if human:
        t, vel = hum(t, vel)
    y = I[inst].note(m, dur, vel, **kw)
    if inst == 'piano' and vel < 0.62:
        y = synth.lp(y, 2500 + 16000 * vel ** 1.2, 1)          # toque suave = timbre mais escuro
    if env is not None:
        u = np.linspace(0, 1, len(y)).astype(f32)
        y = y * env(u)[:, None].astype(f32)
    mus.add(y, t, GAIN[inst] * gain, PAN[inst] if pan is None else pan, SEND[inst] if send is None else send)


def low(note, floor=28):
    m = n(note) if isinstance(note, str) else note
    while m < floor:
        m += 12
    return m


def pno(note, t, vel=0.45, dur=2.5, **kw):
    play('piano', note, t, dur, vel, **kw)


def chord(inst, notes, t0, t1, vel, att=0.6, rel=0.9, swell=None, gain=1.0, **kw):
    """acorde sustentado. swell=(g0, g1): ganho no inicio e no fim (crescendo/decrescendo)."""
    env = None
    if swell is not None:
        g0, g1 = swell
        env = lambda u: g0 + (g1 - g0) * u
    for nt in notes:
        play(inst, nt, t0, t1 - t0, vel, att=att, rel=rel, env=env, gain=gain, human=False, **kw)


def arc(ch, per):
    """desenho do ostinato para 1 tempo (ou 2, nas tercinas): sobe e desce como a bola."""
    r, t3, f5, o8, t10 = ch
    if per == 3:
        return [r, f5, o8, t10, o8, f5]                 # 2 tempos
    if per == 5:
        return [r, t3, f5, o8, f5, t3, f5, o8, t10, o8]  # 2 tempos
    return [r, t3, f5, o8, t10, o8, f5]                 # 7 por tempo


CH = {  # fundamental, terca, quinta, oitava, decima
    'D': [62, 66, 69, 74, 78], 'Bm': [59, 62, 66, 71, 74], 'G': [55, 59, 62, 67, 71], 'A': [57, 61, 64, 69, 73],
    'Em': [52, 55, 59, 64, 67], 'F#m': [54, 57, 61, 66, 69], 'F#': [54, 58, 61, 66, 70], 'Asus': [57, 62, 64, 69, 74],
    'D/F#': [54, 57, 62, 66, 69], 'Em7': [52, 55, 59, 62, 67], 'Dsus': [62, 64, 69, 74, 76],
}


def ostinato(t0, t1, per, chords, vel=0.5, vel1=None, insts=('piano', 'harp'), octave=0, accent=0.16, gains=(1.0, 0.8),
             dur=0.9, skip=None):
    """chords: lista (t_inicio, nome). per = notas por tempo. vel -> vel1 em rampa."""
    vel1 = vel if vel1 is None else vel1
    step = 1.0 / per
    k = 0
    t = t0
    while t < t1 - 1e-6:
        name = [c for (tc, c) in chords if tc <= t + 1e-6][-1]
        pat = arc(CH[name], per)
        beat_pos = (t - t0)
        idx = int(round(beat_pos * per)) % len(pat)
        u = (t - t0) / max(t1 - t0, 1e-6)
        v = vel + (vel1 - vel) * u
        if abs(beat_pos - round(beat_pos)) < 1e-6:
            v = v * (1 + accent)
        if skip is None or not skip(t):
            for ins, g in zip(insts, gains):
                play(ins, pat[idx] + 12 * octave, t, dur, min(v, 1.0), gain=g)
        k += 1
        t = t0 + k * step


# =====================================================================================================
#                                              MUSICA
# =====================================================================================================
def music():
    T = ST
    # ---------------- A. abertura (0-8): alvorada, quintas abertas, sem a terca
    chord('vc', ['D2', 'A2'], 0.3, 8.6, 0.30, att=3.0, rel=1.5, gain=0.8)
    chord('vla', ['A3'], 2.0, 8.6, 0.25, att=2.5, rel=1.5, gain=0.7)
    play('solo', 'A5', 3.2, 4.9, 0.25, att=2.2, rel=1.5, gain=0.5)
    pno('A4', 4.0, 0.30, 4.0); pno('D5', 6.0, 0.30, 4.0); pno('E5', 7.0, 0.28, 4.0)

    # ---------------- B. subida (8-16): o tema, so no piano
    mel = [('F#4', 8.0), ('A4', 9.0), ('D5', 10.0), ('C#5', 11.0), ('B4', 12.0), ('D5', 13.0), ('A4', 14.0)]
    for nt, t in mel:
        pno(nt, t, 0.50, 2.2)
    for nt, t in (('D2', 8.0), ('A2', 8.0), ('B1', 12.0), ('F#2', 12.0), ('G1', 14.0), ('D2', 14.0)):
        pno(nt, t, 0.38, 4.0)
    for nt, t in (('A3', 8.5), ('D4', 9.5), ('A3', 10.5), ('D4', 11.5), ('F#3', 12.5), ('B3', 13.5), ('G3', 14.5), ('B3', 15.5)):
        pno(nt, t, 0.28, 1.6)
    pno('G4', 15.0, 0.36, 2.0)
    chord('vc', ['D2'], 8.0, 12.2, 0.28, att=1.0, gain=0.7); chord('vc', ['B1'], 12.0, 14.2, 0.28, att=0.8, gain=0.7)
    chord('vc', ['G1'], 14.0, 16.2, 0.30, att=0.8, gain=0.7)
    chord('vla', ['F#3', 'A3'], 8.0, 12.2, 0.25, att=1.2, gain=0.6); chord('vla', ['F#3', 'B3'], 12.0, 14.2, 0.25, att=0.8, gain=0.6)
    chord('vla', ['G3', 'B3'], 14.0, 16.2, 0.27, att=0.8, gain=0.6)

    # ---------------- C. chegada (16-24): o sol nasce — cordas aquecem
    mel = [('F#4', 16.0, 0.52), ('A4', 17.0, 0.54), ('E5', 18.0, 0.58), ('C#5', 19.0, 0.55), ('D5', 20.0, 0.62),
           ('F#5', 21.0, 0.66), ('E5', 22.0, 0.60), ('D5', 23.0, 0.55)]
    for nt, t, v in mel:
        pno(nt, t, v, 2.4)
    for nt, t in (('D2', 16.0), ('A2', 16.0), ('C#2', 18.0), ('A2', 18.0), ('B1', 20.0), ('F#2', 20.0), ('G1', 22.0), ('D2', 22.0)):
        pno(nt, t, 0.42, 2.05, rel=0.3)
    for nt, t in (('A3', 16.5), ('D4', 17.5), ('E3', 18.5), ('A3', 19.5), ('F#3', 20.5), ('B3', 21.5), ('G3', 22.5), ('B3', 23.5)):
        pno(nt, t, 0.30, 1.6)
    chord('vc', ['D2', 'A2'], 16.0, 18.15, 0.40, att=0.7, swell=(0.7, 1.0)); chord('vc', ['C#2', 'A2'], 18.0, 20.15, 0.42, att=0.5)
    chord('vc', ['B1', 'F#2'], 20.0, 22.15, 0.48, att=0.5, swell=(1.0, 1.2)); chord('vc', ['G1', 'D2'], 22.0, 24.3, 0.48, att=0.5)
    chord('vla', ['F#3', 'A3'], 16.0, 18.15, 0.36, att=0.7); chord('vla', ['E3', 'A3'], 18.0, 20.15, 0.38, att=0.5)
    chord('vla', ['F#3', 'B3'], 20.0, 22.15, 0.44, att=0.5); chord('vla', ['G3', 'B3'], 22.0, 24.3, 0.44, att=0.5)
    chord('vln', ['D5', 'F#5'], 20.0, 22.15, 0.40, att=1.0, swell=(0.6, 1.1), gain=0.8)
    chord('vln', ['D5', 'G5'], 22.0, 24.3, 0.42, att=0.5, swell=(1.1, 0.8), gain=0.8)
    play('horn', 'A3', 20.0, 4.0, 0.35, att=1.2, rel=1.2, gain=0.6)

    # ---------------- D. olho (24-29): suspenso
    chord('vc', ['G1', 'D2'], 24.0, 27.2, 0.30, att=0.5, rel=1.2, gain=0.8)
    chord('vla', ['A3', 'B3'], 24.0, 27.2, 0.26, att=0.8, rel=1.2, gain=0.7)
    chord('vln', ['D5'], 24.0, 29.2, 0.25, att=1.0, rel=1.3, gain=0.6)
    chord('vc', ['A1', 'E2'], 27.0, 29.3, 0.30, att=0.6, rel=1.2, gain=0.8)
    chord('vla', ['G3', 'A3'], 27.0, 29.3, 0.26, att=0.6, rel=1.2, gain=0.7)
    pno('G4', 24.0, 0.34, 3.5); pno('B4', 24.0, 0.34, 3.5); pno('A4', 25.5, 0.30, 3.0)
    play('glock', 'D6', 26.95, 2.5, 0.22)                       # a piscada
    pno('E4', 27.0, 0.32, 3.0); pno('A4', 27.0, 0.32, 3.0); pno('D5', 28.0, 0.30, 2.5)

    # ---------------- E. orvalho + neblina (29-36): harpa e brilhos
    em9 = ['E3', 'B3', 'F#4', 'G4', 'B4', 'D5', 'F#5', 'B5']
    t = 29.0; k = 0
    seq = em9 + em9[-2:0:-1]
    while t < 32.5 - 1e-6:
        play('harp', seq[k % len(seq)], t, 1.8, 0.42 + 0.10 * np.sin(k * 0.45), gain=0.9)
        t += 0.25; k += 1
    a7s = ['A2', 'E3', 'A3', 'D4', 'E4', 'G4', 'A4', 'D5', 'E5', 'A5']
    seq = a7s + a7s[-2:0:-1]
    t = 32.5; k = 0
    while t < 36.0 - 1e-6:
        u = (t - 32.5) / 3.5
        play('harp', seq[k % len(seq)], t, 1.8, 0.42 + 0.22 * u, gain=0.9)
        t += 0.25; k += 1
    for tt, nt in ((29.35, 'B5'), (29.9, 'F#6'), (30.55, 'D6'), (31.05, 'A6'), (31.6, 'B5'), (32.1, 'F#6')):
        play('glock', nt, tt, 2.0, 0.20 + 0.1 * R.random())
    chord('vla', ['G3', 'B3'], 29.0, 32.65, 0.28, att=0.7, rel=1.0, gain=0.7)
    chord('vc', ['E2'], 29.0, 32.65, 0.30, att=0.7, rel=1.0, gain=0.8)
    chord('vla', ['G3', 'A3'], 32.5, 36.15, 0.32, att=0.5, swell=(0.8, 1.3), gain=0.7)
    chord('vc', ['A1', 'E2'], 32.5, 36.15, 0.34, att=0.5, swell=(0.8, 1.3), gain=0.8)
    chord('trem', ['A4', 'E5'], 33.0, 36.1, 0.30, att=1.0, rel=0.5, swell=(0.3, 1.2), gain=0.5)

    # ---------------- F. bola (36-41): a bola encobre o sol — re maior chega
    for nt in ('D2', 'A2', 'D3', 'A3', 'F#4'):
        pno(nt, 36.0, 0.55, 5.0)
    chord('vc', ['D2', 'A2'], 36.0, 41.1, 0.36, att=0.4, rel=1.0, gain=0.8)
    chord('vla', ['F#3', 'A3'], 36.0, 41.1, 0.32, att=0.6, rel=1.0, gain=0.7)
    chord('vln', ['A4', 'D5'], 36.0, 39.5, 0.30, att=1.0, rel=1.2, gain=0.6)
    pno('A4', 37.0, 0.42, 3.0); pno('D5', 38.0, 0.45, 3.0); pno('E5', 39.0, 0.42, 2.0)
    gl = ['D4', 'F#4', 'A4', 'D5', 'F#5', 'A5', 'D6', 'F#6', 'A6']                  # a bola sobe
    for k, nt in enumerate(gl):
        play('harp', nt, ST['bola'] + 3.42 + 0.055 * k, 2.0, 0.40 + 0.04 * k)
    play('glock', 'D6', ST['bola'] + 3.42 + 0.50, 2.5, 0.35)

    # ---------------- G. tres (41-49): tercinas = 3 bolas
    T0 = ST['tres']
    thud3 = [e[0] for e in EV['tres'] if e[1] == 'thud'][0]
    play('pizz', 'D2', T0, 1.5, 0.6); pno('D3', T0, 0.45, 2.0); pno('A3', T0, 0.40, 2.0)
    ostinato(T0 + 1.0, T0 + 7.0, 3, [(0, 'D'), (T0 + 4, 'Bm'), (T0 + 6, 'G')], vel=0.42, vel1=0.52)
    for nt, t in (('D2', 2), ('A2', 3), ('B1', 4), ('F#2', 5), ('G1', 6)):
        play('pizz', nt, T0 + t, 1.5, 0.55)
    chord('vla', ['F#3', 'A3'], T0 + 1, T0 + 4.1, 0.28, att=0.6, gain=0.6); chord('vla', ['F#3', 'B3'], T0 + 4, T0 + 6.1, 0.28, att=0.4, gain=0.6)
    chord('vla', ['G3', 'B3'], T0 + 6, T0 + 7.2, 0.28, att=0.4, rel=0.4, gain=0.6)
    play('solo', 'A4', T0 + 2.0, 2.0, 0.3, att=0.5, gain=0.55); play('solo', 'B4', T0 + 4.0, 2.0, 0.3, att=0.3, gain=0.55)
    play('solo', 'D5', T0 + 6.0, 1.0, 0.3, att=0.3, rel=0.5, gain=0.55)
    play('pizz', 'G2', thud3, 1.2, 0.45)                                           # a bola cai: um "plof"

    # ---------------- H. tl1 (49-63): treino de 3 bolas, o dia passa
    T0 = ST['tl1']
    ostinato(T0, T0 + 12.0, 3, [(0, 'D'), (T0 + 4, 'Bm'), (T0 + 8, 'G'), (T0 + 10, 'A')], vel=0.50, vel1=0.60)
    for k in range(12):
        nt = ['D2', 'A2', 'D2', 'A2', 'B1', 'F#2', 'B1', 'F#2', 'G1', 'D2', 'A1', 'E2'][k]
        play('pizz', nt, T0 + k, 1.4, 0.6 if k % 2 == 0 else 0.45)
    chord('vc', ['D2', 'A2'], T0, T0 + 4.1, 0.42, att=0.3); chord('vc', ['B1', 'F#2'], T0 + 4, T0 + 8.1, 0.42, att=0.3)
    chord('vc', ['G1', 'D2'], T0 + 8, T0 + 10.1, 0.44, att=0.3); chord('vc', ['A1', 'E2'], T0 + 10, T0 + 14.2, 0.42, att=0.3, rel=1.5, swell=(1.0, 0.5))
    chord('vla', ['F#3', 'A3'], T0, T0 + 4.1, 0.40, att=0.3); chord('vla', ['F#3', 'B3'], T0 + 4, T0 + 8.1, 0.40, att=0.3)
    chord('vla', ['G3', 'B3'], T0 + 8, T0 + 10.1, 0.42, att=0.3); chord('vla', ['E3', 'A3'], T0 + 10, T0 + 14.2, 0.40, att=0.3, rel=1.5, swell=(1.0, 0.5))
    mel = [('A4', 0, 2), ('D5', 2, 2), ('F#5', 4, 3), ('E5', 7, 1), ('D5', 8, 2), ('C#5', 10, 2)]
    for nt, t, d in mel:
        play('vln', nt, T0 + t, d, 0.45, att=0.12, rel=0.5, gain=1.35)
        pno(nt, T0 + t, 0.50, 2.0)
    # noite: as estrelas
    pno('E5', T0 + 12.0, 0.36, 3.0); pno('D5', T0 + 13.0, 0.33, 3.0)
    play('glock', 'A6', T0 + 12.5, 2.0, 0.18); play('glock', 'E6', T0 + 13.3, 2.0, 0.16)
    chord('vln', ['A5'], T0 + 11.5, T0 + 14.1, 0.22, att=1.0, rel=0.8, gain=0.5)

    # ---------------- I. cinco (63-75): quintinas = 5 bolas
    T0 = ST['cinco']
    t_sync = T0 + 0.7 + 28 * 0.215                                               # entra o (6x,4)*
    cds = [(0, 'Bm'), (T0 + 4, 'G'), (T0 + 8, 'D/F#'), (T0 + 10, 'A')]
    ostinato(T0 + 0.6, t_sync, 5, cds, vel=0.44, vel1=0.52)
    ostinato(np.ceil(t_sync * 5) / 5, T0 + 12.0, 5, cds, vel=0.54, vel1=0.64, gains=(1.0, 0.9))
    ostinato(np.ceil(t_sync * 5) / 5, T0 + 12.0, 5, cds, vel=0.34, vel1=0.46, insts=('harp',), octave=1, gains=(0.6,))
    play('pizz', 'B1', T0, 1.5, 0.6)
    for k, nt in enumerate(['B1', 'F#2', 'B1', 'F#2', 'G1', 'D2', 'G1', 'D2', 'F#2', 'D2', 'A1', 'E2']):
        if k > 0:
            play('pizz', nt, T0 + k, 1.4, 0.58 if k % 2 == 0 else 0.44)
    chord('vc', ['B1', 'F#2'], T0, T0 + 4.1, 0.42, att=0.4); chord('vc', ['G1', 'D2'], T0 + 4, T0 + 8.1, 0.44, att=0.3)
    chord('vc', ['F#2', 'D3'], T0 + 8, T0 + 10.1, 0.48, att=0.3); chord('vc', ['A1', 'E2'], T0 + 10, T0 + 12.1, 0.50, att=0.3, swell=(1.0, 1.3))
    chord('vla', ['D3', 'F#3'], T0, T0 + 4.1, 0.38, att=0.4); chord('vla', ['D3', 'G3'], T0 + 4, T0 + 8.1, 0.40, att=0.3)
    chord('vla', ['D3', 'A3'], T0 + 8, T0 + 10.1, 0.44, att=0.3); chord('vla', ['E3', 'A3'], T0 + 10, T0 + 12.1, 0.46, att=0.3, swell=(1.0, 1.3))
    mel = [('D5', 0, 2, .45), ('F#5', 2, 2, .48), ('G5', 4, 3, .52), ('B5', 7, 1, .55), ('A5', 8, 2, .58), ('E5', 10, 1, .58), ('C#5', 11, 1, .6)]
    for nt, t, d, v in mel:
        play('vln', nt, T0 + t, d, v, att=0.12, rel=0.5, gain=1.6)
    play('horn', 'D4', T0 + 8, 2.0, 0.4, att=0.3, gain=0.6); play('horn', 'E4', T0 + 10, 2.0, 0.45, att=0.3, gain=0.6)

    # ---------------- J. tl2 (75-91): quatro estacoes
    T0 = ST['tl2']
    # verao: re maior cheio
    ostinato(T0, T0 + 4, 5, [(0, 'D')], vel=0.62, vel1=0.62)
    ostinato(T0, T0 + 4, 5, [(0, 'D')], vel=0.42, insts=('harp',), octave=1, gains=(0.6,))
    chord('vc', ['D2', 'A2'], T0, T0 + 4.1, 0.58, att=0.2); chord('vla', ['F#3', 'A3', 'D4'], T0, T0 + 4.1, 0.55, att=0.2)
    chord('cb', [low('D1')], T0, T0 + 4.1, 0.5, att=0.2)
    for nt, t, d in (('F#5', 0, 2), ('A5', 2, 2)):
        play('vln', nt, T0 + t, d, 0.62, att=0.1, gain=1.5); play('vln', n(nt) - 12, T0 + t, d, 0.55, att=0.1, gain=1.0)
    play('horn', 'D4', T0, 4.0, 0.5, att=0.2, gain=0.65)
    for k in range(4):
        play('pizz', ['D2', 'A2', 'D2', 'A2'][k], T0 + k, 1.3, 0.62)
    # outono: si menor, linha que desce
    ostinato(T0 + 4, T0 + 8, 5, [(0, 'Bm'), (T0 + 6, 'F#m')], vel=0.44, vel1=0.38, insts=('harp',), gains=(1.0,))
    chord('vc', ['B1', 'F#2'], T0 + 4, T0 + 6.1, 0.42, att=0.3); chord('vc', ['F#1', 'C#2'], T0 + 6, T0 + 8.1, 0.40, att=0.3)
    chord('vln', ['B3', 'D4'], T0 + 4, T0 + 6.1, 0.36, att=0.3, gain=0.7); chord('vln', ['A3', 'C#4'], T0 + 6, T0 + 8.1, 0.34, att=0.3, gain=0.7)
    for nt, t in (('D5', 4), ('C#5', 5), ('B4', 6), ('A4', 7)):
        play('vla', nt, T0 + t, 1.0, 0.50, att=0.1, rel=0.5); pno(nt, T0 + t, 0.46, 1.8)
    for k in range(4):
        play('pizz', ['B1', 'F#2', 'F#1', 'C#2'][k], T0 + 4 + k, 1.3, 0.5)
    # inverno: mi menor, ralo e frio
    ostinato(T0 + 8, T0 + 12, 5, [(0, 'Em')], vel=0.34, vel1=0.34, insts=('piano',), octave=1, gains=(0.8,), dur=0.35)
    chord('cb', [low('E1')], T0 + 8, T0 + 12.1, 0.38, att=0.5); chord('vc', ['E2', 'B2'], T0 + 8, T0 + 12.1, 0.32, att=0.5)
    chord('trem', ['G4', 'B4'], T0 + 8, T0 + 12.1, 0.32, att=0.6, rel=0.5, gain=0.55)
    play('solo', 'B5', T0 + 8.5, 3.2, 0.3, att=0.8, rel=0.8, gain=0.5)
    # primavera: o ipe floresce — dominante, crescendo
    ostinato(T0 + 12, T0 + 16, 5, [(0, 'A')], vel=0.50, vel1=0.70)
    ostinato(T0 + 12, T0 + 16, 5, [(0, 'A')], vel=0.36, vel1=0.52, insts=('harp',), octave=1, gains=(0.7,))
    chord('vc', ['A1', 'E2'], T0 + 12, T0 + 16.1, 0.50, att=0.3, swell=(0.8, 1.4)); chord('vla', ['E3', 'A3', 'C#4'], T0 + 12, T0 + 16.1, 0.48, att=0.3, swell=(0.8, 1.4))
    chord('cb', [low('A1')], T0 + 12, T0 + 16.1, 0.45, att=0.3)
    for nt, t in (('E5', 12), ('F#5', 13), ('G5', 14), ('A5', 15)):
        play('vln', nt, T0 + t, 1.0, 0.55 + 0.04 * (t - 12), att=0.08, rel=0.4, gain=1.5)
    play('horn', 'E4', T0 + 12, 4.0, 0.5, att=0.4, gain=0.65, env=lambda u: 0.7 + 0.6 * u)
    for k, (tt, nt) in enumerate(((12.4, 'A5'), (13.0, 'C#6'), (13.6, 'E6'), (14.2, 'A6'), (14.8, 'C#6'), (15.3, 'E6'), (15.7, 'A6'))):
        play('glock', nt, T0 + tt, 1.5, 0.25 + 0.03 * k)
    for k in range(4):
        play('pizz', ['A1', 'E2', 'A1', 'E2'][k], T0 + 12 + k, 1.3, 0.6)

    # ---------------- K. sete_falha (91-99): sete bolas... e caem
    T0 = ST['sete_falha']
    thuds = [e[0] for e in EV['sete_falha'] if e[1] == 'thud']
    t_th = thuds[0]
    chord('vc', ['A1', 'E2'], T0, T0 + 1.6, 0.40, att=0.5, rel=0.4, swell=(0.6, 1.2))
    chord('vla', ['G3', 'A3', 'D4'], T0, T0 + 1.6, 0.36, att=0.5, rel=0.4, swell=(0.6, 1.2))
    for nt in ('A2', 'E3', 'D4'):
        pno(nt, T0, 0.42, 2.0)
    t_o = T0 + 1.5
    ostinato(t_o, t_th - 0.05, 7, [(0, 'Em7'), (t_o + 2.0, 'Asus')], vel=0.46, vel1=0.66)
    chord('vc', ['E2', 'B2'], t_o, t_o + 2.1, 0.46, att=0.2, swell=(0.9, 1.2)); chord('vc', ['A1', 'E2'], t_o + 2, t_th + 0.1, 0.52, att=0.2, rel=0.25, swell=(1.0, 1.4))
    chord('vla', ['G3', 'B3', 'D4'], t_o, t_o + 2.1, 0.44, att=0.2); chord('vla', ['G3', 'A3', 'D4'], t_o + 2, t_th + 0.1, 0.50, att=0.2, rel=0.25, swell=(1.0, 1.4))
    chord('trem', ['E5', 'A5'], t_o + 2, t_th + 0.1, 0.45, att=0.5, rel=0.2, swell=(0.4, 1.3), gain=0.6)
    # desmorona: notas despencam fora do pulso
    fall = ['A5', 'G5', 'E5', 'D5', 'B4', 'A4', 'G4', 'E4', 'D4', 'B3', 'A3', 'G3', 'E3', 'D3']
    tt = t_th
    for k, nt in enumerate(fall):
        pno(nt, tt, 0.55 - 0.02 * k, 1.2, gain=0.9)
        tt += 0.065 + 0.012 * k + float(R.uniform(0, 0.02))
    # depois: si menor, baixo e triste
    t_s = t_th + 0.35
    chord('vc', ['B1', 'F#2'], t_s, T0 + 8.2, 0.38, att=0.6, rel=1.2, gain=0.9)
    chord('vla', ['D3', 'F#3'], t_s, T0 + 8.2, 0.34, att=0.8, rel=1.2, gain=0.8)
    chord('cb', [low('B0')], t_s, T0 + 8.2, 0.36, att=0.6, rel=1.2)
    pno('B1', t_s + 0.2, 0.40, 3.0); pno('F#4', t_s + 0.6, 0.36, 2.5)

    # ---------------- L. bola_chao (99-103): chuva; o tema ao contrario; a mao recolhe a bola
    T0 = ST['bola_chao']
    chord('vc', ['B1', 'F#2'], T0, T0 + 4.1, 0.34, att=0.4, rel=0.6, swell=(0.8, 1.3))
    chord('vla', ['D3', 'F#3'], T0, T0 + 2.4, 0.30, att=0.4, rel=0.6, gain=0.8)
    pno('D5', T0 + 0.0, 0.40, 2.5); pno('C#5', T0 + 1.0, 0.38, 2.5); pno('B4', T0 + 2.0, 0.40, 2.5)
    for nt in ('B1', 'B2', 'F#3'):
        pno(nt, T0 + 2.35, 0.66, 3.0)                                             # a mao fecha na bola
    chord('vla', ['C#3', 'F#3', 'A#3'], T0 + 2.4, T0 + 4.05, 0.40, att=0.4, rel=0.25, swell=(0.6, 1.5))   # F#: dominante de si menor
    chord('trem', ['F#4', 'C#5'], T0 + 2.6, T0 + 4.05, 0.45, att=0.5, rel=0.2, swell=(0.3, 1.4), gain=0.6)

    # ---------------- M. tl3 (103-111): tempestade — sete notas por tempo, si menor
    T0 = ST['tl3']
    cds = [(0, 'Bm'), (T0 + 2, 'G'), (T0 + 4, 'Em'), (T0 + 6, 'F#')]
    ostinato(T0, T0 + 8.0, 7, cds, vel=0.66, vel1=0.80, octave=-1, insts=('piano',), gains=(0.58,))
    ostinato(T0, T0 + 8.0, 7, cds, vel=0.50, vel1=0.64, insts=('harp',), gains=(0.8,))
    bass = [('B1', 'B0'), ('G1', 'G0'), ('E1', 'E1'), ('F#1', 'F#0')]
    for k, (a, b) in enumerate(bass):
        chord('vc', [a, n(a) + 7], T0 + 2 * k, T0 + 2 * k + 2.1, 0.72, att=0.08, rel=0.3)
        chord('cb', [low(b)], T0 + 2 * k, T0 + 2 * k + 2.1, 0.7, att=0.08, rel=0.3)
        pno(a, T0 + 2 * k, 0.8, 2.0, gain=0.7); pno(n(a) - 12 if n(a) - 12 >= 23 else n(a), T0 + 2 * k, 0.78, 2.0, gain=0.7)
    tr = [(['B4', 'D5', 'F#5'], 0), (['B4', 'D5', 'G5'], 2), (['B4', 'E5', 'G5'], 4), (['A#4', 'C#5', 'F#5'], 6)]
    for nts, t in tr:
        chord('trem', nts, T0 + t, T0 + t + 2.08, 0.72, att=0.08, rel=0.25, gain=0.7, swell=(0.9, 1.15))
    mel = [('B4', 0, 1), ('D5', 1, 1), ('G5', 2, 2), ('E5', 4, 1), ('G5', 5, 1), ('F#5', 6, 2)]
    for nt, t, d in mel:
        play('vln', nt, T0 + t, d, 0.80, att=0.06, rel=0.35, gain=1.4); play('vln', n(nt) + 12, T0 + t, d, 0.74, att=0.06, rel=0.35, gain=1.0)
        play('horn', n(nt) - 12, T0 + t, d, 0.75, att=0.06, rel=0.3, gain=1.0)
    chord('vla', ['F#3', 'B3'], T0, T0 + 2.1, 0.66, att=0.08); chord('vla', ['G3', 'B3'], T0 + 2, T0 + 4.1, 0.66, att=0.08)
    chord('vla', ['G3', 'B3'], T0 + 4, T0 + 6.1, 0.66, att=0.08); chord('vla', ['F#3', 'A#3'], T0 + 6, T0 + 8.05, 0.70, att=0.08, rel=0.25, swell=(1.0, 1.3))

    # ---------------- N. cansaco (111-117): silencio depois da chuva; ele se levanta
    T0 = ST['cansaco']
    chord('vc', ['G1', 'D2'], T0 + 0.3, T0 + 2.2, 0.28, att=1.0, rel=1.0, gain=0.8)
    chord('vla', ['B3', 'D4'], T0 + 0.5, T0 + 2.2, 0.24, att=1.0, rel=1.0, gain=0.7)
    chord('vc', ['A1', 'E2'], T0 + 2.0, T0 + 6.15, 0.30, att=0.8, rel=0.5, swell=(0.8, 1.5))
    chord('vla', ['A3', 'D4'], T0 + 2.0, T0 + 4.3, 0.26, att=0.8, rel=0.8, gain=0.7)
    chord('vla', ['A3', 'C#4', 'E4'], T0 + 4.2, T0 + 6.15, 0.32, att=0.6, rel=0.4, swell=(0.8, 1.6), gain=0.8)
    chord('vln', ['E5'], T0 + 3.0, T0 + 6.15, 0.26, att=1.2, rel=0.4, swell=(0.5, 1.6), gain=0.7)
    for nt, t, v in (('B4', 0.6, .30), ('D5', 1.5, .32), ('E5', 2.3, .36), ('A4', 3.3, .38), ('C#5', 4.4, .44), ('E5', 5.2, .50)):
        pno(nt, T0 + t, v, min(2.6, 6.0 - t), rel=0.3)
    pno('A2', T0 + 2.3, 0.36, 3.5); pno('A1', T0 + 4.4, 0.44, 1.6, rel=0.3)

    # ---------------- O. sete_ok (117-127): a cascata de 7 firme; o tema nas cordas
    T0 = ST['sete_ok']
    T_thr = EV['_info']['torre_T_d']                                             # o lancamento do db97531
    cds = [(0, 'D'), (T0 + 4, 'G'), (T0 + 6, 'Em7'), (T0 + 8, 'Asus'), (T0 + 10, 'A')]
    ostinato(T0, T_thr - 0.02, 7, cds, vel=0.46, vel1=0.80, gains=(0.78, 0.62))
    ostinato(T0 + 4, T_thr - 0.02, 7, cds, vel=0.34, vel1=0.60, insts=('harp',), octave=1, gains=(0.6,))
    for nt in ('D2', 'A2', 'D3', 'F#3', 'A3'):
        pno(nt, T0, 0.6, 3.5)
    bs = [('D2', 0, 4, .50), ('G1', 4, 2, .56), ('E2', 6, 2, .60), ('A1', 8, T_thr - T0 - 8, .70)]
    for a, t, d, v in bs:
        chord('vc', [a, n(a) + 7], T0 + t, T0 + t + d + 0.08, v, att=0.15, rel=0.25, swell=(1.0, 1.25))
        chord('cb', [low(n(a) - 12)], T0 + t, T0 + t + d + 0.08, v, att=0.15, rel=0.25)
    chord('vla', ['F#3', 'A3', 'D4'], T0, T0 + 4.1, 0.46, att=0.3); chord('vla', ['G3', 'B3', 'D4'], T0 + 4, T0 + 6.1, 0.52, att=0.2)
    chord('vla', ['G3', 'B3', 'E4'], T0 + 6, T0 + 8.1, 0.56, att=0.2); chord('vla', ['A3', 'D4', 'E4'], T0 + 8, T0 + 10.1, 0.62, att=0.2)
    chord('vla', ['A3', 'C#4', 'E4', 'G4'], T0 + 10, T_thr + 0.05, 0.70, att=0.15, rel=0.2, swell=(1.0, 1.4))
    mel = [('F#5', 0, 1, .50), ('A5', 1, 1, .52), ('D6', 2, 1, .56), ('C#6', 3, 1, .56), ('B5', 4, 1, .60), ('D6', 5, 1, .62),
           ('G5', 6, 2, .64), ('E5', 8, 1, .68), ('F#5', 9, 1, .70), ('G5', 10, 1, .74), ('A5', 11, T_thr - T0 - 11, .80)]
    for nt, t, d, v in mel:
        play('vln', nt, T0 + t, d, v, att=0.08, rel=0.3, gain=1.9)
        if t >= 4:
            play('vln', n(nt) - 12, T0 + t, d, v - 0.05, att=0.08, rel=0.3, gain=1.2)
    play('horn', 'A3', T0 + 4, 2.0, 0.5, att=0.3, gain=0.6); play('horn', 'B3', T0 + 6, 2.0, 0.55, att=0.2, gain=0.65)
    play('horn', 'A3', T0 + 8, 2.0, 0.6, att=0.2, gain=0.7); play('horn', 'E4', T0 + 10, T_thr - T0 - 10, 0.75, att=0.2, rel=0.2, gain=0.8, env=lambda u: 0.8 + 0.5 * u)
    chord('trem', ['E5', 'A5', 'C#6'], T0 + 9, T_thr + 0.03, 0.70, att=0.8, rel=0.15, swell=(0.3, 1.5), gain=0.7)

    # ---------------- P. torre (127-145): o truque em camera lenta
    tw = EV['torre']
    throws = [e[0] for e in tw if e[1] == 'throw' and e[0] >= T_thr - 0.01]       # 7 lancamentos: d b 9 7 5 3 1
    catches = [e[0] for e in tw if e[1] == 'catch' and e[0] > throws[-1] + 0.05]   # a coluna desce: tac, tac, tac...
    t_tick = [e[0] for e in tw if e[1] == 'tick'][0]                               # o rebote na mao cheia
    t_fin = catches[-1]                                                            # a ultima bola e recolhida
    col = catches[:-1]
    # sobe: re-mi-la, sem a terca (suspenso)
    up = ['D4', 'A4', 'D5', 'E5', 'A5', 'D6', 'E6']
    for k, (t, nt) in enumerate(zip(throws, up)):
        pno(nt, t, 0.50 + 0.02 * k, min(6.0, t_tick - 0.05 - t), gain=1.0, send=0.55, rel=0.3)
        if k >= 4:
            play('glock', n(nt), t, 3.0, 0.20, gain=0.8)
    chord('vln', ['A5', 'E6'], T_thr + 0.6, t_tick + 0.1, 0.26, att=2.5, rel=0.8, gain=0.55)
    chord('vla', ['D4', 'E4', 'A4'], T_thr + 1.5, t_tick + 0.1, 0.24, att=2.5, rel=0.8, gain=0.55)
    chord('vc', ['D2', 'A2'], T_thr + 0.3, t_tick + 0.1, 0.26, att=2.0, rel=0.8, gain=0.7)
    # desce: cada bola na mao, uma nota
    down = ['A5', 'E5', 'D5', 'A4', 'E4', 'D4']
    for k, t in enumerate(col):
        nt = down[min(k + (len(down) - len(col)), len(down) - 1)] if len(col) <= len(down) else down[min(k, len(down) - 1)]
        pno(nt, t, 0.50 + 0.03 * k, min(4.0, t_tick - 0.05 - t), send=0.5, rel=0.3)
        play('pizz', n(nt) - 24 if n(nt) - 24 >= 36 else n(nt) - 12, t, 1.2, 0.5, gain=0.8)
    # o rebote: o ar some
    for nt in ('G3', 'A3', 'D4', 'E4'):
        pno(nt, t_tick, 0.52, t_fin - 0.06 - t_tick, send=0.5, rel=0.12)
    chord('trem', ['A4', 'D5', 'E5', 'G5'], t_tick + 0.05, t_fin + 0.02, 0.62, att=1.6, rel=0.12, swell=(0.15, 1.5), gain=0.75)
    chord('vc', ['A1', 'E2'], t_tick + 0.05, t_fin + 0.02, 0.55, att=1.5, rel=0.12, swell=(0.2, 1.5))
    chord('vla', ['G3', 'A3', 'D4'], t_tick + 0.3, t_fin + 0.02, 0.52, att=1.4, rel=0.12, swell=(0.2, 1.5))
    # ---- a ultima bola na mao: re maior, enfim com a terca
    tf = t_fin
    for nt, v in (('D1', .9), ('D2', .92), ('A2', .88), ('D3', .88), ('F#3', .85), ('A3', .85), ('D4', .88), ('F#4', .9), ('A4', .88), ('D5', .9)):
        pno(nt, tf, v, 6.0, send=0.4, gain=0.62)
    for k, nt in enumerate(['D4', 'F#4', 'A4', 'D5', 'F#5', 'A5', 'D6', 'F#6']):
        play('harp', nt, tf + 0.03 * k, 3.0, 0.7)
    play('glock', 'D6', tf, 3.0, 0.5); play('glock', 'F#6', tf + 0.12, 3.0, 0.45)
    c1 = ST['final']                                                               # corte para o plano final (145)
    prog = [(tf, c1, ['D2', 'A2'], ['D1'], ['F#3', 'A3', 'D4', 'F#4'], ['A4', 'D5', 'F#5']),
            (c1, c1 + 2, ['C#2', 'A2'], ['C#1'], ['E3', 'A3', 'C#4'], ['A4', 'C#5', 'E5']),
            (c1 + 2, c1 + 4, ['B1', 'F#2'], ['B0'], ['F#3', 'B3', 'D4'], ['B4', 'D5', 'F#5']),
            (c1 + 4, c1 + 6, ['G1', 'D2'], ['G0'], ['G3', 'B3', 'D4'], ['B4', 'D5', 'G5'])]
    for k, (a, b, vcn, cbn, vlan, vlnn) in enumerate(prog):
        v = [0.84, 0.76, 0.68, 0.58][k]
        chord('vc', vcn, a, b + 0.1, v, att=0.06 if k == 0 else 0.2, rel=0.4)
        chord('cb', [low(x) for x in cbn], a, b + 0.1, v, att=0.06 if k == 0 else 0.2, rel=0.4)
        chord('vla', vlan, a, b + 0.1, v, att=0.06 if k == 0 else 0.2, rel=0.4, gain=0.8)
        chord('vln', vlnn, a, b + 0.1, v - 0.08, att=0.06 if k == 0 else 0.2, rel=0.4, gain=0.7)
        if k > 0:
            for x in vcn:
                pno(x, a, v - 0.12, 2.5)
    chord('horn', ['A3', 'D4', 'F#4'], tf, c1 + 0.1, 0.80, att=0.08, rel=0.4, gain=0.8)
    chord('horn', ['A3', 'E4'], c1, c1 + 2.1, 0.66, att=0.2, gain=0.7); chord('horn', ['F#3', 'D4'], c1 + 2, c1 + 4.1, 0.58, att=0.2, gain=0.7)
    chord('horn', ['G3', 'D4'], c1 + 4, c1 + 6.1, 0.50, att=0.2, gain=0.7)
    # o tema, inteiro, cantado
    th = [('F#5', tf + 0.35, 1.0), ('A5', tf + 1.35, 1.0), ('D6', tf + 2.35, c1 - tf - 2.35), ('C#6', c1, 1.0), ('E6', c1 + 1, 1.0),
          ('D6', c1 + 2, 1.0), ('B5', c1 + 3, 1.0), ('A5', c1 + 4, 1.0), ('G5', c1 + 5, 1.0)]
    for k, (nt, t, d) in enumerate(th):
        v = 0.86 - 0.035 * k
        play('vln', nt, t, d, v, att=0.07, rel=0.35, gain=1.55)
        play('solo', nt, t, d, 0.5, att=0.07, rel=0.35, gain=0.6)
        pno(nt, t, v - 0.1, 2.2); pno(n(nt) - 12, t, v - 0.2, 2.2)
    # ---------------- Q. final (145-153): repouso
    te = c1 + 6
    chord('vc', ['D2', 'A2'], te, te + 3.2, 0.50, att=0.3, rel=2.0, swell=(1.0, 0.25))
    chord('cb', [low('D1')], te, te + 3.0, 0.46, att=0.3, rel=2.0, swell=(1.0, 0.25))
    chord('vla', ['F#3', 'A3', 'D4'], te, te + 3.2, 0.46, att=0.3, rel=2.0, swell=(1.0, 0.25))
    chord('vln', ['A4', 'D5', 'F#5'], te, te + 3.2, 0.40, att=0.3, rel=2.0, swell=(1.0, 0.2), gain=0.7)
    chord('horn', ['A3', 'D4'], te, te + 2.6, 0.42, att=0.3, rel=1.5, swell=(1.0, 0.2), gain=0.6)
    for nt, v in (('D2', .5), ('A2', .46), ('D3', .46), ('F#4', .5), ('A4', .46), ('D5', .48)):
        pno(nt, te, v, 5.0)
    pno('F#5', te + 1.0, 0.36, 4.0); pno('A5', te + 1.6, 0.32, 4.0)

    # ---------------- R. texto (153-164): piano so, uma frase por acorde
    T0 = ST['texto']
    def ph(t, low, high, v, t_end):
        for nt in low:
            pno(nt, T0 + t, v - 0.06, t_end - t, send=0.42, rel=0.35)
        for k, nt in enumerate(high):
            pno(nt, T0 + t + 0.02 + 0.5 * k, v, max(t_end - t - 0.5 * k, 0.4), send=0.42, rel=0.35)
    ph(0.9, ['D3', 'A3'], ['F#4', 'A4'], 0.46, 5.25)                               # Maos sujas,
    ph(2.1, ['B2', 'F#3'], ['D5', 'B4'], 0.46, 5.25)                               # corpo cansado,
    ph(3.3, ['G2', 'D3'], ['B4', 'D5', 'G5'], 0.48, 5.25)                          # mente tranquila.
    ph(5.3, ['A2', 'E3', 'G3'], ['C#5', 'E5'], 0.50, 6.55)                         # Isso e malabarismo.
    for nt, v in (('D2', .46), ('A2', .44), ('D3', .44), ('F#4', .5), ('A4', .46), ('D5', .5)):
        pno(nt, T0 + 6.6, v, 6.0, send=0.45)
    pno('F#5', T0 + 7.4, 0.36, 5.0, send=0.5); pno('D6', T0 + 8.2, 0.30, 5.0, send=0.5)
    chord('vc', ['D2', 'A2'], T0 + 6.6, T0 + 10.2, 0.22, att=1.5, rel=1.5, swell=(1.0, 0.3), gain=0.6)

    # ---------------- S. logo (164-171.5): a vinheta
    T0 = ST['logo']
    play('pizz', 'D3', T0 + 0.37, 0.8, 0.5)                                        # o monstrinho aparece
    play('pizz', 'A2', T0 + 1.55, 0.9, 0.75); play('glock', 'A5', T0 + 1.55, 1.0, 0.35)      # a bola bate na cabeca
    play('glock', 'D6', T0 + 2.20, 1.5, 0.42); play('pizz', 'D3', T0 + 2.20, 0.9, 0.6)       # ele recupera
    play('glock', 'F#6', T0 + 3.22, 1.2, 0.30)                                     # joga de volta
    for k, nt in enumerate(['D5', 'F#5', 'A5', 'D6']):                             # a assinatura
        play('glock', nt, T0 + 3.75 + 0.13 * k, 2.5, 0.40 + 0.03 * k)
        play('harp', nt, T0 + 3.75 + 0.13 * k, 2.5, 0.55)
        play('pizz', n(nt) - 24, T0 + 3.75 + 0.13 * k, 1.0, 0.5)
    tc = T0 + 4.40
    for nt in ('D2', 'A2', 'D3'):
        play('pizz', nt, tc, 1.6, 0.72)
    for nt in ('D4', 'F#4', 'A4', 'D5'):
        play('harp', nt, tc, 3.0, 0.62)
    play('glock', 'D6', tc, 3.0, 0.5); play('glock', 'F#6', tc, 3.0, 0.42); play('glock', 'A6', tc + 0.02, 3.0, 0.36)
    for nt in ('D3', 'A3', 'F#4'):
        pno(nt, tc, 0.5, 3.0)


# =====================================================================================================
#                                           EFEITOS SONOROS
# =====================================================================================================
def env_pts(pts, n_s, t0=0.0):
    """envelope por pontos (t, ganho) -> vetor de n_s amostras a partir de t0."""
    ts = np.array([p[0] for p in pts]); gs = np.array([p[1] for p in pts])
    t = t0 + np.arange(n_s) / SR
    return np.interp(t, ts, gs).astype(f32)


def effects():
    r = np.random.default_rng(77)
    # ---- vento (leito continuo com automacao)
    w = synth.wind(DUR + 1, r, gust=1.0)
    g = env_pts([(0, 0.0), (2.5, 0.9), (8, 1.0), (24, 0.9), (29, 0.6), (36, 0.75), (41, 0.55), (49, 0.5), (63, 0.4), (75, 0.45),
                 (91, 0.6), (99, 0.8), (103, 1.5), (110.8, 1.6), (111.2, 0.6), (117, 0.5), (127, 0.5), (129.2, 0.35),
                 (141.5, 0.35), (145, 0.6), (150, 0.5), (153.2, 0.0), (DUR + 1, 0.0)], len(w))
    sfx.add(w * g[:, None], 0.0, gain=0.30, send=0.05)
    # ---- passaros (alvorada, e de novo na manha depois da chuva)
    def birds(t0, t1, rate, gain):
        t = t0
        while t < t1:
            k = int(r.integers(0, 4))
            s = synth.bird_song(r, k)
            u = (t - t0) / (t1 - t0)
            sfx.add(synth.st(s), t, gain=gain * float(r.uniform(0.35, 1.0)) * float(np.sin(np.pi * min(max(u, 0.04), 0.96)) ** 0.5),
                    pan=float(r.uniform(-0.85, 0.85)), send=0.7)
            t += float(r.exponential(1.0 / rate)) + 0.15
    birds(2.5, 41.0, 0.75, 0.016)
    birds(41.0, 49.0, 0.4, 0.009)
    birds(111.2, 119.0, 0.8, 0.016)
    # ---- passos na relva
    for k in range(14):
        t = ST['subida'] + k * (0.74 / 1.22)
        if t < ST['chegada'] - 0.1:
            sfx.add(synth.sfx_step(r, 0.55 + 0.15 * r.random()), t, gain=0.20 * (0.5 + 0.5 * k / 13), pan=0.05, send=0.12)
    for tt in (0.32, 0.965, 1.61, 2.25, 2.90):
        sfx.add(synth.sfx_step(r, 0.7), ST['chegada'] + tt, gain=0.26, pan=-0.35, send=0.12)
    # ---- a bola na mao e o primeiro lancamento
    sfx.add(synth.sfx_rustle(r, 0.20, 0.5, 900, 4000), ST['bola'] + 0.9, gain=0.25, send=0.1)
    sfx.add(synth.sfx_throw(r, 0.9), ST['bola'] + 3.42, gain=0.5, send=0.15)
    sfx.add(synth.whoosh(0.7, r, 400, 2600), ST['bola'] + 3.35, gain=0.35, send=0.3)
    # ---- malabares em tempo real: cada pegada e cada lancamento
    def juggle(name, g_catch, g_throw, pan=0.0, t_max=1e9, t_min=-1e9):
        for (t, typ, inten, _) in EV[name]:
            if t > t_max or t < t_min:
                continue
            if typ == 'catch':
                sfx.add(synth.sfx_catch(r, inten), t, gain=g_catch, pan=pan + float(r.uniform(-0.12, 0.12)), send=0.10)
            elif typ == 'throw':
                sfx.add(synth.sfx_throw(r, inten), t, gain=g_throw, pan=pan, send=0.10)
            elif typ == 'thud':
                sfx.add(synth.sfx_thud(r, inten), t, gain=g_catch * 1.4, pan=pan + float(r.uniform(-0.3, 0.3)), send=0.12)
            elif typ == 'tick':
                sfx.add(synth.sfx_catch(r, 0.7, bright=1.6), t, gain=g_catch * 1.2, pan=pan, send=0.25)
    juggle('tres', 0.30, 0.10, pan=-0.15)
    juggle('cinco', 0.24, 0.07, pan=-0.2)
    juggle('sete_falha', 0.26, 0.07, pan=-0.3)
    juggle('sete_ok', 0.20, 0.06, pan=-0.1)
    T_thr = EV['_info']['torre_T_d']
    juggle('torre', 0.20, 0.06, pan=0.0, t_max=T_thr - 0.02)
    # camera lenta: sons mais graves e espaçados, com muita sala
    for (t, typ, inten, _) in EV['torre']:
        if t < T_thr - 0.02:
            continue
        if typ == 'throw':
            y = synth.repitch(synth.sfx_throw(r, 1.0), -7)
            sfx.add(y, t, gain=0.20, send=0.5)
        elif typ == 'catch':
            last = abs(t - EV['_info']['torre_t_last']) < 0.05
            y = synth.repitch(synth.sfx_catch(r, 1.0), -5 if not last else -3)
            sfx.add(y, t, gain=0.50 if not last else 0.68, send=0.45)
        elif typ == 'tick':
            y = synth.repitch(synth.sfx_catch(r, 0.8, bright=1.7), -2)
            sfx.add(y, t, gain=0.5, send=0.5)
    # batimento do coracao enquanto a coluna paira e a ultima bola quica
    t_tick = EV['_info']['torre_t_touch']; t_fin = EV['_info']['torre_t_last']
    hb = T_thr + 1.2
    per = 1.05
    while hb < t_fin - 0.35:
        gain = 0.55 if hb < t_tick else 0.85
        sfx.add(synth.heartbeat(r), hb, gain=gain * 0.5, send=0.08)
        per = 1.05 if hb < t_tick else max(0.62, per * 0.86)                     # acelera na duvida
        hb += per
    # ---- time-lapses: sopro de passagem do tempo e viradas de estacao
    for t, g in ((ST['tl1'] - 0.25, 0.5), (ST['tl2'] - 0.25, 0.5), (ST['tl2'] + 3.75, 0.6), (ST['tl2'] + 7.75, 0.6),
                 (ST['tl2'] + 11.75, 0.6), (ST['tl3'] - 0.3, 0.6)):
        sfx.add(synth.whoosh(0.6, r, 300, 3600), t, gain=g * 0.5, send=0.35)
    # ---- chuva e trovoada
    rn = synth.rain(ST['cansaco'] - ST['sete_falha'] - 6.0 + 0.5, r)
    t0r = ST['sete_falha'] + 6.3
    g = env_pts([(t0r, 0.0), (ST['bola_chao'], 0.55), (ST['bola_chao'] + 3.5, 0.8), (ST['tl3'], 1.25), (ST['tl3'] + 7.6, 1.3),
                 (ST['cansaco'], 0.9), (ST['cansaco'] + 0.35, 0.0)], len(rn), t0r)
    sfx.add(rn * g[:, None], t0r, gain=0.40, send=0.05)
    # pingos na relva depois da chuva
    for k in range(18):
        t = ST['cansaco'] + 0.4 + float(r.uniform(0, 4.5))
        d = synth.bp(synth.white(int(0.02 * SR), r), 1800, 6000) * synth.ad(int(0.02 * SR), 0.0005, 0.004)[:, None]
        sfx.add(d, t, gain=0.05 * float(r.uniform(0.3, 1.0)), pan=float(r.uniform(-0.8, 0.8)), send=0.5)
    sfx.add(synth.thunder(r, 5.5, crack=0.2), ST['bola_chao'] + 2.9, gain=0.28, pan=0.3, send=0.25)       # ao longe
    for tf, gn, pn in ((2.05, 0.85, -0.2), (4.85, 0.75, 0.3), (6.4, 1.0, 0.0)):
        sfx.add(synth.thunder(r, 5.0, crack=0.6), ST['tl3'] + tf + 0.18, gain=gn * 0.55, pan=pn, send=0.2)
    # ---- recolhe a bola da relva molhada
    sfx.add(synth.sfx_rustle(r, 0.35, 0.8, 700, 5000), ST['bola_chao'] + 2.25, gain=0.5, send=0.1)
    # ---- respiracao ofegante (ruido filtrado) no cansaco
    for k in range(4):
        t = ST['cansaco'] + 0.35 + k * 1.05
        nb = int(0.55 * SR)
        y = synth.bp(synth.white(nb, r), 500 if k % 2 == 0 else 700, 2400) * synth.bell(nb, 1.4)[:, None]
        sfx.add(y, t, gain=0.035 * (1 - 0.15 * k), pan=0.25, send=0.15)
    # ---- vinheta do monstrinho
    T0 = ST['logo']
    s, f = synth.sweep(320, 980, 0.07); sfx.add(synth.st(s * synth.bell(len(s), 0.8)), T0 + 0.36, gain=0.20, send=0.2)       # pop
    s, f = synth.sweep(1500, 520, 0.50)                                                                                 # a bola cai
    vib = 1 + 0.0 * s
    sfx.add(synth.st(s * synth.bell(len(s), 0.7) * np.linspace(0.4, 1.0, len(s))), T0 + 1.04, gain=0.11, send=0.25)
    nb = int(0.5 * SR); tt = np.arange(nb) / SR                                                                         # boing
    fb = 210 * (1 + 0.55 * np.exp(-tt / 0.16) * np.sin(2 * np.pi * 11 * tt))
    boing = np.sin(2 * np.pi * np.cumsum(fb) / SR) * np.exp(-tt / 0.16)
    sfx.add(synth.st(boing.astype(f32)), T0 + 1.55, gain=0.22, send=0.2)
    sfx.add(synth.sfx_catch(r, 0.9, bright=1.5), T0 + 1.55, gain=0.5, send=0.2)
    s, f = synth.sweep(500, 900, 0.30); sfx.add(synth.st(s * synth.bell(len(s), 0.8)), T0 + 1.75, gain=0.07, send=0.3)       # quica
    sfx.add(synth.sfx_catch(r, 0.8, bright=1.3), T0 + 2.20, gain=0.45, send=0.2)                                         # recupera
    sfx.add(synth.whoosh(0.45, r, 500, 4500), T0 + 3.15, gain=0.5, send=0.3)                                            # joga
    s, f = synth.sweep(600, 1900, 0.40); sfx.add(synth.st(s * synth.bell(len(s), 0.7) * np.linspace(1.0, 0.3, len(s))), T0 + 3.22, gain=0.09, send=0.3)


# =====================================================================================================
#                                         MIXAGEM E MASTER
# =====================================================================================================
def lufs(path):
    out = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-i', path, '-af', 'ebur128=peak=true', '-f', 'null', '-'],
                         capture_output=True, text=True).stderr
    tail = out[out.rfind('Summary:'):]
    import re
    i = float(re.search(r'I:\s*(-?[\d.]+) LUFS', tail).group(1))
    lra = float(re.search(r'LRA:\s*(-?[\d.]+) LU', tail).group(1))
    tp = float(re.search(r'Peak:\s*(-?[\d.]+) dBFS', tail).group(1))
    return i, lra, tp


def build(out_wav, mus_gain=1.0, sfx_gain=1.0, stems=False):
    music()
    effects()
    ir = synth.make_ir()
    n_out = int(DUR * SR)
    m = mus.dry * 0.86 + synth.reverb(mus.send, ir) * 0.62
    s = sfx.dry + synth.reverb(sfx.send, ir) * 0.55
    # automacao geral da musica (equilibrio entre as secoes)
    ga = env_pts(MUS_AUTO, len(m))
    m = m * ga[:, None]
    mix = m * mus_gain + s * sfx_gain
    mix = synth.hp(mix, 28, 2)
    mix = mix + 0.45 * synth.hp(mix, 5200, 1)          # prateleira de agudos (+3 dB): um pouco de ar
    mix = mix[:n_out]
    # fecha o audio em zero no ultimo meio segundo
    nf = int(0.6 * SR)
    mix[-nf:] *= np.linspace(1, 0, nf)[:, None].astype(f32) ** 1.5
    if stems:
        np.save(os.path.join(AUD, 'stem_mus.npy'), (m * mus_gain)[:n_out].astype(f32))
        np.save(os.path.join(AUD, 'stem_sfx.npy'), (s * sfx_gain)[:n_out].astype(f32))
    return mix


def section_auto(gains_db, ramp=0.25):
    """ganho por plano (dB) com rampas curtas nos cortes."""
    names = sorted([k for k in ST if k != '_fim'], key=lambda k: ST[k])
    pts = []
    for i, k in enumerate(names):
        a = ST[k]; b = ST[names[i + 1]] if i + 1 < len(names) else ST['_fim']
        g = 10 ** (gains_db.get(k, 0.0) / 20)
        pts += [(a + ramp, g), (b - ramp, g)]
    return [(0.0, pts[0][1])] + pts + [(DUR + 5, pts[-1][1])]


MUS_DB = dict(abertura=6.5, subida=5.5, chegada=3.5, olho=4.0, orvalho=4.0, neblina=3.5, bola=3.0, tres=4.0, tl1=2.0, cinco=1.0,
              tl2=0.5, sete_falha=1.0, bola_chao=3.5, tl3=-2.5, cansaco=3.0, sete_ok=-0.5, torre=0.0, final=0.5, texto=3.5,
              logo=-2.0)
MUS_AUTO = section_auto(MUS_DB)
# dentro do plano da torre: sobe o trecho suspenso, segura o acorde da chegada
_tt, _tf = EV['_info']['torre_T_d'], EV['_info']['torre_t_last']
MUS_AUTO = sorted([p for p in MUS_AUTO if not (_tt - 0.3 < p[0] < _tf + 0.3)] +
                  [(_tt - 0.25, 10 ** (-1.0 / 20)), (_tt + 0.15, 10 ** (3.0 / 20)), (_tf - 0.12, 10 ** (3.0 / 20)), (_tf - 0.02, 10 ** (-2.0 / 20))])


def compressor(x, thr_db=-10.0, ratio=2.4, att=0.006, rel=0.18, knee=6.0, hop=24):
    """compressor de barramento suave (reduz o fator de crista antes do limitador)."""
    pk = np.abs(x).max(1)
    nb = len(pk) // hop
    env_in = pk[:nb * hop].reshape(nb, hop).max(1)
    aa = np.exp(-hop / (att * SR)); ar = np.exp(-hop / (rel * SR))
    env = np.empty(nb)
    e = 0.0
    for i in range(nb):
        v = env_in[i]
        e = aa * e + (1 - aa) * v if v > e else ar * e + (1 - ar) * v
        env[i] = e
    lv = 20 * np.log10(np.maximum(env, 1e-7))
    over = lv - thr_db
    gr = np.where(over <= -knee / 2, 0.0, np.where(over >= knee / 2, over * (1 - 1 / ratio),
                                                   (1 - 1 / ratio) * (over + knee / 2) ** 2 / (2 * knee)))
    g = 10 ** (-gr / 20)
    gi = np.interp(np.arange(len(pk)), (np.arange(nb) + 0.5) * hop, g)
    return (x * gi[:, None]).astype(f32), float(gr.max())


def master(mix, out_wav, target=-14.0, ceil_tp=-1.5):
    tmp = os.path.join(AUD, '_m.wav')
    g = 0.5 / (np.abs(mix).max() + 1e-9)
    x, grmax = compressor(mix * g, thr_db=-13.0)
    print('  compressor: reducao maxima %.1f dB' % grmax)
    lim_ceil = ceil_tp - 0.6
    for it in range(6):
        y, lst = synth.limiter(x, lim_ceil, stats=True)
        synth.write_wav24(tmp, y)
        i, lra, tp = lufs(tmp)
        print('  passo %d: %.2f LUFS  LRA %.1f  pico real %.2f dBTP | limitador: max %.1f dB, >1 dB em %.1f%% do tempo, >3 dB em %.2f%%'
              % (it, i, lra, tp, lst['max_db'], 100 * lst['frac_1db'], 100 * lst['frac_3db']), flush=True)
        if abs(i - target) < 0.08 and tp <= ceil_tp:
            break
        if tp > ceil_tp and abs(i - target) < 0.3:
            lim_ceil -= (tp - ceil_tp) + 0.1
        x = x * 10 ** ((target - i) / 20)
    synth.write_wav24(out_wav, y)
    return i, lra, tp


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(AUD, 'trilha.wav')
    mix = build(out, stems=True)
    print('pico bruto %.3f' % np.abs(mix).max())
    np.save(os.path.join(AUD, 'mix_raw.npy'), mix.astype(f32))
    print(master(mix, out))
