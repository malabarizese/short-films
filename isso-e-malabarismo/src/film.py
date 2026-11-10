# film.py — definicao dos planos do filme
import numpy as np
from core import *
from shots import *
import juggle
from juggle import Choreo, Geom, L as HL, R as HR
from meadow import mound

ORIGIN = np.array([0.0, float(mound(0, 0)), 0.0])
TOWER_GEOM = Geom(0.33, 0.21, 1.12, 1.22, 0.38)


# ---------------- coreografias ----------------
def ch_tres(seed=4):
    c = Choreo(3, seed); c.hold(-5)
    c.look_keys += [(0.0, -8.0), (0.7, -8.0), (1.3, 0.0)]
    te = c.run_async(1.1, [3], 0.36, r=0.50, n_beats=17, jit=0.028, bad={15: (0.42, 0.0, 0.38)})
    c.root_keys += [(1.0, 0.0, 0.0), (2.6, 0.05, 0.03), (4.0, -0.06, 0.06), (5.6, 0.07, 0.10), (6.6, 0.10, 0.14)]
    c.finalize()
    return c


def ch_training(n, tau, r, runs, seed, jit0=0.03, jit1=0.012, t0=0.5, geom=None, look=0.0):
    """sequencia de tentativas: runs = lista de duracoes (s) antes de cada queda. ultima termina sem queda se <0."""
    c = Choreo(n, seed); c.hold(-5)
    rng = np.random.default_rng(seed + 100)
    t = t0
    first = HR
    for k, d in enumerate(runs):
        nb = int(abs(d) / tau)
        jit = jit0 + (jit1 - jit0) * k / max(len(runs) - 1, 1)
        bad = {}
        if d > 0:
            e = (rng.uniform(0.3, 0.6) * rng.choice([-1, 1]), 0.0, rng.uniform(0.2, 0.55))
            bad[nb - 1] = e
            if n >= 5 and rng.random() < 0.7:
                bad[nb - 3] = (rng.uniform(0.3, 0.7) * rng.choice([-1, 1]), 0.0, rng.uniform(0.1, 0.5))
            if n >= 7:
                bad[nb - 2] = (rng.uniform(0.3, 0.8) * rng.choice([-1, 1]), 0.0, rng.uniform(-0.2, 0.6))
                bad[nb - 4] = (rng.uniform(0.2, 0.6) * rng.choice([-1, 1]), 0.0, rng.uniform(0.0, 0.5))
        te = c.run_async(t, [n], tau, r=r, n_beats=nb, jit=jit, bad=bad, first=first)
        # espera as bolas pousarem
        tw = max(c.t_end, te) + 0.35
        tp = c.pickup(tw, dur=1.15)
        tp = c.rebalance(tp + 0.1, first=first)
        t = tp + 0.5
    c.finalize()
    return c


def ch_cinco(seed=6):
    c = Choreo(5, seed); c.hold(-5)
    tau = 0.215
    t0 = 0.7
    c.run_async(t0, [5], tau, r=0.42, n_beats=27, jit=0.010)            # batidas 0..26 (R nas pares)
    k0 = 28
    pairs = [((6, True), (4, False)), ((4, False), (6, True))]
    c.run_sync(t0 + k0 * tau, pairs, tau, r=0.42, n_steps=22, jit=0.010)
    c.finalize()
    return c


def ch_sete_falha(seed=8):
    c = Choreo(7, seed); c.hold(-5)
    tau = 0.20
    bad = {15: (0.45, 0, 0.35), 16: (-0.5, 0, 0.5), 18: (0.7, 0, 0.1), 19: (-0.35, 0, 0.7), 21: (0.25, 0, 0.9),
           22: (-0.75, 0, 0.3)}
    c.look_keys += [(0.0, -10.0), (1.0, -10.0), (1.5, 0.0), (4.2, 0.0), (5.4, -62.0), (9.0, -62.0)]
    c.pump_keys += [(1.4, 0.6)]
    te = c.run_async(1.5, [7], tau, r=0.30, n_beats=23, jit=0.03, bad=bad)
    c.root_keys += [(1.5, 0, 0), (3.2, 0.06, 0.08), (4.4, -0.08, 0.2)]
    c.finalize()
    return c


def ch_final(seed=9, lead=10.0):
    """cascata de 7 solida -> db97531 -> recolhe com o quique final. retorna (choreo, T_d)."""
    c = Choreo(7, seed); c.hold(-5)
    tau = 0.20
    c.pump_keys += [(0.45, 0.5)]
    nb = int(lead / tau)
    if nb % 2 == 1:
        nb += 1
    c.run_async(0.5, [7], tau, r=0.30, n_beats=nb, jit=0.007)
    T_d = 0.5 + nb * tau
    vals = [13, 11, 9, 7, 5, 3]
    dw = 2 * 0.30 * tau
    for k, v in enumerate(vals):
        h = HR if k % 2 == 0 else HL
        if k == 0:
            tb = c.throw_bounce(T_d, h, v, tau, dw, geom=TOWER_GEOM)
        else:
            c.throw(T_d + k * tau, h, v, tau, dw, geom=TOWER_GEOM, jit=0.004)
    c.throw(T_d + 6 * tau, HR, 1, tau, dw, geom=TOWER_GEOM)
    c.pump_keys += [(T_d - 0.02, 1.0), (T_d + 0.2, 0.8), (T_d + 0.4, 0.5), (T_d + 0.6, 0.3)]
    c.look_keys += [(T_d - 0.3, 0.0), (T_d + 0.5, 22.0), (T_d + 1.5, 24.0), (T_d + 2.3, 8.0), (T_d + 2.6, -30.0),
                    (T_d + 3.1, -34.0), (T_d + 4.2, -8.0)]
    c.lean_keys += [(T_d - 0.3, 0.0), (T_d + 0.6, 0.10), (T_d + 2.0, 0.06), (T_d + 2.7, -0.10), (T_d + 3.4, 0.0)]
    c.bend_keys += [(tb[0] - 0.05, 0.0), (tb[1] - 0.02, 0.16), (tb[1] + 0.5, 0.0)]
    c.T_d = T_d
    c.t_touch, c.t_last = tb
    c.finalize()
    return c, T_d


def speed_map(keys, dt=1 / 240.0):
    """keys: [(t_filme, velocidade)]. retorna funcao t_filme -> deslocamento de tempo simulado."""
    ts = np.array([k[0] for k in keys]); vs = np.array([k[1] for k in keys])
    grid = np.arange(0, ts[-1] + 1.0, dt)
    v = np.interp(grid, ts, vs)
    cum = np.concatenate([[0], np.cumsum((v[1:] + v[:-1]) * 0.5 * dt)])
    return lambda t: float(np.interp(t, grid, cum))


SHOTS = {}


def shot(fn):
    SHOTS[fn.__name__] = fn
    return fn


CIRRUS_SOFT = dict(amount=0.16, opacity=0.075, soft=0.40)


@shot
def tres():
    cam = orbit_cam(30.0, 6.9, height=0.95, target_h=1.62, vfov=30.0, off=(9.0, 1.5))
    ch = ch_tres()
    return HillShot('tres', 8.0, cam, hour=6.72, cloud=dict(coverage=0.20, ox=0.05, oz=0.55), cirrus=CIRRUS_SOFT,
                    fog=0.75, vis=55.0, tree=dict(age=0.02, leaves=1.0, bloom=0.0, leaf_age=0.2), choreo=ch,
                    nballs=3, meadow_kw=dict(d_far=8.2, focus=6.9, d_near=0.9), fig_kw=dict(silhouette=0.35),
                    ball_kw=dict(rimk=2.2, glow=0.0, radius=0.040), dolly=lambda t: (0.0, 0.0, 0.035 * t), rays=0.7, glow=1.2,
                    props=bag_prop())


@shot
def cinco():
    cam = orbit_cam(27.0, 7.6, height=0.95, target_h=1.95, vfov=33.0, off=(11.0, 2.0))
    ch = ch_cinco()
    return HillShot('cinco', 12.0, cam, hour=6.95, cloud=dict(coverage=0.22, ox=3.3, oz=2.1), cirrus=CIRRUS_SOFT,
                    fog=0.55, vis=55.0, tree=dict(age=0.30, leaves=1.0, bloom=0.0, leaf_age=0.45), choreo=ch,
                    nballs=5, meadow_kw=dict(d_far=8.9, focus=7.6, d_near=0.9), fig_kw=dict(silhouette=0.3),
                    ball_kw=dict(rimk=2.2, glow=0.0, radius=0.040), dolly=lambda t: (0.02 * t, 0.0, 0.03 * t), rays=0.7, glow=1.2,
                    seed=2, props=bag_prop())


# ---------------- cameras ----------------
def cam_master():
    return Camera((-2.5, float(mound(-2.5, -17.0)) + 1.0, -17.0), yaw=np.radians(6), pitch=np.radians(12.6), vfov=36.0)


def cam_wide2():
    return Camera((-2.2, float(mound(-2.2, -11.5)) + 1.0, -11.5), yaw=np.radians(8), pitch=np.radians(11.0), vfov=36.0)


def bag_prop(pos=(2.95, 2.05)):
    """mochila no chao, ao lado da arvore."""
    def fn(shot, cam, L, t):
        p = np.array([pos[0], float(mound(*pos)) + 0.13, pos[1]])
        x, y, z = cam.project(p)
        if z < 0.3:                    # mochila atras da camera: nada a desenhar
            return []
        k = cam.f / z

        def draw(img):
            amb = L['skyAvg']
            sunv = float(smooth(-0.8, 1.0, L['el']))
            col = np.array([0.10, 0.075, 0.035]) * (L['sunCol'] / np.pi * 0.5 * sunv + amb * 0.8)
            splat_disc(img, float(x), float(y), 0.15 * k, col, soft=max(1.0, 0.02 * k))
            splat_disc(img, float(x) + 0.05 * k, float(y) - 0.11 * k, 0.11 * k, col * 0.9, soft=max(1.0, 0.02 * k))
        return [(float(z), draw)]
    return fn


def birds_extra(n=7, seed=3, t_off=0.0, speed=1.0, y_el=3.0, size=1.0):
    r = np.random.default_rng(seed)
    offs = r.uniform(-1, 1, (n, 2)) * np.array([3.5, 0.9])
    ph = r.random(n) * 6.28
    fr = r.uniform(2.6, 3.4, n)

    def fn(shot, cam, L, t, img):
        def draw(img):
            tt = t + t_off
            base_az = np.degrees(cam.yaw) - 14.0 + 3.2 * tt * speed
            col = (L['skyAvg'] * 0.12 + 0.002).astype(np.float32)
            for i in range(n):
                az = base_az + offs[i, 0]
                el = y_el + offs[i, 1] + 0.25 * np.sin(tt * 0.5 + ph[i])
                d = dir_from(np.radians(az), np.radians(el))
                x, y, z = cam.project_dir(d)
                if z <= 0:
                    continue
                w = 7.0 * CFG.S * size * (cam.f / (2216.0 * CFG.S)) ** 0.6
                a = 0.75 * np.sin(tt * fr[i] * 2 * np.pi * 0.5 + ph[i])
                pts = np.array([[x - w, y - w * a], [x - w * 0.45, y - w * a * 0.35], [x, y], [x + w * 0.45, y - w * a * 0.35], [x + w, y - w * a]])
                lay = np.zeros((int(4 * w + 8), int(4 * w + 8)), np.uint8)
                o = np.array([x - 2 * w - 4, y - 2 * w - 4])
                cv2.polylines(lay, [np.round((pts - o) * 16).astype(np.int32).reshape(-1, 1, 2)], False, 255,
                              max(1, int(round(1.6 * CFG.S * size))), cv2.LINE_AA, shift=4)
                al = lay.astype(np.float32) / 255.0 * 0.9
                over(img, al[..., None] * col, al, int(o[0]), int(o[1]))
        return [(1e6, draw)]
    return fn


def rain_extra(amount=1.0, seed=1):
    def fn(shot, cam, L, t, img):
        def draw(img):
            H, W = img.shape[:2]
            r = np.random.default_rng(int(t * 30) * 7 + seed)
            col = (L['skyAvg'] * 1.3 + 0.01).astype(np.float32)
            for (n, ln, th, a, bl) in ((int(900 * amount), 26, 1, 0.16, 0.8), (int(220 * amount), 70, 2, 0.22, 2.2)):
                lay = np.zeros((H // 2, W // 2), np.uint8)
                x = r.uniform(-0.05 * W, 1.05 * W, n) / 2
                y = r.uniform(-0.1 * H, 1.0 * H, n) / 2
                l = ln * CFG.S * r.uniform(0.6, 1.3, n) / 2
                for i in range(n):
                    cv2.line(lay, (int(x[i] * 16), int(y[i] * 16)), (int((x[i] - l[i] * 0.16) * 16), int((y[i] + l[i]) * 16)),
                             int(255 * r.uniform(0.4, 1.0)), th, cv2.LINE_AA, shift=4)
                lf = cv2.GaussianBlur(lay.astype(np.float32) / 255.0, (0, 0), bl * max(CFG.S, 0.5))
                lf = cv2.resize(lf, (W, H), interpolation=cv2.INTER_LINEAR)
                img += lf[..., None] * (col * a)
        return [(0.5, draw)]
    return fn


def ground_balls_prop(points, radius=BALL_R):
    def fn(shot, cam, L, t):
        out = []
        for (x, z) in points:
            p = np.array([x, float(mound(x, z)) + radius * 0.9, z])
            zc = float((p - cam.pos) @ cam.fwd)
            out.append((zc, lambda img, p=p: draw_ball(img, cam, [p], radius, L, rimk=1.2)))
        return out
    return fn


def merge_props(*fns):
    def fn(shot, cam, L, t):
        out = []
        for f in fns:
            out += f(shot, cam, L, t)
        return out
    return fn


# ---------------- planos da colina ----------------
@shot
def subida():
    cam = Camera((-1.3, float(mound(-1.3, -25.0)) + 0.85, -25.0), yaw=np.radians(3.5), pitch=np.radians(6.2), vfov=21.0)
    p0, p1 = np.array([0.9, -13.5]), np.array([0.25, -3.6])
    dirv = (p1 - p0) / np.linalg.norm(p1 - p0)
    yaw = np.arctan2(dirv[0], dirv[1])

    def pose_fn(shot, t):
        s = 1.22 * t
        xz = p0 + dirv * s
        gy = float(mound(xz[0], xz[1]))
        g2 = float(mound(*(xz + dirv * 0.3)))
        slope = (g2 - gy) / 0.3
        p = walk_pose(s, np.array([xz[0], gy, xz[1]]), yaw, slope=slope)
        return [dict(pose=p, J=p.solve(), kw=dict(bag=True))], []
    return HillShot('subida', 8.0, cam, hour=lambda t: 5.80 + 0.012 * t, cloud=dict(coverage=0.22, ox=0.05, oz=0.55),
                    cirrus=dict(amount=0.22, opacity=0.11, soft=0.40), fog=0.95, vis=60.0,
                    tree=dict(age=0.02, leaves=1.0, bloom=0.0, leaf_age=0.2), pose_fn=pose_fn,
                    meadow_kw=dict(d_far=25.3, focus=16.0, d_near=1.6, dof=0.02), fig_kw=dict(silhouette=0.9, rim=0.4),
                    rays=0.0, glow=0.0, flare=0.0, key_dt=1.0, shake=1.6, dolly=lambda t: (0.0, 0.0, 0.11 * t),
                    expo_mul=1.25, wind=0.7, stars=lambda t: 0.5 * float(smooth(7.0, 0.0, t)), milky=0.0)


@shot
def chegada():
    cam = orbit_cam(26.0, 9.0, height=0.95, target_h=1.55, vfov=27.0, off=(12.5, 2.4))
    p0, p1 = np.array([0.5, -3.3]), np.array([0.0, 0.0])
    dirv = (p1 - p0) / np.linalg.norm(p1 - p0)
    dist = float(np.linalg.norm(p1 - p0))
    yaw0 = np.arctan2(dirv[0], dirv[1])
    T1 = dist / 1.15

    def pose_fn(shot, t):
        if t < T1:
            s = 1.15 * t
            xz = p0 + dirv * s
            p = walk_pose(s + 0.37, np.array([xz[0], float(mound(*xz)), xz[1]]), yaw0, slope=0.04)
        else:
            u = float(smooth(T1, T1 + 0.9, t))
            p = walk_pose(dist + 0.37, np.array([0.0, float(mound(0, 0)), 0.0]), yaw0 * (1 - u), slope=0.0)
            # assenta: pes juntos, corpo ereto
            for h, sg in ((0, -1.0), (1, 1.0)):
                p.feet[h] = p.feet[h] * (1 - u) + np.array([sg * 0.13, 0.07, -0.02]) * u
            p.lean = p.lean * (1 - u) + 0.02 * u
            p.arm_swing = (p.arm_swing[0] * (1 - u), p.arm_swing[1] * (1 - u))
            p.head = 4.0 + 5.0 * float(smooth(T1 + 0.6, T1 + 2.6, t))
            p.breath = np.sin(t * 2.0)
            p.squat = 0.03 + 0.006 * np.sin(t * 2.0)
        return [dict(pose=p, J=p.solve(), kw=dict(bag=True))], []
    return HillShot('chegada', 8.0, cam, hour=lambda t: 6.075 + 0.0135 * t, cloud=dict(coverage=0.24, ox=0.05, oz=0.55),
                    cirrus=dict(amount=0.18, opacity=0.08, soft=0.40), fog=0.95, vis=60.0,
                    tree=dict(age=0.02, leaves=1.0, bloom=0.0, leaf_age=0.2), pose_fn=pose_fn,
                    meadow_kw=dict(d_far=10.4, focus=9.0, d_near=1.0), fig_kw=dict(silhouette=0.55),
                    rays=0.9, glow=1.5, flare=1.2, key_dt=0.5, shake=1.5, dolly=lambda t: (0.0, 0.0, 0.05 * t),
                    seed=3)


def hour_tl1(t):
    if t < 11.6:
        return 6.50 + 12.0 * (t / 11.6)
    return 18.50 + (t - 11.6) / 2.4 * 2.6


@shot
def tl1():
    ch = ch_training(3, 0.36, 0.50, [3.5, 5.5, 7.5, 10.0, 13.0, -9.0], seed=21, jit0=0.035, jit1=0.014)
    return HillShot('tl1', 14.0, cam_wide2(), hour=hour_tl1, tl=True,
                    cloud=lambda t: dict(coverage=0.26 + 0.20 * np.sin(t * 0.45) ** 2, ox=0.2, oz=0.1),
                    cirrus=dict(amount=0.16, opacity=0.07, soft=0.40), cloud_speed=330.0,
                    fog=lambda t: float(lerp(0.75, 0.10, smooth(0.5, 3.5, t)) + 0.3 * smooth(10.5, 12.5, t)),
                    vis=lambda t: float(lerp(50, 70, smooth(1, 4, t))),
                    tree=dict(age=0.02, leaves=1.0, bloom=0.0, leaf_age=0.2), choreo=ch, nballs=3,
                    sim=lambda t: 0.2 + 6.0 * min(t, 11.9), ghost=True, sub=10, shutter=1.0, cloud_div=3,
                    cloud_N=48, shake=0.0, star_ang=lambda t: np.radians(15.0 * (hour_tl1(t) - 18.0)) * 1.0,
                    ball_kw=dict(rimk=1.4, glow=0.6, radius=0.046), props=bag_prop(), wind=1.6, seed=4,
                    fade_out=0.30, rays=0.5, key_dt=0.5, vis_soft=5.0)


SEASONS = [
    dict(dry=0.0, tree=dict(age=0.36, leaves=1.0, bloom=0.0, leaf_age=0.5), decl=-20.0, cov=0.34, fog=0.15, pet=None),
    dict(dry=0.55, tree=dict(age=0.44, leaves=0.75, bloom=0.0, leaf_age=0.95), decl=5.0, cov=0.18, fog=0.2,
         pet=(0.5, dict(col=np.array([0.33, 0.20, 0.035]), carpet=0.5))),
    dict(dry=1.0, tree=dict(age=0.52, leaves=0.0, bloom=0.0), decl=23.0, cov=0.55, fog=0.6, pet=None),
    dict(dry=0.35, tree=dict(age=0.60, leaves=0.08, bloom=1.0, leaf_age=0.1), decl=8.0, cov=0.25, fog=0.2,
         pet=(1.0, dict(carpet=0.8))),
]
SEG = 4.0


@shot
def tl2():
    ch = ch_training(5, 0.215, 0.42, [4.0, 6.0, 9.0, 12.0, 15.0, -9.0], seed=33, jit0=0.022, jit1=0.010)

    def seg(t):
        return SEASONS[int(min(t // SEG, 3))]

    def hour(t):
        u = (t % SEG) / SEG
        return 7.6 + 10.2 * u

    def expo_mul(t):
        m = 1.0
        for b in (SEG, 2 * SEG, 3 * SEG):
            m *= 1.0 - 0.93 * np.exp(-((t - b) / 0.16) ** 2)
        return m
    return HillShot('tl2', 16.0, cam_wide2(), hour=hour, tl=True, decl=lambda t: seg(t)['decl'],
                    cloud=lambda t: dict(coverage=seg(t)['cov'] + 0.08 * np.sin(t * 1.3), ox=0.5 + 0.2 * (t // SEG), oz=0.3),
                    cirrus=dict(amount=0.16, opacity=0.07, soft=0.40), cloud_speed=300.0,
                    fog=lambda t: seg(t)['fog'], vis=60.0, tree=lambda t: seg(t)['tree'], dry=lambda t: seg(t)['dry'],
                    petals=lambda t: seg(t)['pet'], choreo=ch, nballs=5, sim=lambda t: 0.2 + 5.0 * t, ghost=True,
                    sub=10, shutter=1.0, cloud_div=3, cloud_N=48, shake=0.0, expo_mul=expo_mul,
                    ball_kw=dict(rimk=1.4, glow=0.6, radius=0.046), props=bag_prop(), wind=1.6, seed=5, vis_soft=5.0)


@shot
def sete_falha():
    cam = orbit_cam(-20.0, 9.2, height=0.95, target_h=2.35, vfov=36.0, off=(4.0, 2.0))
    ch = ch_sete_falha()
    return HillShot('sete_falha', 8.0, cam, hour=18.13, cloud=dict(coverage=0.46, ox=0.9, oz=0.72), cirrus=CIRRUS_SOFT,
                    fog=0.45, vis=55.0, tree=dict(age=0.72, leaves=1.0, bloom=0.0, leaf_age=0.6), choreo=ch, nballs=7,
                    meadow_kw=dict(d_far=10.6, focus=9.2, d_near=1.0), fig_kw=dict(silhouette=0.5),
                    ball_kw=dict(rimk=1.0, glow=0.7, radius=0.040), dolly=lambda t: (0.0, 0.0, 0.03 * t), rays=0.0, seed=6,
                    props=bag_prop(), expo_mul=1.45)


@shot
def tl3():
    ch = ch_training(7, 0.20, 0.30, [2.6, 3.2, 3.8, 4.4, 5.2, 6.0], seed=45, jit0=0.03, jit1=0.02)
    flashes = [2.05, 2.25, 4.85, 6.4, 6.55]

    def flash(t):
        return sum(np.exp(-((t - f) / 0.045) ** 2) for f in flashes)

    def post(shot, disp, t):
        f = flash(t)
        if f > 0.01:
            disp = np.clip(disp + f * 0.55 * (0.25 + 0.75 * disp) * np.float32([0.85, 0.92, 1.1]), 0, 1)
        return disp
    return HillShot('tl3', 8.0, cam_wide2(), hour=lambda t: 15.6 + 0.42 * t, tl=True,
                    cloud=lambda t: dict(coverage=0.93, lcmax=1.0, hb=2.0, ht=5.6, sigma=60.0, type=1.0, ox=1.3, oz=0.9,
                                         dens_mult=1.25),
                    cirrus=None, cloud_speed=260.0, fog=0.55, vis=22.0, tree=dict(age=0.78, leaves=1.0, bloom=0.0, leaf_age=0.6),
                    choreo=ch, nballs=7, sim=lambda t: 0.2 + 6.0 * t, ghost=True, sub=10, shutter=1.0, cloud_div=3,
                    cloud_N=48, shake=0.0, ball_kw=dict(rimk=0.6, glow=0.8, radius=0.046), props=bag_prop(), wind=2.6,
                    seed=7, extra=rain_extra(1.0), ground_sun=0.10, sky_sun=0.13, expo_mul=lambda t: 1.9 - 0.05 * t, post=post,
                    rays=0.0, glow=0.0, flare=0.0, sat=0.8)


@shot
def cansaco():
    cam = orbit_cam(33.0, 6.2, height=0.9, target_h=1.25, vfov=30.0, off=(-9.0, 3.0))

    def pose_fn(shot, t):
        p = Pose(shot.origin, -0.15)
        u = float(smooth(2.3, 4.4, t))
        p.bend = 0.50 * (1 - u)
        p.squat = 0.07 * (1 - u) + 0.03 + 0.012 * np.sin(t * 3.4) * (1 - u)
        p.breath = np.sin(t * 3.4) * (1 - 0.6 * u)
        p.head = -40.0 * (1 - u) + 9.0 * float(smooth(4.0, 5.6, t))
        wfree = float(smooth(0.30, 0.72, u))
        if wfree < 1.0:
            for h, sg in ((0, -1.0), (1, 1.0)):
                kn = np.array([sg * 0.15, 0.585 - 0.03 * np.sin(t * 3.4) * 0.3, 0.085])
                p.hands[h] = kn
                p.hand_free[h] = wfree
        return [dict(pose=p, J=p.solve())], []
    pts = [(-0.6, 0.9), (0.5, 1.4), (1.1, 0.3), (-1.2, 0.2), (0.2, -0.7), (-0.4, 1.9), (0.9, -0.5)]
    return HillShot('cansaco', 6.0, cam, hour=lambda t: 5.845 + 0.012 * t, cloud=dict(coverage=0.30, ox=1.7, oz=0.2),
                    cirrus=dict(amount=0.2, opacity=0.09, soft=0.40), fog=0.9, vis=60.0,
                    tree=dict(age=0.86, leaves=0.05, bloom=1.0), pose_fn=pose_fn, meadow_kw=dict(d_far=7.5, focus=6.2, d_near=0.9),
                    fig_kw=dict(silhouette=0.75, rim=0.6), props=merge_props(bag_prop(), ground_balls_prop(pts)),
                    rays=0.0, glow=0.0, flare=0.0, key_dt=1.0, dolly=lambda t: (0.0, 0.0, 0.03 * t), seed=8,
                    expo_mul=1.15, petals=(0.35, dict(carpet=1.0)), stars=lambda t: 0.4 * float(smooth(5.0, 0.0, t)), milky=0.0)


FINAL = {}


def final_choreo():
    if 'ch' not in FINAL:
        FINAL['ch'], FINAL['T_d'] = ch_final(lead=10.0)
    return FINAL['ch'], FINAL['T_d']


@shot
def sete_ok():
    cam = orbit_cam(-17.0, 10.2, height=0.95, target_h=2.5, vfov=36.0, off=(5.5, 1.5))
    ch, T_d = final_choreo()
    return HillShot('sete_ok', 10.0, cam, hour=lambda t: 17.60 + 0.012 * t, cloud=dict(coverage=0.30, ox=2.2, oz=1.4),
                    cirrus=CIRRUS_SOFT, fog=0.30, vis=60.0, tree=dict(age=1.0, leaves=0.06, bloom=1.0), choreo=ch,
                    nballs=7, sim=lambda t: T_d - 12.2 + t, meadow_kw=dict(d_far=11.6, focus=10.2, d_near=1.0),
                    ball_kw=dict(rimk=0.8, glow=0.25, radius=0.040), dolly=lambda t: (0.0, 0.0, 0.07 * t), rays=0.0, seed=9,
                    props=bag_prop(), petals=(0.8, dict(carpet=1.0)), key_dt=1.0)


TOWER_KEYS = [(0.0, 1.0), (1.95, 1.0), (2.45, 0.30), (6.1, 0.27), (6.9, 0.16), (9.6, 0.15), (10.4, 0.30), (11.9, 0.30),
              (12.4, 0.17), (15.0, 0.16), (15.8, 1.0), (18.0, 1.0)]
TOWER_LEAD = 2.1


def tower_sim():
    ch, T_d = final_choreo()
    sm = speed_map(TOWER_KEYS)
    return lambda t: T_d - TOWER_LEAD + sm(t)


def moon_fn(expo_ref=0.25, az=-7.5, el=4.6):
    d = dir_from(np.radians(az), np.radians(el))
    return dict(dir=d, col=np.array([1.0, 0.93, 0.80]) * (0.80 / expo_ref), size=2.1, light=(0.05, 0.06, 0.085))


@shot
def torre():
    ch, T_d = final_choreo()
    sim = tower_sim()
    line = lambda t: float(lerp(0.2, 2.45, smooth(0.0, 2.3, t)))

    def ball_fn(shot, b, pts, t):
        y = float(np.mean(pts[:, 1])) - shot.origin[1]
        hl = line(t)
        return dict(lit_override=float(smooth(hl - 0.25, hl + 0.45, y)), glow=0.55, halo=1.0, boost=2.6)
    return HillShot('torre', 18.0, cam_master(), hour=17.945, cloud=dict(coverage=0.30, ox=2.2, oz=1.4),
                    cirrus=CIRRUS_SOFT, fog=0.38, vis=60.0, tree=dict(age=1.0, leaves=0.06, bloom=1.0), choreo=ch, nballs=7,
                    sim=sim, ball_fn=ball_fn, ball_kw=dict(rimk=0.6, radius=0.046), sub=4, shutter=0.5,
                    ground_sun=lambda t: float(1.0 - smooth(0.0, 2.0, t)),
                    tree_sun_h=lambda t: (line(t) - 0.3, line(t) + 0.7), rays=0.0, seed=10, props=bag_prop(),
                    petals=lambda t: (0.8, dict(carpet=1.0, t=sim(t) + 40.0)), key_dt=1.0, moon=moon_fn(0.25),
                    shake=0.8, wind=lambda t: 0.35 + 0.65 * float(smooth(2.4, 1.6, t)) + 0.65 * float(smooth(15.0, 16.0, t)))


RECOLHE_T0 = 9.0          # tempo do plano 'torre' em que entra o plano medio
RECOLHE_T1 = 16.0         # ...e em que volta o plano geral
RECOLHE_CAM = dict(phi=-32.0, dist=5.4, height=1.0, target_h=1.72, vfov=26.0, off=(0.0, 0.0))


@shot
def recolhe():
    """plano medio da descida: as ultimas bolas na mao, o rebote e a recuperacao (mesmo tempo simulado da 'torre')."""
    ch, T_d = final_choreo()
    sim0 = tower_sim()
    c = RECOLHE_CAM
    cam = orbit_cam(c['phi'], c['dist'], height=c['height'], target_h=c['target_h'], vfov=c['vfov'], off=c['off'])
    hl = 2.45

    def ball_fn(shot, b, pts, t):
        y = float(np.mean(pts[:, 1])) - shot.origin[1]
        return dict(lit_override=float(smooth(hl - 0.25, hl + 0.45, y)), glow=0.55, halo=0.7, boost=2.2)
    return HillShot('recolhe', RECOLHE_T1 - RECOLHE_T0, cam, hour=17.945, cloud=dict(coverage=0.30, ox=2.2, oz=1.4),
                    cirrus=CIRRUS_SOFT, fog=0.38, vis=60.0, tree=dict(age=1.0, leaves=0.06, bloom=1.0), choreo=ch, nballs=7,
                    sim=lambda t: sim0(t + RECOLHE_T0), ball_fn=ball_fn, ball_kw=dict(rimk=0.6, radius=0.040), sub=6, shutter=0.5,
                    ground_sun=0.0, tree_sun_h=(hl - 0.3, hl + 0.7), rays=0.0, seed=12, props=bag_prop(),
                    petals=lambda t: (0.3, dict(carpet=1.0, t=sim0(t + RECOLHE_T0) + 40.0)), key_dt=1.0, moon=moon_fn(0.25),
                    shake=0.8, wind=lambda t: 0.35 + 0.65 * float(smooth(15.0, 16.0, t + RECOLHE_T0)),
                    meadow_kw=dict(d_far=c['dist'] + 1.3, focus=c['dist'], d_near=0.9), fig_kw=dict(silhouette=0.88),
                    dolly=lambda t: (0.0, 0.0, 0.035 * t))


@shot
def final():
    cam = orbit_cam(22.0, 7.4, height=0.95, target_h=1.45, vfov=30.0, off=(-8.0, 3.2))
    ch, T_d = final_choreo()
    t_hold = ch.t_last + 0.9
    return HillShot('final', 8.0, cam, hour=lambda t: 18.02 + 0.022 * t, cloud=dict(coverage=0.30, ox=2.2, oz=1.4),
                    cirrus=CIRRUS_SOFT, fog=0.45, vis=60.0, tree=dict(age=1.0, leaves=0.06, bloom=1.0), choreo=ch, nballs=7,
                    sim=lambda t: t_hold + t, meadow_kw=dict(d_far=8.7, focus=7.4, d_near=0.9), fig_kw=dict(silhouette=0.35),
                    ball_kw=dict(rimk=0.5, glow=0.5, radius=0.040), dolly=lambda t: (0.0, 0.0, -0.10 * t), rays=0.0, seed=11,
                    props=bag_prop(), petals=(0.6, dict(carpet=1.0)), key_dt=1.0, fade_out=3.2,
                    moon=lambda t: moon_fn(exposure_for(float(np.degrees(np.arcsin(skylib.sun_dir(18.02 + 0.022 * t)[1])))), az=-6.5, el=5.4),
                    expo_mul=lambda t: 1.0 - 0.25 * float(smooth(0, 8, t)))


# ---------------- vistas e natureza ----------------
def cam_vista(vfov=13.0, yaw=2.0, pitch=-0.8):
    return Camera((-2.5, 0.05, -17.0), yaw=np.radians(yaw), pitch=np.radians(pitch), vfov=vfov)


@shot
def abertura():
    return HillShot('abertura', 8.0, cam_vista(13.0, 2.0, -0.7), hour=lambda t: 5.80 + 0.0185 * t,
                    cloud=dict(coverage=0.20, ox=0.05, oz=0.55), cirrus=dict(amount=0.20, opacity=0.10, soft=0.40),
                    fog=1.0, vis=62.0, with_meadow=False, cloud_div=2, cloud_N=90, key_dt=0.5, shake=0.0,
                    zoom=lambda t: 1.0 + 0.05 * (t / 8.0), fade_in=2.6, rays=0.0, glow=0.0, flare=0.0,
                    extra=birds_extra(7, 3, speed=1.0, y_el=1.9), stars=lambda t: 0.6 * float(smooth(6.0, 0.0, t)),
                    milky=0.0, expo_mul=1.1, fog_glow=0.35)


@shot
def neblina():
    return HillShot('neblina', 3.5, cam_vista(7.5, -9.0, -1.9), hour=lambda t: 6.40 + 0.004 * t,
                    cloud=dict(coverage=0.20, ox=0.05, oz=0.55), cirrus=None, fog=1.0, vis=62.0, with_meadow=False,
                    cloud_div=2, cloud_N=90, key_dt=0.5, shake=0.0, zoom=lambda t: 1.0 + 0.04 * (t / 3.5),
                    rays=0.0, glow=0.0, flare=0.0, extra=birds_extra(9, 8, speed=1.6, y_el=-1.4, size=1.6),
                    fog_glow=0.45, seed=12, expo_mul=0.9)


def dew_extra(n=120, seed=5):
    r = np.random.default_rng(seed)
    px = r.uniform(0.02, 0.98, n)
    py = r.uniform(0.42, 0.98, n)
    rad = r.uniform(7, 34, n) * (0.5 + py)
    ph = r.uniform(0, 6.28, n)
    sp = r.uniform(0.6, 1.8, n)
    gl = r.random(n)

    def fn(shot, cam, L, t, img):
        def draw(img):
            H, W = img.shape[:2]
            col = (L['sunCol'] / np.pi * 0.85 + L['skyAvg'] * 0.4).astype(np.float32)
            for i in range(n):
                tw = 0.5 + 0.5 * np.sin(ph[i] + t * sp[i] * 2.2)
                a = (0.05 + 0.20 * gl[i]) * (0.35 + 0.65 * tw)
                x = px[i] * W + 6 * CFG.S * np.sin(t * 0.9 + ph[i])
                y = py[i] * H
                rr = rad[i] * CFG.S
                splat_disc(img, x, y, rr, col, alpha=a, soft=max(1.5, rr * 0.22), additive=True)
                if gl[i] > 0.82:
                    splat_disc(img, x, y, 2.2 * CFG.S, col * 9.0, alpha=tw, soft=1.6, additive=True)
        return [(0.2, draw)]
    return fn


@shot
def orvalho():
    hr = 6.34
    sun = skylib.sun_dir(hr)
    az = float(np.arctan2(sun[0], sun[2]))
    cam = Camera((-3.4, float(mound(-3.4, -8.5)) + 0.20, -8.5), yaw=az - np.radians(5.5), pitch=np.radians(6.5), vfov=27.0)
    return HillShot('orvalho', 3.5, cam, hour=hr, cloud=dict(coverage=0.20, ox=0.05, oz=0.55), cirrus=None, fog=0.9,
                    vis=60.0, tree=None, meadow_kw=dict(d_near=0.55, d_far=9.5, focus=1.386, dof=0.055, density=1.35,
                                                        ratio=1.26, clearing=(40.0, 40.0), stalks=1.6),
                    rays=0.0, glow=1.6, flare=0.8, key_dt=1.0, bg_blur=20.0, extra=dew_extra(), shake=1.0,
                    dolly=lambda t: (0.035 * t, 0.0, 0.03 * t), wind=1.3, seed=13, expo_mul=1.1)
