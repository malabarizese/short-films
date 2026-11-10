# juggle.py — simulacao de siteswap com maos, quedas e recolhas
# Coordenadas locais do malabarista: x direita, y cima, z frente. Origem nos pes.
import numpy as np

G = 9.81
L, R = 0, 1
SIDE = {L: -1.0, R: 1.0}


def hermite(p0, m0, p1, m1, u):
    u2, u3 = u * u, u * u * u
    return (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * m1


class Flight:
    def __init__(self, ball, t0, t1, p0, p1, h_from, h_to, value, caught=True):
        self.ball, self.t0, self.t1 = ball, t0, t1
        self.p0, self.p1 = np.asarray(p0, float), np.asarray(p1, float)
        self.h_from, self.h_to, self.value, self.caught = h_from, h_to, value, caught
        T = t1 - t0
        self.T = T
        self.v0 = (self.p1 - self.p0) / T
        self.v0[1] += 0.5 * G * T

    def pos(self, t):
        dt = t - self.t0
        p = self.p0 + self.v0 * dt
        p[1] -= 0.5 * G * dt * dt
        return p

    def vel(self, t):
        v = self.v0.copy()
        v[1] -= G * (t - self.t0)
        return v

    def apex(self):
        return self.p0[1] + self.v0[1] ** 2 / (2 * G)


class Geom:
    """posicoes de pegada/lancamento (locais)."""

    def __init__(self, xo=0.28, xi=0.11, yc=1.13, yt=1.20, z=0.36):
        self.xo, self.xi, self.yc, self.yt, self.z = xo, xi, yc, yt, z

    def catch(self, h):
        return np.array([SIDE[h] * self.xo, self.yc, self.z])

    def throw(self, h):
        return np.array([SIDE[h] * self.xi, self.yt, self.z])

    def rest(self, h):
        return np.array([SIDE[h] * (self.xo * 0.85), self.yc - 0.07, self.z - 0.05])


GEOM = {3: Geom(0.27, 0.10, 1.13, 1.20, 0.36), 5: Geom(0.31, 0.12, 1.13, 1.21, 0.37),
        7: Geom(0.35, 0.15, 1.12, 1.22, 0.38)}

SLOTS = [np.array([0.0, 0.0, 0.0]), np.array([0.066, -0.012, 0.012]), np.array([-0.060, -0.010, 0.022]),
         np.array([0.006, 0.002, 0.070])]


class Choreo:
    """Coreografia: trilhas de bolas, eventos de maos, parametros do corpo e eventos sonoros."""

    def __init__(self, n, seed=0, ground_y=0.0):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.tracks = [[] for _ in range(n)]      # por bola: lista de segmentos (t0, tipo, dados)
        self.hand_ev = {L: [], R: []}             # (t, tipo, pos, vel)
        self.held = {L: [], R: []}                # [(bola, t_pegada)]
        self.body = []                            # (t, dict)
        self.sfx = []                             # (t, tipo, intensidade)
        self.ground_y = ground_y
        self.geom = GEOM.get(n, Geom())
        self.t_end = 0.0
        self.root_keys = []                       # (t, x, z)
        self.bend_keys = []                       # (t, valor 0..1)
        self.look_keys = []                       # (t, graus para cima)
        self.pump_keys = []                       # (t, amplitude) agachada de impulso
        self.lean_keys = []
        self.busy = []                            # intervalos (t0,t1,tipo)

    # ---------- estado inicial ----------
    def hold(self, t, nR=None, nL=None):
        n = self.n
        if nR is None:
            nR = (n + 1) // 2
        if nL is None:
            nL = n - nR
        ids = list(range(n))
        self.held = {R: [(b, -1e9) for b in ids[:nR]], L: [(b, -1e9) for b in ids[nR:nR + nL]]}
        for h in (L, R):
            for k, (b, _) in enumerate(self.held[h]):
                self.tracks[b].append((t, 'hand', (h, k)))

    def _slot_refresh(self, h, t):
        k = 0
        for (b, tc) in self.held[h]:
            if tc <= t:
                self.tracks[b].append((t, 'hand', (h, k)))
                k += 1

    # ---------- lancamento generico ----------
    def throw(self, t, h, value, tau, dw, cross=None, geom=None, err=(0, 0, 0), caught=True, jit=0.0):
        g = geom or self.geom
        q = self.held[h]
        avail = [i for i, (b, tc) in enumerate(q) if tc <= t - 0.015]
        if not avail:
            self.n_fail = getattr(self, 'n_fail', 0) + 1
            return None
        b, _ = q.pop(avail[0])
        self.n_throws = getattr(self, 'n_throws', 0) + 1
        if cross is None:
            cross = (value % 2 == 1)
        h_to = (1 - h) if cross else h
        t_c = t + value * tau - dw
        if value == 1:
            t_c = t + max(tau - dw, 0.07)
        p0 = g.throw(h) + self.rng.normal(0, jit * 0.5, 3)
        p1 = g.catch(h_to) + self.rng.normal(0, jit, 3) + np.asarray(err, float)
        if value == 1:
            p0 = g.catch(h) * 0.5 + g.throw(h) * 0.5
            p1 = g.catch(h_to) * 0.55 + g.throw(h_to) * 0.45
            p1[1] = p0[1] - 0.01
        f = Flight(b, t, t_c, p0, p1, h, h_to, value, caught)
        self.tracks[b].append((t, 'fly', f))
        self.hand_ev[h].append((t, 'T', p0, f.vel(t), b))
        self.sfx.append((t, 'throw', min(1.0, value / 9.0)))
        self._slot_refresh(h, t)
        if caught:
            self.hand_ev[h_to].append((t_c, 'C', f.pos(t_c), f.vel(t_c), b))
            self.held[h_to].append((b, t_c))
            self.held[h_to].sort(key=lambda e: e[1])
            # o slot real e definido no momento da pegada
            self.tracks[b].append((t_c, 'hand_auto', h_to))
            self.sfx.append((t_c, 'catch', min(1.0, 0.35 + value / 14.0)))
        else:
            self._fall(b, f)
        self.t_end = max(self.t_end, t_c)
        return f

    def throw_bounce(self, t, h, value, tau, dw, geom=None, jit=0.0, vup=1.9, drift=(-0.16, 0.0, 0.07), dt_b=0.44,
                     drop=(-0.13, -0.11, 0.09)):
        """lanca uma bola que quica na mao cheia (quase cai) e e recuperada num bote. retorna (t_toque, t_pegada)."""
        g = geom or self.geom
        q = self.held[h]
        avail = [i for i, (b, tc) in enumerate(q) if tc <= t - 0.015]
        if not avail:
            self.n_fail = getattr(self, 'n_fail', 0) + 1
            return None
        b, _ = q.pop(avail[0])
        self.n_throws = getattr(self, 'n_throws', 0) + 1
        h_to = (1 - h) if value % 2 == 1 else h
        t_c = t + value * tau - dw
        p0 = g.throw(h)
        p1 = g.catch(h_to) + np.array([0.0, 0.055, 0.0])       # toca nas bolas ja seguras
        f = Flight(b, t, t_c, p0, p1, h, h_to, value, True)
        self.tracks[b].append((t, 'fly', f))
        self.hand_ev[h].append((t, 'T', p0, f.vel(t), b))
        self.sfx.append((t, 'throw', 1.0))
        self._slot_refresh(h, t)
        # toque: a mao esta la, mas a bola escapa
        self.hand_ev[h_to].append((t_c, 'C', g.catch(h_to), f.vel(t_c) * 0.5, -1))
        self.sfx.append((t_c, 'tick', 0.8))
        sg = SIDE[h_to]
        pb = p1 + np.array([sg * abs(drop[0]), drop[1], drop[2]])
        f2 = Flight(b, t_c, t_c + dt_b, p1, pb, h_to, h_to, 0, True)
        # velocidade inicial desejada para cima: ajusta via duracao (parabola fixa pelos extremos)
        self.tracks[b].append((t_c, 'fly', f2))
        self.hand_ev[h_to].append((t_c + dt_b, 'C', pb - np.array([0, 0.03, 0]), f2.vel(t_c + dt_b), b))
        self.held[h_to].append((b, t_c + dt_b))
        self.tracks[b].append((t_c + dt_b, 'hand_auto', h_to))
        self.sfx.append((t_c + dt_b, 'catch', 1.0))
        self.t_end = max(self.t_end, t_c + dt_b)
        return t_c, t_c + dt_b

    def _fall(self, b, f):
        """bola nao pega: segue a parabola ate o chao e rola um pouco."""
        gy = self.ground_y + 0.035
        a, bq, c = -0.5 * G, f.v0[1], f.p0[1] - gy
        tg = (-bq - np.sqrt(max(bq * bq - 4 * a * c, 0))) / (2 * a)
        t_hit = f.t0 + tg
        p_hit = f.pos(t_hit)
        p_hit[1] = gy
        v = f.vel(t_hit)
        roll = np.array([v[0], 0.0, v[2]]) * 0.16 + self.rng.normal(0, 0.03, 3) * np.array([1, 0, 1])
        self.tracks[b].append((t_hit, 'ground', (p_hit, roll, abs(v[1]))))
        self.sfx.append((t_hit, 'thud', min(1.0, abs(v[1]) / 8.0)))
        self.t_end = max(self.t_end, t_hit + 0.6)
        return t_hit, p_hit + roll

    # ---------- padroes ----------
    def run_async(self, t0, values, tau, r=0.3, first=R, geom=None, jit=0.0, look=None, bad=None, loop=True,
                  n_beats=None):
        """values: lista de siteswap; repete ate n_beats. bad = {batida: erro(x,y,z)} -> bola cai."""
        dw = 2 * r * tau
        n_beats = n_beats or len(values)
        bad = bad or {}
        t = t0
        stop_at = None
        for k in range(n_beats):
            t = t0 + k * tau
            if stop_at is not None and t > stop_at:
                break
            h = first if k % 2 == 0 else 1 - first
            v = values[k % len(values)]
            if v == 0 or v == 2:
                continue
            if k in bad:
                f = self.throw(t, h, v, tau, dw, geom=geom, err=bad[k], caught=False, jit=jit)
                if f is not None:
                    # percebe quando a bola passa pela altura das maos
                    stop_at = min(stop_at or 1e9, f.t1 + 0.12)
            else:
                self.throw(t, h, v, tau, dw, geom=geom, jit=jit)
        return (t0 + n_beats * tau) if stop_at is None else stop_at

    def run_sync(self, t0, pairs, tau, r=0.3, geom=None, jit=0.0, n_steps=None):
        """pairs: lista de ((vL, crossL), (vR, crossR)) aplicada a cada 2 batidas."""
        dw = 2 * r * tau
        n_steps = n_steps or len(pairs)
        for k in range(n_steps):
            t = t0 + k * 2 * tau
            (vl, cl), (vr, cr) = pairs[k % len(pairs)]
            if vl:
                self.throw(t, L, vl, tau, dw, cross=cl, geom=geom, jit=jit)
            if vr:
                self.throw(t, R, vr, tau, dw, cross=cr, geom=geom, jit=jit)
        return t0 + n_steps * 2 * tau

    # ---------- chao / recolha ----------
    def ground_balls(self, t):
        out = []
        for b in range(self.n):
            seg = self._seg(b, t)
            if seg is not None and seg[1] == 'ground':
                out.append((b, self.ball_local(b, t)))
        return out

    def pickup(self, t0, dur=1.25, order=None):
        """recolhe todas as bolas do chao, uma por vez. retorna t final."""
        t = t0
        gb = self.ground_balls(t0 + 1e-3)
        if order is None:
            gb.sort(key=lambda e: np.hypot(e[1][0], e[1][2]))
        for b, p in gb:
            h = R if p[0] >= 0 else L
            # mao com menos bolas tem prioridade se estiver vazia
            if len(self.held[1 - h]) + 1 < len(self.held[h]):
                h = 1 - h
            dist = np.hypot(p[0], p[2] - 0.15)
            d = dur * (0.85 + 0.25 * min(dist, 1.5))
            tm = t + d * 0.52
            # raiz anda ate perto da bola
            rx, rz = p[0] - SIDE[h] * 0.22, p[2] - 0.30
            reach = np.hypot(rx, rz)
            if reach > 0.25:
                self.root_keys += [(t, None, None), (t + d * 0.40, rx, rz), (tm + d * 0.10, rx, rz)]
            self.bend_keys += [(t, 0.0), (t + d * 0.42, 1.0), (tm + 0.06, 1.0), (t + d, 0.0)]
            self.look_keys += [(t, None), (t + d * 0.3, -55.0), (tm, -55.0), (t + d, 0.0)]
            self.hand_ev[h].append((tm, 'P', np.array([p[0], p[1] + 0.02, p[2]]), np.zeros(3), b))
            self.held[h].append((b, tm))
            self.tracks[b].append((tm, 'hand', (h, len(self.held[h]) - 1)))
            self.sfx.append((tm, 'grab', 0.5))
            self.busy.append((t, t + d, 'pickup'))
            t += d
        if gb:
            self.root_keys += [(t, None, None), (t + 0.55, 0.0, 0.0)]
            t += 0.55
        self.t_end = max(self.t_end, t)
        return t

    def rebalance(self, t, first=R):
        """passa bolas de uma mao para a outra ate a configuracao inicial (ceil(n/2) na mao que lanca primeiro)."""
        want = {first: (self.n + 1) // 2, 1 - first: self.n // 2}
        tt = t
        for _ in range(self.n):
            src = None
            for h in (L, R):
                if len(self.held[h]) > want[h]:
                    src = h
            if src is None:
                break
            dst = 1 - src
            b, _ = self.held[src].pop()
            p0 = self.geom.rest(src) + SLOTS[min(len(self.held[src]), 3)]
            p1 = self.geom.rest(dst) + SLOTS[min(len(self.held[dst]), 3)]
            f = Flight(b, tt, tt + 0.30, p0, p1, src, dst, 1)
            self.tracks[b].append((tt, 'fly', f))
            self.held[dst].append((b, tt + 0.30))
            self.tracks[b].append((tt + 0.30, 'hand', (dst, len(self.held[dst]) - 1)))
            self.sfx.append((tt + 0.30, 'catch', 0.3))
            tt += 0.34
        self.t_end = max(self.t_end, tt)
        return tt

    # ---------- finalizacao ----------
    def finalize(self):
        for h in (L, R):
            self.hand_ev[h].sort(key=lambda e: e[0])
        # resolve slots 'hand_auto': ordem de chegada na mao naquele instante
        occ = {L: [], R: []}       # lista de (t_in, t_out, bola, slot)
        for b in range(self.n):
            self.tracks[b].sort(key=lambda s: s[0])
        # reconstroi ocupacao percorrendo eventos em ordem temporal
        evs = []
        for b in range(self.n):
            for i, s in enumerate(self.tracks[b]):
                evs.append((s[0], b, i))
        evs.sort()
        inhand = {L: {}, R: {}}    # bola -> slot
        where = {}
        for t, b, i in evs:
            s = self.tracks[b][i]
            if b in where:
                inhand[where[b]].pop(b, None)
                del where[b]
            if s[1] == 'hand_auto':
                h = s[2]
                used = set(inhand[h].values())
                slot = next(k for k in range(8) if k not in used)
                inhand[h][b] = slot
                where[b] = h
                self.tracks[b][i] = (t, 'hand', (h, slot))
            elif s[1] == 'hand':
                h, slot = s[2]
                used = set(inhand[h].values())
                if slot in used:
                    slot = next(k for k in range(8) if k not in used)
                    self.tracks[b][i] = (t, 'hand', (h, slot))
                inhand[h][b] = slot
                where[b] = h
        self._ht = {h: np.array([e[0] for e in self.hand_ev[h]]) for h in (L, R)}
        for name in ('bend_keys', 'look_keys', 'pump_keys', 'lean_keys'):
            ks = sorted(getattr(self, name), key=lambda e: e[0])
            setattr(self, name, ks)
        self.root_keys.sort(key=lambda e: e[0])
        self.sfx.sort(key=lambda e: e[0])
        return self

    # ---------- consulta ----------
    def _seg(self, b, t):
        tr = self.tracks[b]
        cur = None
        for s in tr:
            if s[0] <= t:
                cur = s
            else:
                break
        return cur

    def _key(self, keys, t, default=0.0):
        """interpolacao suave entre chaves (t, v); v=None herda o valor corrente."""
        if not keys:
            return default
        vals = []
        cur = default
        for (kt, v) in keys:
            if v is None:
                v = cur
            vals.append((kt, v))
            cur = v
        if t <= vals[0][0]:
            return vals[0][1]
        for (a, va), (b_, vb) in zip(vals[:-1], vals[1:]):
            if a <= t <= b_:
                if b_ - a < 1e-6:
                    return vb
                u = (t - a) / (b_ - a)
                u = u * u * (3 - 2 * u)
                return va + (vb - va) * u
        return vals[-1][1]

    def root(self, t):
        if not self.root_keys:
            return np.zeros(2)
        xs, zs = [], []
        cx, cz = 0.0, 0.0
        for (kt, x, z) in self.root_keys:
            if x is None:
                x, z = cx, cz
            xs.append((kt, x)); zs.append((kt, z))
            cx, cz = x, z
        return np.array([self._key(xs, t), self._key(zs, t)])

    def hand_local(self, h, t):
        """posicao da palma (local, relativa aos pes na raiz atual)."""
        ev = self.hand_ev[h]
        g = self.geom
        rest = g.rest(h)
        if not ev:
            return rest
        ts = self._ht[h]
        i = int(np.searchsorted(ts, t, side='right')) - 1
        root = self.root(t)

        def P(e):
            # eventos de recolha do chao ficam em coordenadas absolutas -> converte para relativas a raiz
            if e[1] == 'P':
                return e[2] - np.array([root[0], 0.0, root[1]])
            return e[2]

        if i < 0:
            e = ev[0]
            dt = e[0] - t
            if dt > 0.45:
                return rest
            u = 1 - dt / 0.45
            m1 = e[3] * (0.12 if e[1] == 'T' else 0.05)
            return hermite(rest, np.zeros(3), P(e), m1, u * u * (3 - 2 * u))
        e0 = ev[i]
        if i + 1 >= len(ev):
            dt = t - e0[0]
            if dt > 0.5:
                return rest
            u = dt / 0.5
            m0 = e0[3] * (0.10 if e0[1] == 'C' else 0.06)
            return hermite(P(e0), m0, rest, np.zeros(3), u * u * (3 - 2 * u))
        e1 = ev[i + 1]
        D = e1[0] - e0[0]
        u = (t - e0[0]) / max(D, 1e-5)
        Dc = min(D, 0.26)
        s0 = {'C': 0.38, 'T': 0.30, 'P': 0.0}[e0[1]]
        s1 = {'C': 0.38, 'T': 1.0, 'P': 0.0}[e1[1]]
        p0, p1 = P(e0), P(e1)
        if D > 0.9:
            # intervalo longo: passa pelo repouso
            if u < 0.5:
                return hermite(p0, e0[3] * s0 * 0.2, rest, np.zeros(3), min(u * 2 * D / 0.9, 1.0) ** 0.8)
            uu = max(1 - (1 - u) * 2 * D / 0.9, 0.0)
            return hermite(rest, np.zeros(3), p1, e1[3] * s1 * 0.2, uu * uu * (3 - 2 * uu))
        return hermite(p0, e0[3] * s0 * Dc, p1, e1[3] * s1 * Dc, u)

    def ball_local(self, b, t):
        s = self._seg(b, t)
        if s is None:
            return self.geom.rest(R)
        kind = s[1]
        if kind == 'fly':
            f = s[2]
            if t <= f.t1 or not f.caught:
                return f.pos(t)
            return f.pos(f.t1)
        if kind == 'hand':
            h, slot = s[2]
            r = self.root(t)
            p = self.hand_local(h, t) + SLOTS[min(slot, 3)] * np.array([SIDE[h], 1, 1])
            return p + np.array([r[0], 0.0, r[1]])
        if kind == 'ground':
            p, roll, vy = s[2]
            dt = t - s[0]
            k = 1 - np.exp(-dt * 4.0)
            q = p + roll * k
            # quique pequeno amortecido (bola de sementes quase nao quica)
            bh = 0.035 * vy * np.exp(-dt * 7.0) * abs(np.sin(dt * 16.0))
            q = q.copy()
            q[1] += bh
            return q
        return self.geom.rest(R)

    def ball_state(self, b, t):
        s = self._seg(b, t)
        return None if s is None else s[1]

    def in_air(self, t):
        return [b for b in range(self.n) if self.ball_state(b, t) == 'fly']

    def bend(self, t):
        return self._key(self.bend_keys, t, 0.0)

    def look(self, t):
        return self._key(self.look_keys, t, 0.0)

    def lean(self, t):
        return self._key(self.lean_keys, t, 0.0)

    def pump(self, t):
        v = 0.0
        for (kt, a) in self.pump_keys:
            d = t - kt
            if -0.22 < d < 0.30:
                v += a * (np.exp(-((d + 0.07) / 0.085) ** 2) - 0.35 * np.exp(-((d - 0.10) / 0.10) ** 2))
        return v


# ---------- montadores de cenas ----------
def cascade_values(n):
    return [n]


def make_run(n, t0, dur, tau, r, seed=0, jit=0.012, drop_at=None, drop_err=None, look=None, collapse=0):
    """corrida simples de cascata com n bolas a partir de t0. Retorna Choreo (nao finalizada) e t final."""
    c = Choreo(n, seed)
    c.hold(t0 - 1.0)
    beats = int(dur / tau)
    bad = {}
    if drop_at is not None:
        kb = int(drop_at / tau)
        bad[kb] = drop_err if drop_err is not None else (0.35, 0.0, 0.25)
    te = c.run_async(t0, [n], tau, r, n_beats=beats, jit=jit, bad=bad)
    return c, te
