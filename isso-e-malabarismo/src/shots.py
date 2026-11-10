# shots.py — infraestrutura de planos (cenas) e renderizacao por cena
import os
import sys
import subprocess
import time
import numpy as np
import cv2
from core import *
import sky as skylib
import fx
from scene import *
from atmos import exposure_for
from tree import Petals
import juggle


def val(x, t):
    return x(t) if callable(x) else x


def orbit_cam(phi_deg, dist, height=1.0, target_h=1.55, vfov=30.0, off=(0.0, 0.0), center=(0.0, 0.0)):
    """camera em orbita atras do malabarista. phi=0: exatamente atras; phi>0: para o lado direito dele (+x)."""
    phi = np.radians(phi_deg)
    x = center[0] + dist * np.sin(phi)
    z = center[1] - dist * np.cos(phi)
    y = float(mound(x, z)) + height
    yaw = np.arctan2(center[0] - x, center[1] - z) + np.radians(off[0])
    pitch = np.arctan2(target_h + float(mound(*center)) - y, dist) + np.radians(off[1])
    return Camera((x, y, z), yaw=yaw, pitch=pitch, vfov=vfov)


class HillShot:
    def __init__(self, name, dur, cam, hour=7.0, tl=False, cloud=None, cirrus=None, wind_kms=(0.007, -0.003),
                 cloud_speed=1.0, fog=0.5, vis=50.0, tree=None, petals=None, dry=0.0, choreo=None, sim=None, nballs=3,
                 origin=None, body_yaw=0.0, dolly=None, shake=1.2, expo_mul=1.0, grade=None, fade_in=0.0, fade_out=0.0,
                 moon=None, rays=0.5, glow=1.0, flare=1.0, ghost=False, ball_kw=None, fig_kw=None, wind=1.0,
                 cloud_div=2, key_dt=0.5, meadow_kw=None, cloud_N=64, seed=1, extra=None, pose_fn=None, green=1.0,
                 tree_base=(3.6, 2.2), fog_glow=0.25, sub=8, shutter=0.5, stars=None, star_ang=0.0, clouds=True,
                 ball_fn=None, bloom_amt=0.10, sat=1.0, decl=0.0, world=None, grain_amt=0.016, milky=1.0,
                 ridge_seed=3, props=None, sun_fn=None, ground_sun=None, with_meadow=True, zoom=None, tree_sun_h=None,
                 post=None, bg_blur=0.0, sky_sun=1.0, vis_soft=1.2):
        self.name, self.dur = name, dur
        self.n = int(round(dur * FPS))
        self.cam0 = cam
        self.__dict__.update(dict(hour=hour, tl=tl, cloud=cloud, cirrus=cirrus, wind_kms=wind_kms, cloud_speed=cloud_speed,
                                  fog=fog, vis=vis, tree=tree, petals=petals, dry=dry, choreo=choreo, sim=sim,
                                  nballs=nballs, body_yaw=body_yaw, dolly=dolly, shake=shake, expo_mul=expo_mul,
                                  grade=grade or {}, fade_in=fade_in, fade_out=fade_out, moon=moon, rays=rays,
                                  glow=glow, flare=flare, ghost=ghost, ball_kw=ball_kw, fig_kw=fig_kw, wind=wind,
                                  cloud_div=cloud_div, key_dt=key_dt, meadow_kw=meadow_kw, cloud_N=cloud_N, seed=seed,
                                  extra=extra, pose_fn=pose_fn, green=green, tree_base=tree_base, fog_glow=fog_glow,
                                  sub=sub, shutter=shutter, stars=stars, star_ang=star_ang, clouds=clouds,
                                  ball_fn=ball_fn, bloom_amt=bloom_amt, sat=sat, decl=decl, grain_amt=grain_amt,
                                  milky=milky, ridge_seed=ridge_seed, props=props, sun_fn=sun_fn, ground_sun=ground_sun, with_meadow=with_meadow, zoom=zoom,
                                  tree_sun_h=tree_sun_h, post=post, bg_blur=bg_blur, sky_sun=sky_sun, vis_soft=vis_soft))
        self.origin = np.array([0.0, float(mound(0, 0)), 0.0]) if origin is None else np.asarray(origin, float)
        self.world = world
        self._keys = {}
        self._L = None
        self._pet = None
        self.rng = np.random.default_rng(seed * 7919 + 13)

    # ---------- preparo ----------
    def setup(self):
        if self.world is None:
            self.cam0.W, self.cam0.H = CFG.W, CFG.H
            self.cam0.update()
            self.world = World(self.cam0, tree_base=self.tree_base, meadow_kw=self.meadow_kw,
                               cloud_div=self.cloud_div, ridge_seed=self.ridge_seed, with_meadow=self.with_meadow)
        return self

    def sun(self, t):
        if self.sun_fn is not None:
            return np.asarray(self.sun_fn(t), dtype=np.float32)
        return skylib.sun_dir(val(self.hour, t), decl_deg=val(self.decl, t))

    def cloud_kw(self, t):
        c = dict(val(self.cloud, t) or {})
        P = dict(skylib.CP_DEFAULT)
        P.update(dict(sigma=55.0, tmax=95.0))
        ox, oz = c.pop('ox', 0.0), c.pop('oz', 0.0)
        P.update(c)
        ct = t * val(self.cloud_speed, t) if not callable(self.cloud_speed) else self.cloud_speed(t)
        wx, wz = self.wind_kms
        P['bwx'] = wx * ct * P['bs'] + ox * 0.37
        P['bwz'] = wz * ct * P['bs'] + oz * 0.21
        P['wwx'] = wx * ct * P['ws'] * 0.85 + ox
        P['wwz'] = wz * ct * P['ws'] * 0.85 + oz
        P['dwx'] = wx * ct * P['ds'] * 1.25
        P['dwz'] = wz * ct * P['ds'] * 1.25
        P['dwy'] = 0.0025 * ct * P['ds']
        P['evol'] = 0.0012 * ct
        return P

    def cirrus_kw(self, t):
        c = val(self.cirrus, t)
        if c is None:
            return None
        c = dict(c)
        ct = t * (self.cloud_speed(t) if callable(self.cloud_speed) else self.cloud_speed)
        c['offx'] = c.get('offx', 0.0) + 0.012 * ct * c.get('scale', 1 / 55.0)
        c['offz'] = c.get('offz', 0.0) + 0.004 * ct * c.get('scale', 1 / 55.0)
        return c

    def _bg(self, t, tk=None):
        """renderiza o fundo completo no instante t."""
        W_ = self.world
        tt = t if tk is None else tk
        sun = self.sun(tt)
        moon = val(self.moon, tt)
        L = W_.lighting(sun, moon)
        img, occ, info = W_.background(sun, L, self.cloud_kw(tt), self.cirrus_kw(tt), fog=val(self.fog, tt),
                                       vis_km=val(self.vis, tt), stars=val(self.stars, tt),
                                       star_ang=val(self.star_ang, tt), moon=moon, t=tt, seed=int(tt * 30) % 997,
                                       cloud_N=self.cloud_N, green=val(self.green, tt), fog_glow=val(self.fog_glow, tt),
                                       clouds=self.clouds, milky=self.milky, fog_t=tt * 0.4)
        return img, occ, info, L

    def background(self, t):
        if self.tl:
            return self._bg(t)
        k = int(np.floor(t / self.key_dt + 1e-6))
        u = t / self.key_dt - k
        for kk in (k, k + 1):
            if kk not in self._keys:
                self._keys[kk] = self._bg(kk * self.key_dt)
        for kk in list(self._keys):
            if kk < k:
                del self._keys[kk]
        a, b = self._keys[k], self._keys[k + 1]
        u = float(min(max(u, 0.0), 1.0))
        img = a[0] * np.float32(1 - u) + b[0] * np.float32(u)
        occ = a[1] * np.float32(1 - u) + b[1] * np.float32(u)
        # luz e posicao do sol interpoladas entre as chaves (sem degraus de exposicao)
        L = {}
        for key, va in a[3].items():
            vb = b[3].get(key, va)
            if isinstance(va, np.ndarray):
                L[key] = (va * (1 - u) + vb * u).astype(va.dtype)
            elif isinstance(va, (int, float, np.floating)):
                L[key] = float(va) * (1 - u) + float(vb) * u
            else:
                L[key] = va
        if 'sun' in L:
            L['sun'] = (L['sun'] / max(float(np.linalg.norm(L['sun'])), 1e-9)).astype(a[3]['sun'].dtype)
        info = dict(a[2])
        for key in ('sun_px', 'moon_px'):
            if key in a[2] and key in b[2]:
                info[key] = tuple(float(x) * (1 - u) + float(y) * u for x, y in zip(a[2][key], b[2][key]))
        return img, occ, info, L

    # ---------- personagem ----------
    def sim_t(self, t):
        return self.sim(t) if self.sim is not None else t

    def figures_balls(self, t):
        if self.pose_fn is not None:
            return self.pose_fn(self, t)
        figs, balls = [], []
        ch = self.choreo
        if ch is None:
            return figs, balls
        dt = 1.0 / FPS
        if self.ghost:
            ts0 = self.sim_t(t - 0.5 * dt)
            p0 = juggler_pose(ch, ts0, self.origin, self.body_yaw, self.nballs, breath_t=t)
            figs.append(dict(pose=p0, J=p0.solve(), alpha=0.45))
        ts = self.sim_t(t)
        pose = juggler_pose(ch, ts, self.origin, self.body_yaw, self.nballs, breath_t=t)
        figs.append(dict(pose=pose, J=pose.solve()))
        sub = self.sub
        sts = [self.sim_t(t + dt * self.shutter * (i / max(sub - 1, 1) - 0.5)) for i in range(sub)] if sub > 1 else [ts]
        c, s = np.cos(self.body_yaw), np.sin(self.body_yaw)
        Rm = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        for b in range(ch.n):
            pts = np.array([Rm @ ch.ball_local(b, st) + self.origin for st in sts])
            if self.ball_fn is not None:
                balls.append((pts, self.ball_fn(self, b, pts, t)))
            else:
                balls.append(pts)
        return figs, balls

    # ---------- quadro ----------
    def frame(self, i):
        t = i / FPS
        W_ = self.world
        img, occ, info, L = self.background(t)
        if self.bg_blur > 0:
            img = cv2.GaussianBlur(img, (0, 0), self.bg_blur * CFG.S)
        cam = self.cam0
        if self.dolly is not None:
            d = np.asarray(self.dolly(t), float)
            fh = np.array([cam.fwd[0], 0.0, cam.fwd[2]]); fh /= np.linalg.norm(fh)
            cam = Camera(self.cam0.pos + self.cam0.right * d[0] + np.array([0, 1.0, 0]) * d[1] + fh * d[2],
                         self.cam0.yaw, self.cam0.pitch, self.cam0.vfov, self.cam0.W, self.cam0.H)
        figs, balls = self.figures_balls(t)
        tree = val(self.tree, t)
        if tree is not None and self.tree_sun_h is not None:
            tree = dict(tree)
            tree['sun_h'] = val(self.tree_sun_h, t)
        pet = None
        pa = val(self.petals, t)
        if pa is not None and tree is not None:
            if self._pet is None or abs(self._pet_age - tree.get('age', 1.0)) > 0.03:
                self._pet = Petals(W_.tree, tree.get('age', 1.0), mound)
                self._pet_age = tree.get('age', 1.0)
            amt, kwp = pa if isinstance(pa, tuple) else (pa, {})
            pet = (self._pet, amt, kwp)
        extra = []
        if self.props is not None:
            extra += self.props(self, cam, L, t)
        if self.extra is not None:
            extra += self.extra(self, cam, L, t, img)
        Lg, Ls = L, None
        gs = val(self.ground_sun, t)
        if gs is not None:
            Lg = dict(L)
            Lg['sunCol'] = L['sunCol'] * np.float32(gs)
            Ls = L
            ks = val(self.sky_sun, t)
            if ks != 1.0:                       # ceu fechado: arvore e bolas tambem sem sol direto
                Ls = dict(L)
                Ls['sunCol'] = L['sunCol'] * np.float32(ks)
        light_key = None if (self.tl or gs is not None) else ('static', round(float(L['el']), 2), round(val(self.dry, t), 3))
        if gs is not None and not self.tl:
            light_key = ('gs', round(float(gs), 3), round(float(L['el']), 3))
        W_.foreground(img, cam, Lg, t, occ, figures=figs, balls=balls, tree=tree, dry=val(self.dry, t),
                      wind=val(self.wind, t), light_key=light_key, petals=pet, extra=extra,
                      ball_kw=val(self.ball_kw, t), fig_kw=val(self.fig_kw, t), L_sky=Ls)
        g = self.grade
        expo = exposure_for(L['el']) * val(self.expo_mul, t)
        if 'expo' in g:
            expo = val(g['expo'], t)
        disp = W_.finish(img, L, info, occ, expo=expo, rays=val(self.rays, t), glow=val(self.glow, t),
                         flare=val(self.flare, t), bloom_amt=self.bloom_amt, sat=val(g.get('sat', self.sat), t),
                         grain_amt=self.grain_amt, rng=self.rng, gain=val(g.get('gain', (1, 1, 1)), t),
                         lift=val(g.get('lift', (0, 0, 0)), t), contrast=g.get('contrast', 1.0), vis_soft=self.vis_soft)
        if self.post is not None:
            disp = self.post(self, disp, t)
        return finish_frame(disp, t, self.dur, self.fade_in, self.fade_out, self.shake, self.seed, val(self.zoom, t))


def finish_frame(disp, t, dur, fade_in=0.0, fade_out=0.0, shake=0.0, seed=0, zoom=None):
    f = 1.0
    if fade_in > 0:
        f *= float(smooth(0.0, fade_in, t))
    if fade_out > 0:
        f *= float(smooth(dur, dur - fade_out, t))
    if f < 0.999:
        disp = disp * np.float32(f)
    u8 = to_u8(disp)
    if shake > 0 or (zoom is not None and zoom != 1.0):
        s = shake * CFG.S
        dx = s * (np.sin(t * 1.31 + seed) * 0.6 + np.sin(t * 2.9 + 1.7 * seed) * 0.25 + np.sin(t * 0.53 + 0.4) * 0.8)
        dy = s * (np.sin(t * 1.07 + 2.1 + seed) * 0.6 + np.sin(t * 3.3 + 0.3) * 0.2 + np.sin(t * 0.47 + 1.9) * 0.7)
        H, W = u8.shape[:2]
        k = (1.0 + 2.6 * s * 2 / H) * (zoom or 1.0)
        M = np.float32([[k, 0, dx - (k - 1) * W / 2], [0, k, dy - (k - 1) * H / 2]])
        u8 = cv2.warpAffine(u8, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return u8


# ---------------- renderizacao ----------------
def ffmpeg_writer(path, W, H, crf=17, preset='medium'):
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (W, H),
           '-r', str(FPS), '-i', '-', '-an',
           '-vf', 'scale=in_range=pc:out_range=tv:out_color_matrix=bt709:flags=accurate_rnd+full_chroma_int,format=yuv420p',
           '-c:v', 'libx264', '-crf', str(crf), '-preset', preset,
           '-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
           '-color_range', 'tv', '-movflags', '+faststart', '-g', '60', '-x264-params', 'keyint=60:min-keyint=30',
           '-video_track_timescale', '15360', path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def render_shot(shot, out_dir, crf=17, preset='medium', frames=None, log=None):
    os.makedirs(out_dir, exist_ok=True)
    final = os.path.join(out_dir, shot.name + '.mp4')
    tmp = os.path.join(out_dir, shot.name + '.tmp.mp4')
    shot.setup()
    W, H = CFG.W, CFG.H
    p = ffmpeg_writer(tmp, W, H, crf, preset)
    t0 = time.time()
    rng = range(shot.n) if frames is None else frames
    for k, i in enumerate(rng):
        fr = shot.frame(i)
        if fr.shape[0] != H or fr.shape[1] != W:
            fr = cv2.resize(fr, (W, H))
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
        if log and (k % 30 == 0):
            el = time.time() - t0
            log('%s %d/%d  %.2fs/quadro' % (shot.name, k + 1, len(rng), el / (k + 1)))
    p.stdin.close()
    rc = p.wait()
    if rc != 0:
        raise RuntimeError('ffmpeg falhou em ' + shot.name)
    os.replace(tmp, final)
    return final


def preview(shot, times, path, cols=3, tile=(960, 540)):
    shot.setup()
    tiles = []
    for t in times:
        i = int(round(t * FPS))
        t0 = time.time()
        fr = shot.frame(i)
        el = time.time() - t0
        im = cv2.resize(fr, tile, interpolation=cv2.INTER_AREA)
        cv2.putText(im, '%s t=%.2f (%.2fs)' % (shot.name, t, el), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)]
    cv2.imwrite(path, np.ascontiguousarray(np.vstack(rows)[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 88])
