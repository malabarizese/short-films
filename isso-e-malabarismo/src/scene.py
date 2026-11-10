# scene.py — montagem de um quadro do mundo (ceu, serras, relva, arvore, malabarista, bolas, pos)
import numpy as np
import cv2
from core import *
import sky as skylib
import fx
from atmos import Atmos, exposure_for, STARS
from terrain import Ridges
from meadow import Meadow, mound
from tree import Ipe, Petals, BLOOM
from figure import Pose, draw_figure, draw_ball
import juggle

BALL_R = 0.038


def moon_dir_for(hour_after_sunset=0.3):
    """lua cheia nascendo a leste, oposta ao sol."""
    return None


class World:
    def __init__(self, cam, tree_base=(3.6, 2.2), seed=1, meadow_kw=None, ridge_seed=3, sky_div=8, cloud_div=4,
                 with_meadow=True):
        self.cam = cam
        self.atmos = Atmos(cam, sky_div=sky_div, cloud_div=cloud_div)
        self.ridges = Ridges(cam, seed=ridge_seed)
        self.meadow = Meadow(cam, **(meadow_kw or {})) if with_meadow else None
        tb = np.array([tree_base[0], float(mound(tree_base[0], tree_base[1])) - 0.03, tree_base[1]])
        self.tree = Ipe(tb)
        self.tree_base = tb
        self._bg_key = None
        self._cloud_cache = {}

    # ---------- fundo ----------
    def lighting(self, sun, moon=None):
        L = self.atmos.light(sun)
        if moon is not None:
            L['moon_dir'] = np.asarray(moon['dir'], float)
            L['moonCol'] = np.asarray(moon.get('light', (0.05, 0.06, 0.085)), np.float32)
        return L

    def background(self, sun, L, cloud_kw=None, cirrus_kw=None, fog=0.5, vis_km=50.0, stars=None, star_ang=0.0,
                   moon=None, t=0.0, seed=0, cloud_N=56, green=1.0, fog_glow=0.25, clouds=True, milky=1.0,
                   fog_t=0.0, cloud_buf=None, cirrus_buf=None, grad=None):
        at = self.atmos
        if grad is None:
            grad = at.sky_gradient(sun)
        P = None
        if clouds:
            if cloud_buf is None:
                cloud_buf, P = at.clouds(sun, L, cloud_kw, seed=seed, N=cloud_N)
            if cirrus_buf is None and cirrus_kw is not None:
                cirrus_buf = at.cirrus(sun, L, **cirrus_kw)
        if stars is None:
            stars = float(smooth(-4.5, -11.0, L['el']))
        img, calpha, info = at.compose(grad, L, cloud_buf if clouds else None, cirrus_buf if clouds else None,
                                       stars=stars, star_ang=star_ang, t=t, moon=moon, milky=milky)
        hc = at.horizon_colors(grad)
        self.ridges.render(img, L, hc, fog=fog, vis_km=vis_km, green=green, fog_glow=fog_glow, fog_t=fog_t)
        occ = calpha[..., 0].copy()
        for Lr in self.ridges.layers:
            if not Lr['skip']:
                occ[Lr['r0']:Lr['r1']] = np.maximum(occ[Lr['r0']:Lr['r1']], Lr['alpha'])
        info['grad'] = grad
        info['cloud_buf'] = cloud_buf
        info['cirrus_buf'] = cirrus_buf
        info['hc'] = hc
        return img, occ, info

    # ---------- primeiro plano ----------
    def foreground(self, img, cam, L, t, occ=None, figures=(), balls=(), tree=None, dry=0.0, wind=1.0,
                   light_key=None, petals=None, extra=(), shadows=True, ball_kw=None, fig_kw=None, shadow_fig=None,
                   L_sky=None):
        """figures = lista de dict(pose, J, alpha) ; balls = lista de arrays (subamostras de posicao no mundo)
        ou tuplas (array, kw) ; tree = dict(age, leaves, bloom...)"""
        items = []
        sh = []
        Lf = L if L_sky is None else L_sky       # luz plena (acima da linha de sombra) para arvore e bolas
        if shadows and L['el'] > 0.3 and float(np.max(L['sunCol'])) > 1e-3:
            for fg in (figures[-1:] if shadow_fig is None else shadow_fig):
                o = fg['pose'].origin
                r = fg['pose'].root
                sh.append((o[0] + r[0], o[2] + r[1], 0.42, 1.72 - 0.5 * fg['pose'].bend, 0.80))
            if tree is not None:
                wfn, Ht = self.tree.width_at(tree.get('age', 1.0))
                dens = max(tree.get('leaves', 0.0), tree.get('bloom', 0.0) * 0.85, 0.35)
                sh.append((self.tree_base[0], self.tree_base[2], wfn, Ht, 0.78 * dens))
        if self.meadow is not None:
            items += self.meadow.items(cam, L, t, dry=dry, shadows=sh, wind=wind, light_key=light_key)
        masks = []
        zt = float((self.tree_base - cam.pos) @ cam.fwd)
        if tree is not None:
            def tf(img, tree=tree):
                m = self.tree.draw(img, cam, Lf, t=t, wind=wind, **tree)
                if m is not None:
                    masks.append(m)
            items.append((zt, tf))
        if petals is not None:
            pet, amt, kwp = petals
            carp = kwp.get('carpet', 1.0)
            slow = kwp.get('slow', 1.0)
            tp = kwp.get('t', t)
            if amt > 0.005:
                pc = kwp.get('col', BLOOM)
                items.append((zt - 0.02, lambda img: pet.draw(img, cam, L, tp, amount=amt, zmin=zt, slow=slow, col=pc)))
                items.append((0.9, lambda img: pet.draw(img, cam, L, tp, amount=amt, zmax=zt, slow=slow, col=pc)))
            if carp > 0.005 and self.meadow is not None:
                ds = sorted([S.d for S in self.meadow.strips])
                edges = [0.0] + ds + [1e9]
                for da, db in zip(edges[:-1], edges[1:]):
                    if db < 5:
                        continue
                    items.append((db - 0.001, lambda img, da=da, db=db: pet.draw_carpet(img, cam, L, amount=carp, zmin=da, zmax=db, col=kwp.get('col', BLOOM))))
        zfig = None
        for fg in figures:
            o = fg['pose'].origin + np.array([0.0, 0.9, 0.0])
            Rm = fg['J']['_R']
            o = o + Rm @ np.array([fg['pose'].root[0], 0.0, fg['pose'].root[1]])
            zf = float((o - cam.pos) @ cam.fwd)
            zfig = zf

            def ff(img, fg=fg):
                kw = dict(fig_kw or {})
                kw.update(fg.get('kw', {}))
                m = draw_figure(img, cam, fg['J'], L, return_mask=True, alpha=fg.get('alpha', 1.0), **kw)
                if m is not None and fg.get('alpha', 1.0) > 0.9:
                    masks.append(m)
            items.append((zf, ff))
        for be in balls:
            pts, bkw = (be if isinstance(be, tuple) else (be, {}))
            pts = np.atleast_2d(pts)
            zb = float((pts.mean(0) - cam.pos) @ cam.fwd)
            if zfig is not None and abs(zb - zfig) < 0.9:
                zb = zfig + (0.05 if zb > zfig - 0.03 else -0.05)
            bk = dict(ball_kw or {})
            bk.update(bkw)
            items.append((zb, lambda img, pts=pts, bk=bk: draw_ball(img, cam, pts, bk.pop('radius', BALL_R), Lf, **bk)))
        items += list(extra)
        for d, fn in sorted(items, key=lambda e: -e[0]):
            fn(img)
        if occ is not None:
            H, W = occ.shape
            for (x0, y0, a) in masks:
                h, w = a.shape
                xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
                if xb > xa and yb > ya:
                    occ[ya:yb, xa:xb] = np.maximum(occ[ya:yb, xa:xb], a[ya - y0:yb - y0, xa - x0:xb - x0])
            if self.meadow is not None:
                for S in self.meadow.strips:
                    if S.div == 1 and getattr(S, 'al', None) is not None:
                        h, w = S.al.shape
                        xa, ya, xb, yb = max(S.x0, 0), max(S.y0, 0), min(S.x0 + w, W), min(S.y0 + h, H)
                        if xb > xa and yb > ya:
                            occ[ya:yb, xa:xb] = np.maximum(occ[ya:yb, xa:xb], S.al[ya - S.y0:yb - S.y0, xa - S.x0:xb - S.x0])
        return img

    # ---------- pos ----------
    def finish(self, img, L, info, occ, expo=None, rays=0.5, glow=1.0, flare=1.0, bloom_amt=0.10, vig=0.26,
               sat=1.0, grain_amt=0.016, rng=None, gain=(1, 1, 1), lift=(0, 0, 0), contrast=1.0, vis_soft=1.2):
        if expo is None:
            expo = exposure_for(L['el'])
        if 'sun_px' in info:
            px = info['sun_px']
            vis = fx.sun_visibility(occ, px, px[2], vis_soft) if occ is not None else 1.0
            H, W = img.shape[:2]
            # sol fora do quadro: o brilho some aos poucos (sem degrau quando ele cruza a borda)
            d_out = max(-px[0] / W, px[0] / W - 1.0, -px[1] / H, px[1] / H - 1.0, 0.0)
            w_in = 1.0 - float(smooth(0.2, 0.9, d_out))
            if w_in > 0.002:
                if rays > 0:
                    fx.god_rays(img, px, occ, strength=rays * w_in, thresh=0.14 / expo)
                if glow > 0:
                    fx.sun_glow(img, px, L['sunCol'] * 1.7, vis=0.2 + 0.8 * vis, strength=glow * w_in)
                if flare > 0:
                    fx.lens_flare(img, px, L['sunCol'], vis=vis, strength=flare * w_in)
        img = bloom(img, bloom_amt)
        vignette(img, vig)
        disp = tonemap(img, expo, sat=sat, gain=gain, lift=lift, contrast=contrast)
        if grain_amt > 0:
            grain(disp, grain_amt, rng)
        return disp


def juggler_pose(ch, t, origin, yaw=0.0, n_balls=3, breath_t=None):
    """pose do malabarista a partir da coreografia."""
    p = Pose(origin, yaw)
    r = ch.root(t)
    p.root = np.array([r[0], r[1]])
    p.bend = ch.bend(t)
    look_base = {3: 14.0, 5: 30.0, 7: 46.0}.get(n_balls, 20.0)
    active = len(ch.in_air(t)) > 0
    lk = ch.look(t)
    p.head = (look_base if active else 6.0) + lk if p.bend < 0.05 else lk * p.bend
    p.lean = np.radians({3: 1.5, 5: 4.0, 7: 7.5}.get(n_balls, 2.0)) * (1.0 if active else 0.3) + ch.lean(t)
    p.squat = 0.035 + 0.05 * max(ch.pump(t), -0.5) + 0.012 * np.sin(t * 1.3)
    bt = t if breath_t is None else breath_t
    p.breath = np.sin(bt * 2.1)
    for h in (0, 1):
        p.hands[h] = ch.hand_local(h, t)
    return p


def ball_world_fn(ch, b, pose_origin, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    Rm = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    o = np.asarray(pose_origin, float)

    def fn(t):
        return Rm @ ch.ball_local(b, t) + o
    return fn


def walk_pose(dist, origin, yaw, slope=0.0, step=0.74, bag=True, t=0.0):
    """pose de caminhada em funcao da distancia percorrida (m). origin = posicao dos pes (mundo)."""
    p = Pose(origin, yaw)
    ph = (dist / (2 * step)) % 1.0
    p.bag = bag
    p.lean = -np.radians(5.0 + 30.0 * slope)
    p.squat = 0.03 + 0.018 * np.cos(4 * np.pi * ph)
    p.head = 4.0
    sw = []
    for h, off in ((0, 0.0), (1, 0.5)):
        u = (ph + off) % 1.0
        if u < 0.62:      # apoio: pe vai de frente para tras
            z = step * 0.5 * (1 - 2 * u / 0.62)
            y = 0.0
        else:             # balanco
            w = (u - 0.62) / 0.38
            z = step * 0.5 * (-1 + 2 * (w * w * (3 - 2 * w)))
            y = 0.10 * np.sin(np.pi * w)
        sg = -1.0 if h == 0 else 1.0
        p.feet[h] = np.array([sg * 0.085, 0.07 + y + slope * z, z])
        sw.append(-0.20 * np.sin(2 * np.pi * (ph + off)))
    p.arm_swing = (sw[1] * 1.0, sw[0] * 1.0)
    return p
