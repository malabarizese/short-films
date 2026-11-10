import sys, time, numpy as np, cv2
sys.path.insert(0, '/home/claude/filme/src')
import sky

def aces(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)

def to8(img):
    return np.ascontiguousarray((np.clip(aces(img), 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)[..., ::-1])

if __name__ == '__main__':
    t0 = time.time(); sky.init(); print('init', time.time() - t0)
    W, H = 640, 360
    tany = np.tan(np.radians(18)); tanx = tany * 16 / 9
    bas = sky.basis(0.0, np.radians(12.6))
    tiles = []
    cases = [('6.05', 6.05), ('9', 9.0), ('13', 13.0), ('17.5', 17.5), ('17.95', 17.95), ('18.2', 18.2)]
    for name, hr in cases:
        sun = sky.sun_dir(hr)
        el = np.degrees(np.arcsin(sun[1]))
        t0 = time.time()
        s = sky.render_sky(W, H, bas, tanx, tany, sun)
        t1 = time.time()
        sunCol = 20.0 * sky.sun_trans(3.5, sun[1])
        zen = sky.render_sky(8, 8, sky.basis(0, np.radians(80)), 0.3, 0.3, sun).mean((0, 1))
        hor = sky.render_sky(16, 4, sky.basis(0, np.radians(4)), 2.0, 0.05, sun).mean((0, 1))
        P = sky.cloud_params(coverage=0.45)
        c = sky.render_clouds(W, H, bas, tanx, tany, sun, sunCol, zen * 3.0, (zen + hor) * 0.9, P)
        t2 = time.time()
        ci = sky.render_cirrus(W, H, bas, tanx, tany, sun, 20.0 * sky.sun_trans(9.5, sun[1]), zen * 2.0)
        img = s * (1 - ci[..., 3:4]) + ci[..., :3]
        c = sky.soften_far(c)
        a = c[..., 3:4]
        f = np.exp(-c[..., 4:5] / 60.0)
        crgb = c[..., :3] * f + img * a * (1 - f)
        img = img * (1 - a) + crgb
        expo = 0.18 / max(np.percentile(img.mean(-1), 55), 1e-4)
        expo = min(expo, 40.0)
        print(name, 'el %.1f' % el, 'sky %.2fs clouds %.2fs' % (t1 - t0, t2 - t1), 'expo %.2f' % expo, 'sunCol', sunCol.round(2), 'zen', zen.round(3), 'hor', hor.round(3))
        im8 = to8(img * expo)
        cv2.putText(im8, '%s el=%.1f' % (name, el), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im8)
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, 6, 3)]
    cv2.imwrite('/home/claude/filme/work/test_sky.jpg', np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])

