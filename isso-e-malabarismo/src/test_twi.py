import sys, numpy as np, cv2
sys.path.insert(0, '/home/claude/filme/src')
import sky
from test_sky import aces, to8
sky.init()
W, H = 480, 270
tany = np.tan(np.radians(18)); tanx = tany * 16 / 9
msK = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
tint = (0.25, 0.5, 1.0)
tiles = []
for yaw, lab in ((0.0, 'E'), (np.pi, 'W')):
    for el in (1.0, -1.5, -4.0, -7.0):
        # sol a oeste (atras da camera quando yaw=0)
        sun = sky.dir_from(np.pi, np.radians(el))
        bas = sky.basis(yaw, np.radians(12.6))
        s = sky.render_sky(W, H, bas, tanx, tany, sun)
        m = s.mean((0, 1))
        expo = min(0.16 / max(np.percentile(s.mean(-1), 60), 1e-5), 60.0)
        im8 = to8(s * expo)
        cv2.putText(im8, '%s el=%.1f x%.1f' % (lab, el, expo), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im8)
        print(lab, el, 'mean', m.round(4), 'expo %.1f' % expo)
cv2.imwrite('/home/claude/filme/work/test_twi.jpg', np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])]), [cv2.IMWRITE_JPEG_QUALITY, 88])
