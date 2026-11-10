# still-base (2560x1440) e miniatura (1280x720) com titulo e monstrinho
import sys, os
sys.path.insert(0, '/home/claude/filme/src')
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
import thumb, cards
from core import *
OUT = '/home/claude/filme/final'
P = dict(phi=70.0, dist=2.9, height=0.45, vfov=60.0, target_h=1.95, yaw_off=-14.0, radius=0.09)
T = float(os.environ.get('THUMB_T', '4.7'))
TITLE = os.environ.get('THUMB_TITLE', 'ISSO É|MALABARISMO').split('|')
still_path = os.path.join(OUT, 'still-base-miniatura.png')
if not os.path.exists(still_path) or os.environ.get('THUMB_REDO'):
    sh = thumb.thumb_shot('torre', **P); sh.setup()
    fr = sh.frame(int(round(T * FPS)))
    cv2.imwrite(still_path, np.ascontiguousarray(fr[..., ::-1]))
still = cv2.imread(still_path)[..., ::-1]
H0, W0 = still.shape[:2]
img = cv2.resize(still, (1280, 720), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
W, H = 1280, 720
# leve escurecimento a esquerda, atras do titulo
xx = np.linspace(0, 1, W)[None, :]
shade = 1.0 - 0.42 * np.clip(1 - xx / 0.62, 0, 1) ** 1.3
# miniatura pede mais luz e cor do que o quadro do filme
img = np.clip(img * 1.22, 0, 1)
g = img.mean(2, keepdims=True)
img = np.clip(g + (img - g) * 1.14, 0, 1)
img *= shade[..., None]
img = np.clip((img - 0.5) * 1.08 + 0.5, 0, 1)
# titulo em NeoRetro, maiusculas
font_path = os.path.join(BRAND, 'NeoRetroFillPT.ttf')
size = 112
y = 150
for k, line in enumerate(TITLE):
    m, asc = cards.text_mask(line, font_path, size, 0.03)
    while m.shape[1] > 0.505 * W:
        size -= 4
        m, asc = cards.text_mask(line, font_path, size, 0.03)
    x0 = 58
    sh_ = cv2.GaussianBlur(m, (0, 0), 7)                         # sombra suave so no texto (nao no logo)
    hh, ww = m.shape
    reg = img[y:y + hh, x0:x0 + ww]
    reg *= (1 - 0.55 * sh_[..., None])
    reg[:] = reg * (1 - m[..., None]) + m[..., None]
    y += int(size * 1.16)
# monstrinho verde no canto superior direito, sem bola branca atras
masc, word, vb = cards.load_logo()
allp = np.concatenate([p for _, subs in masc for p in subs])
x_a, y_a, x_b, y_b = allp[:, 0].min(), allp[:, 1].min(), allp[:, 0].max(), allp[:, 1].max()
mh = 150.0
k = mh / (y_b - y_a)
ox = W - 44 - (x_b - x_a) * k - x_a * k
oy = 36 - y_a * k
can = (img * 255 + 0.5).astype(np.uint8)
can = np.ascontiguousarray(can)
for fillc, subs in masc:
    c = cards.VERDE if fillc == '#22A557' else cards.BRANCO
    col = tuple(int(v) for v in to_u8(np.asarray(c, np.float32)))
    pts = [np.round(np.stack([p[:, 0] * k + ox, p[:, 1] * k + oy], 1) * 16).astype(np.int32).reshape(-1, 1, 2) for p in subs]
    cv2.fillPoly(can, pts, col, lineType=cv2.LINE_AA, shift=4)
cv2.imwrite(os.path.join(OUT, 'miniatura-1280x720.jpg'), np.ascontiguousarray(can[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 93])
cv2.imwrite(os.path.join(OUT, 'still-base-miniatura.jpg'), np.ascontiguousarray(still[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 94])
print('ok', still.shape)
