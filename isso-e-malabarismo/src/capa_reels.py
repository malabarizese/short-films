# capa_reels.py — capa do Reels (1080x1920) do teaser, com titulo e monstrinho
# rodar com FILM_H=1920 FILM_ASPECT=9:16
import sys, os
sys.path.insert(0, '/home/claude/filme/src')
import numpy as np, cv2
import thumb, cards
from core import *
OUT = '/home/claude/filme/final'
P = dict(phi=-8.0, dist=5.6, height=0.4, vfov=92.0, target_h=4.3, yaw_off=0.0, radius=0.10)
T = float(os.environ.get('CAPA_T', '4.5'))
TITLE = os.environ.get('THUMB_TITLE', 'ISSO É|MALABARISMO').split('|')
base_path = os.path.join(OUT, 'still-base-capa-reels.png')
if not os.path.exists(base_path) or os.environ.get('CAPA_REDO'):
    sh = thumb.thumb_shot('torre', **P); sh.setup()
    fr = sh.frame(int(round(T * FPS)))
    cv2.imwrite(base_path, np.ascontiguousarray(fr[..., ::-1]))
img = cv2.imread(base_path)[..., ::-1].astype(np.float32) / 255.0
H, W = img.shape[:2]
img = np.clip(img * 1.18, 0, 1)
g = img.mean(2, keepdims=True)
img = np.clip(g + (img - g) * 1.12, 0, 1)
yy = np.linspace(0, 1, H)[:, None]
img *= (1.0 - 0.38 * np.clip(1 - np.abs(yy - 0.255) / 0.13, 0, 1) ** 1.5)[..., None]   # leve sombra atras do titulo
font_path = os.path.join(BRAND, 'NeoRetroFillPT.ttf')
# monstrinho verde centralizado acima do titulo
masc, word, vb = cards.load_logo()
allp = np.concatenate([p for _, subs in masc for p in subs])
x_a, y_a, x_b, y_b = allp[:, 0].min(), allp[:, 1].min(), allp[:, 0].max(), allp[:, 1].max()
mh = 110.0; k = mh / (y_b - y_a)
ox = W / 2 - (x_b - x_a) * k / 2 - x_a * k
oy = 296 - y_a * k
y = int(296 + mh + 14)
masks = []
size = 140
for line in TITLE:
    m, asc = cards.text_mask(line, font_path, size, 0.03)
    while m.shape[1] > 0.86 * W:
        size -= 4; m, asc = cards.text_mask(line, font_path, size, 0.03)
for line in TITLE:
    m, asc = cards.text_mask(line, font_path, size, 0.03)
    hh, ww = m.shape
    x0 = int(W / 2 - ww / 2)
    sh_ = cv2.GaussianBlur(m, (0, 0), 8)
    reg = img[y:y + hh, x0:x0 + ww]
    reg *= (1 - 0.5 * sh_[..., None])
    reg[:] = reg * (1 - m[..., None]) + m[..., None]
    print('linha', line, 'y', y, y + hh, 'x', x0, x0 + ww)
    y += int(size * 1.16)
can = np.ascontiguousarray((img * 255 + 0.5).astype(np.uint8))
for fillc, subs in masc:
    c = cards.VERDE if fillc == '#22A557' else cards.BRANCO
    col = tuple(int(v) for v in to_u8(np.asarray(c, np.float32)))
    pts = [np.round(np.stack([p[:, 0] * k + ox, p[:, 1] * k + oy], 1) * 16).astype(np.int32).reshape(-1, 1, 2) for p in subs]
    cv2.fillPoly(can, pts, col, lineType=cv2.LINE_AA, shift=4)
cv2.imwrite(os.path.join(OUT, 'capa-reels-1080x1920.jpg'), np.ascontiguousarray(can[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 94])
print('ok', can.shape)
