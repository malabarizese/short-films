# Isso é malabarismo — código-fonte

Filme de 2 min 56 s (2560×1440, 30 fps) gerado inteiramente por código: imagem em Python + C (sem modelos de imagem), trilha
montada nota a nota com amostras de instrumentos de domínio público (CC0).

## Pastas
- `src/` — renderizador e trilha
  - `skylib.c`, `sky.py`, `atmos.py` — céu físico (Rayleigh/Mie/ozônio), nuvens volumétricas, cirros, estrelas, lua
  - `terrain.py`, `meadow.py`, `tree.py` — mar de morros com neblina, relva em camadas com foco, ipê-amarelo que cresce e floresce
  - `juggle.py`, `figure.py` — simulador de siteswap (cascatas, (6x,4)*, db97531, quedas e recolhidas) e o malabarista
  - `scene.py`, `shots.py`, `film.py` — composição, pós (bloom, raios, flare, tonemap) e a lista de planos
  - `eye.py`, `closeups.py`, `cards.py` — olho, mão com a bola, bola na chuva, letreiro e vinheta do monstrinho
  - `timeline.py`, `render_all.py`, `assemble.py`, `finalize.py`, `qc_video.py` — montagem, render em trechos, conferências
  - `smp.py`, `synth.py`, `score.py` — sampler, efeitos sintetizados, partitura e master (−14 LUFS)
- `audio/` — varredura das amostras, eventos de sincronia e análises
- `brand/` — marca Malabarize-se usada na vinheta e no letreiro

## Como refazer
1. `gcc -O3 -march=native -ffast-math -fopenmp -shared -fPIC src/skylib.c -o src/skylib.so -lm`
2. `python3 src/render_all.py <planos separados por vírgula> A` (um mp4 por trecho de 3 s em `out/`; trechos prontos são pulados)
3. `python3 src/assemble.py` (junta sem recodificar)
4. Trilha: baixar o VSCO-2 Community Edition (github.com/sgossner/VSCO-2-CE), ajustar `ROOT` em `src/smp.py`,
   rodar `audio/scan.py`, `audio/events.py` e `src/score.py saida.wav`
5. `python3 src/finalize.py saida.wav filme.mp4`

Os caminhos estão fixos em `/home/claude/filme`; para rodar em outro lugar, troque esse prefixo.
Prévia rápida de um plano: `FILM_H=720 python3 src/run_preview.py <plano> <t1,t2,...>`.

## Créditos da trilha
Amostras: Versilian Studios Chamber Orchestra 2 Community Edition (Sam Gossner / Versilian Studios; piano de armário de
Simon Dalzell / Ivy Audio), licença CC0. Composição, efeitos e mixagem gerados pelo código em `src/score.py`.

## Teaser vertical e miniatura
- Teaser (1080×1920, 37,5 s): `FILM_H=1920 FILM_ASPECT=9:16 FILM_TIMELINE=trailer python3 src/render_all.py <trechos> V out_v`,
  depois `python3 src/score_trailer.py teaser.wav` e `python3 src/assemble_trailer.py teaser.wav teaser.mp4`.
  Os trechos e as câmeras verticais estão em `src/trailer.py`; o cartão final, em `TrailerCard` (`src/cards.py`).
- Capa do Reels: `FILM_H=1920 FILM_ASPECT=9:16 python3 src/capa_reels.py` (1080×1920; título em `THUMB_TITLE`).
- Miniatura: `python3 src/thumb_final.py` (still-base 2560×1440 + miniatura 1280×720; título em `THUMB_TITLE`, linhas separadas por `|`).
