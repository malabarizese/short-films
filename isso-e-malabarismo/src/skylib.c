// skylib.c — atmosfera fisica (Rayleigh+Mie+ozonio) e nuvens volumetricas.
// Unidades: km. Compilar: gcc -O3 -march=native -ffast-math -fopenmp -shared -fPIC skylib.c -o skylib.so -lm
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define PI 3.14159265358979f

static float Rg = 6360.0f, Rt = 6460.0f;
static float bR[3] = {5.802e-3f, 13.558e-3f, 33.1e-3f};
static float bMs = 3.996e-3f, bMe = 4.44e-3f;
static float bO[3] = {0.650e-3f, 1.881e-3f, 0.085e-3f};
static float HR = 8.0f, HM = 1.2f, gM = 0.80f;

#define TW 256
#define TH 64
static float tlut[TH * TW * 3];
static float Hatm;

static inline float clampf(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }
static inline float smooth(float a, float b, float x) {
    float t = clampf((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

static void od_to_top(float r, float mu, float od[3]) {
    float disc = r * r * (mu * mu - 1.0f) + Rt * Rt;
    float d = -r * mu + sqrtf(fmaxf(disc, 0.0f));
    int N = 80;
    float dt = d / N, sR = 0, sM = 0, sO = 0;
    for (int i = 0; i < N; i++) {
        float t = (i + 0.5f) * dt;
        float rr = sqrtf(r * r + t * t + 2.0f * r * mu * t);
        float h = fmaxf(rr - Rg, 0.0f);
        sR += expf(-h / HR);
        sM += expf(-h / HM);
        sO += fmaxf(0.0f, 1.0f - fabsf(h - 25.0f) / 15.0f);
    }
    od[0] = sR * dt; od[1] = sM * dt; od[2] = sO * dt;
}

void atm_init(float mie_mult, float ozone_mult, float mie_g, float mie_h) {
    bMs = 3.996e-3f * mie_mult; bMe = 4.44e-3f * mie_mult;
    bO[0] = 0.650e-3f * ozone_mult; bO[1] = 1.881e-3f * ozone_mult; bO[2] = 0.085e-3f * ozone_mult;
    gM = mie_g; HM = mie_h;
    Hatm = sqrtf(Rt * Rt - Rg * Rg);
    for (int j = 0; j < TH; j++)
        for (int i = 0; i < TW; i++) {
            float xr = (j + 0.5f) / TH, xmu = (i + 0.5f) / TW;
            float rho = Hatm * xr;
            float r = sqrtf(rho * rho + Rg * Rg);
            float dmin = Rt - r, dmax = rho + Hatm;
            float d = dmin + xmu * (dmax - dmin);
            float mu = d < 1e-6f ? 1.0f : (Hatm * Hatm - rho * rho - d * d) / (2.0f * r * d);
            mu = clampf(mu, -1.0f, 1.0f);
            float od[3];
            od_to_top(r, mu, od);
            for (int c = 0; c < 3; c++)
                tlut[(j * TW + i) * 3 + c] = expf(-(bR[c] * od[0] + bMe * od[1] + bO[c] * od[2]));
        }
}

// transmitancia do ponto (r, mu) ate o topo, com sombra da Terra suavizada
static inline void trans(float r, float mu, float T[3]) {
    float rho = sqrtf(fmaxf(r * r - Rg * Rg, 0.0f));
    float muh = -rho / r;
    float vis = smooth(muh - 0.006f, muh + 0.004f, mu);
    if (vis <= 0.0f) { T[0] = T[1] = T[2] = 0.0f; return; }
    float m = fmaxf(mu, muh);
    float disc = r * r * (m * m - 1.0f) + Rt * Rt;
    float d = fmaxf(-r * m + sqrtf(fmaxf(disc, 0.0f)), 0.0f);
    float dmin = Rt - r, dmax = rho + Hatm;
    float fx = clampf((d - dmin) / (dmax - dmin), 0.0f, 1.0f) * TW - 0.5f;
    float fy = clampf(rho / Hatm, 0.0f, 1.0f) * TH - 0.5f;
    fx = clampf(fx, 0.0f, TW - 1.001f); fy = clampf(fy, 0.0f, TH - 1.001f);
    int ix = (int)fx, iy = (int)fy;
    float ax = fx - ix, ay = fy - iy;
    const float *p00 = &tlut[(iy * TW + ix) * 3], *p10 = p00 + 3, *p01 = p00 + TW * 3, *p11 = p01 + 3;
    for (int c = 0; c < 3; c++)
        T[c] = vis * ((p00[c] * (1 - ax) + p10[c] * ax) * (1 - ay) + (p01[c] * (1 - ax) + p11[c] * ax) * ay);
}

void sun_trans(float h, float mu, float *T) { trans(Rg + h, mu, T); }

static inline float phaseM(float c, float g) {
    float g2 = g * g;
    return 3.0f / (8.0f * PI) * ((1.0f - g2) * (1.0f + c * c)) / ((2.0f + g2) * powf(1.0f + g2 - 2.0f * g * c, 1.5f));
}
static inline float phaseHG(float c, float g) {
    float g2 = g * g;
    return (1.0f - g2) / (4.0f * PI * powf(1.0f + g2 - 2.0f * g * c, 1.5f));
}

// ceu: out[H*W*3]. basis = right(3), up(3), fwd(3). sun = direcao unitaria.
void sky_render(float *out, int W, int H, const float *basis, float tanx, float tany, float h0,
                const float *sun, float sunE, int N, float msK, const float *msTint, float msLift, const float *tw) {
    const float r0 = Rg + h0;
#pragma omp parallel for schedule(dynamic, 4)
    for (int j = 0; j < H; j++) {
        for (int i = 0; i < W; i++) {
            float u = ((i + 0.5f) / W * 2.0f - 1.0f) * tanx, v = (1.0f - (j + 0.5f) / H * 2.0f) * tany;
            float dx = basis[6] + u * basis[0] + v * basis[3];
            float dy = basis[7] + u * basis[1] + v * basis[4];
            float dz = basis[8] + u * basis[2] + v * basis[5];
            float il = 1.0f / sqrtf(dx * dx + dy * dy + dz * dz);
            dx *= il; dy *= il; dz *= il;
            float b = r0 * dy;
            float dg = b * b - (r0 * r0 - Rg * Rg);
            float tend;
            if (dy < 0.0f && dg > 0.0f) tend = -b - sqrtf(dg);
            else tend = -b + sqrtf(fmaxf(b * b - (r0 * r0 - Rt * Rt), 0.0f));
            float cv = dx * sun[0] + dy * sun[1] + dz * sun[2];
            float pR = 3.0f / (16.0f * PI) * (1.0f + cv * cv);
            float pM = phaseM(cv, gM);
            float tR = 0, tM = 0, tO = 0;
            float L[3] = {0, 0, 0};
            for (int k = 0; k < N; k++) {
                float a0 = (float)k / N, a1 = (k + 1.0f) / N, am = (k + 0.5f) / N;
                float t = tend * am * am, dt = tend * (a1 * a1 - a0 * a0);
                float px = dx * t, py = r0 + dy * t, pz = dz * t;
                float r = sqrtf(px * px + py * py + pz * pz);
                float h = fmaxf(r - Rg, 0.0f);
                float dR = expf(-h / HR), dM = expf(-h / HM), dO = fmaxf(0.0f, 1.0f - fabsf(h - 25.0f) / 15.0f);
                tR += dR * dt * 0.5f; tM += dM * dt * 0.5f; tO += dO * dt * 0.5f;
                float mus = (px * sun[0] + py * sun[1] + pz * sun[2]) / r;
                float Ts[3], Tm[3];
                trans(r, mus, Ts);
                trans(fminf(r + msLift, Rt - 0.05f), mus, Tm);
                for (int c = 0; c < 3; c++) {
                    float Tv = expf(-(bR[c] * tR + bMe * tM + bO[c] * tO));
                    float s1 = Ts[c] * (bR[c] * dR * pR + bMs * dM * pM);
                    float s2 = (msK * msTint[c] * Tm[c] + tw[c]) * (bR[c] * dR + bMs * dM) * (1.0f / (4.0f * PI));
                    L[c] += Tv * (s1 + s2) * dt;
                }
                tR += dR * dt * 0.5f; tM += dM * dt * 0.5f; tO += dO * dt * 0.5f;
            }
            float *o = out + (j * W + i) * 3;
            o[0] = L[0] * sunE; o[1] = L[1] * sunE; o[2] = L[2] * sunE;
        }
    }
}

// ---------------- nuvens ----------------
static const float *g_base = 0, *g_detail = 0, *g_weather = 0;
static int NB = 0, ND = 0, NWt = 0;

void clouds_set_noise(const float *base, int nb, const float *detail, int nd, const float *weather, int nw) {
    g_base = base; NB = nb; g_detail = detail; ND = nd; g_weather = weather; NWt = nw;
}

static inline float tex3(const float *tex, int N, float x, float y, float z) {
    x *= N; y *= N; z *= N;
    float fx = floorf(x), fy = floorf(y), fz = floorf(z);
    float ax = x - fx, ay = y - fy, az = z - fz;
    int m = N - 1;
    int x0 = ((int)fx) & m, y0 = ((int)fy) & m, z0 = ((int)fz) & m;
    int x1 = (x0 + 1) & m, y1 = (y0 + 1) & m, z1 = (z0 + 1) & m;
    const float *s0 = tex + (size_t)z0 * N * N, *s1 = tex + (size_t)z1 * N * N;
    float c00 = s0[y0 * N + x0] * (1 - ax) + s0[y0 * N + x1] * ax;
    float c10 = s0[y1 * N + x0] * (1 - ax) + s0[y1 * N + x1] * ax;
    float c01 = s1[y0 * N + x0] * (1 - ax) + s1[y0 * N + x1] * ax;
    float c11 = s1[y1 * N + x0] * (1 - ax) + s1[y1 * N + x1] * ax;
    return (c00 * (1 - ay) + c10 * ay) * (1 - az) + (c01 * (1 - ay) + c11 * ay) * az;
}
static inline void tex2w(float x, float y, float *a, float *b) {
    int N = NWt, m = N - 1;
    x *= N; y *= N;
    float fx = floorf(x), fy = floorf(y);
    float ax = x - fx, ay = y - fy;
    int x0 = ((int)fx) & m, y0 = ((int)fy) & m, x1 = (x0 + 1) & m, y1 = (y0 + 1) & m;
    const float *p00 = g_weather + (y0 * N + x0) * 2, *p10 = g_weather + (y0 * N + x1) * 2;
    const float *p01 = g_weather + (y1 * N + x0) * 2, *p11 = g_weather + (y1 * N + x1) * 2;
    *a = (p00[0] * (1 - ax) + p10[0] * ax) * (1 - ay) + (p01[0] * (1 - ax) + p11[0] * ax) * ay;
    *b = (p00[1] * (1 - ax) + p10[1] * ax) * (1 - ay) + (p01[1] * (1 - ax) + p11[1] * ax) * ay;
}

typedef struct {
    float hb, ht;
    float bs, ds, ws;          // escalas (1/periodo km)
    float bwx, bwz, evol;      // offsets ruido base
    float dwx, dwy, dwz;       // offsets detalhe
    float wwx, wwz;            // offsets weather
    float coverage, cov_soft, type, type_var;
    float sigma, erode, dens_mult;
    float tmax;
    float sun_boost, amb_boost, powder, farK;
    float cx, cz;              // posicao da camera no mundo (km)
    float lcmax;
} CP;

static inline float hgrad(float h, float type) {
    float st = smooth(0.0f, 0.07f, h) * smooth(0.32f, 0.12f, h);
    float sc = smooth(0.0f, 0.10f, h) * smooth(0.70f, 0.25f, h);
    float cu = smooth(0.0f, 0.12f, h) * smooth(1.0f, 0.42f, h);
    float a = clampf(type * 2.0f, 0.0f, 1.0f), b = clampf(type * 2.0f - 1.0f, 0.0f, 1.0f);
    return (st * (1 - a) + sc * a) * (1 - b) + cu * b;
}

static inline float local_cov(const CP *P, float x, float z, float *wt) {
    float wc;
    tex2w(x * P->ws + P->wwx, z * P->ws + P->wwz, &wc, wt);
    return smooth(1.0f - P->coverage - P->cov_soft, 1.0f - P->coverage + P->cov_soft, wc);
}

static inline float dens(const CP *P, float x, float z, float alt, int full) {
    float h = (alt - P->hb) / (P->ht - P->hb);
    if (h <= 0.0f || h >= 1.0f) return 0.0f;
    float wt;
    float lc = local_cov(P, x, z, &wt) * P->lcmax;
    if (lc <= 0.002f) return 0.0f;
    float type = clampf(P->type + (wt - 0.5f) * P->type_var, 0.0f, 1.0f);
    float hg = hgrad(h, type);
    float bn = tex3(g_base, NB, x * P->bs + P->bwx, alt * P->bs + P->evol, z * P->bs + P->bwz);
    float d = (bn * hg - (1.0f - lc)) / lc;
    if (d <= 0.0f) return 0.0f;
    d *= lc;
    if (full) {
        float dn = tex3(g_detail, ND, x * P->ds + P->dwx, alt * P->ds + P->dwy, z * P->ds + P->dwz);
        float dm = dn + (1.0f - 2.0f * dn) * clampf(h * 5.0f, 0.0f, 1.0f);
        float e = dm * P->erode;
        d = (d - e) / (1.0f - e);
        if (d <= 0.0f) return 0.0f;
    }
    return d * P->dens_mult * (0.45f + 1.1f * h);
}

static inline uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}

// out: H*W*5 = rgb (premultiplicado), alpha, distancia media (km)
void clouds_render(float *out, int W, int H, const float *basis, float tanx, float tany, float h0,
                   const float *sun, const float *sunCol, const float *ambTop, const float *ambBot,
                   const float *params, int N, int NL, int seed) {
    CP P;
    memcpy(&P, params, sizeof(CP));
    const float r0 = Rg + h0;
    const float Ri = Rg + P.hb, Ro = Rg + P.ht;
    const float lstep[6] = {0.03f, 0.06f, 0.11f, 0.20f, 0.36f, 0.60f};
#pragma omp parallel for schedule(dynamic, 2)
    for (int j = 0; j < H; j++) {
        for (int i = 0; i < W; i++) {
            float *o = out + ((size_t)j * W + i) * 5;
            o[0] = o[1] = o[2] = o[3] = 0.0f; o[4] = P.tmax;
            float u = ((i + 0.5f) / W * 2.0f - 1.0f) * tanx, v = (1.0f - (j + 0.5f) / H * 2.0f) * tany;
            float dx = basis[6] + u * basis[0] + v * basis[3];
            float dy = basis[7] + u * basis[1] + v * basis[4];
            float dz = basis[8] + u * basis[2] + v * basis[5];
            float il = 1.0f / sqrtf(dx * dx + dy * dy + dz * dz);
            dx *= il; dy *= il; dz *= il;
            float b = r0 * dy;
            float dg = b * b - (r0 * r0 - Rg * Rg);
            if (dy < 0.0f && dg > 0.0f) continue;   // bate no chao
            float t0 = -b + sqrtf(fmaxf(b * b - (r0 * r0 - Ri * Ri), 0.0f));
            float t1 = -b + sqrtf(fmaxf(b * b - (r0 * r0 - Ro * Ro), 0.0f));
            if (t0 > P.tmax) continue;
            if (t1 > P.tmax) t1 = P.tmax;
            float cv = dx * sun[0] + dy * sun[1] + dz * sun[2];
            float jit = (hash32((uint32_t)(i * 1973 + j * 9277 + seed * 26699)) & 0xffff) / 65536.0f;
            float T = 1.0f, L[3] = {0, 0, 0}, sw = 0.0f, swt = 0.0f;
            float rel = 0.55f / (float)N;            // passo relativo a distancia
            float t = t0, dt = fmaxf(0.045f, t * rel);
            t += dt * jit;
            for (int k = 0; k < 400 && t < t1; k++) {
                dt = fmaxf(0.045f, t * rel);
                float px = dx * t, py = r0 + dy * t, pz = dz * t;
                float alt = sqrtf(px * px + py * py + pz * pz) - Rg;
                float wx = px + P.cx, wz = pz + P.cz;
                int full = t < 70.0f;
                float d = dens(&P, wx, wz, alt, full);
                if (d <= 0.0005f) { t += dt; continue; }
                float fade = 1.0f - smooth(P.tmax * 0.6f, P.tmax, t);
                d *= fade;
                float h = (alt - P.hb) / (P.ht - P.hb);
                float taus = 0.0f, ll = 0.0f;
                for (int q = 0; q < NL; q++) {
                    ll += lstep[q];
                    float sx = px + sun[0] * ll, sy = py + sun[1] * ll, sz = pz + sun[2] * ll;
                    float salt = sqrtf(sx * sx + sy * sy + sz * sz) - Rg;
                    taus += dens(&P, sx + P.cx, sz + P.cz, salt, q < 2) * lstep[q];
                }
                if (P.farK > 0.0f) {
                    float hl = sqrtf(sun[0] * sun[0] + sun[2] * sun[2]) + 1e-4f;
                    float hx = sun[0] / hl, hz = sun[2] / hl, wt;
                    float f = local_cov(&P, wx + hx * 3.0f, wz + hz * 3.0f, &wt) * 0.5f +
                              local_cov(&P, wx + hx * 8.0f, wz + hz * 8.0f, &wt) * 0.3f +
                              local_cov(&P, wx + hx * 18.0f, wz + hz * 18.0f, &wt) * 0.2f;
                    taus += f * P.farK * hl / (fabsf(sun[1]) + 0.12f) * 0.05f;
                }
                float e = P.sigma * taus;
                float lum = 0.0f, a = 1.0f, bb = 1.0f, c = 1.0f;
                for (int oc = 0; oc < 3; oc++) {
                    lum += bb * expf(-e * a) * (0.72f * phaseHG(cv, 0.78f * c) + 0.28f * phaseHG(cv, -0.28f * c));
                    a *= 0.42f; bb *= 0.55f; c *= 0.55f;
                }
                float pw = 1.0f - P.powder * expf(-P.sigma * d * 0.12f) * (0.5f + 0.5f * (1.0f - fabsf(cv)));
                lum *= pw * P.sun_boost;
                float ao = (0.35f + 0.65f * h) * (0.55f + 0.45f * expf(-e * 0.35f));
                float st = expf(-P.sigma * d * dt);
                float wgt = T * (1.0f - st);
                for (int cc = 0; cc < 3; cc++) {
                    float amb = (ambBot[cc] + (ambTop[cc] - ambBot[cc]) * h) * P.amb_boost * ao;
                    L[cc] += wgt * (sunCol[cc] * lum + amb);
                }
                sw += wgt; swt += wgt * t;
                T *= st;
                if (T < 0.005f) { T = 0.0f; break; }
                t += dt;
            }
            o[0] = L[0]; o[1] = L[1]; o[2] = L[2]; o[3] = 1.0f - T;
            o[4] = sw > 1e-5f ? swt / sw : P.tmax;
        }
    }
}

// sombra de nuvens no chao: profundidade optica vertical aproximada (para sombras moveis)
void cloud_shadow_map(float *out, int W, int H, float x0, float z0, float x1, float z1, const float *params) {
    CP P;
    memcpy(&P, params, sizeof(CP));
#pragma omp parallel for
    for (int j = 0; j < H; j++)
        for (int i = 0; i < W; i++) {
            float x = x0 + (x1 - x0) * (i + 0.5f) / W, z = z0 + (z1 - z0) * (j + 0.5f) / H;
            float s = 0.0f;
            for (int k = 0; k < 6; k++) {
                float alt = P.hb + (P.ht - P.hb) * (k + 0.5f) / 6.0f;
                s += dens(&P, x, z, alt, 0);
            }
            out[j * W + i] = s * (P.ht - P.hb) / 6.0f * P.sigma;
        }
}

// ---------------- cirrus (camada fina) ----------------
static const float *g_cirrus = 0; static int NCI = 0;
void cirrus_set(const float *tex, int n) { g_cirrus = tex; NCI = n; }
static inline float tex2c(float x, float y) {
    int N = NCI, m = N - 1;
    x *= N; y *= N;
    float fx = floorf(x), fy = floorf(y);
    float ax = x - fx, ay = y - fy;
    int x0 = ((int)fx) & m, y0 = ((int)fy) & m, x1 = (x0 + 1) & m, y1 = (y0 + 1) & m;
    return (g_cirrus[y0 * N + x0] * (1 - ax) + g_cirrus[y0 * N + x1] * ax) * (1 - ay) +
           (g_cirrus[y1 * N + x0] * (1 - ax) + g_cirrus[y1 * N + x1] * ax) * ay;
}
// out: H*W*4 = rgb premult, alpha.  prm: alt, scale(1/km), offx, offz, amount(0..1), soft, opacity, rot(rad), stretch, sunBoost, ambBoost
void cirrus_render(float *out, int W, int H, const float *basis, float tanx, float tany, float h0,
                   const float *sun, const float *sunCol, const float *amb, const float *prm) {
    const float r0 = Rg + h0, Rc = Rg + prm[0];
    const float cr = cosf(prm[7]), sr = sinf(prm[7]);
#pragma omp parallel for schedule(dynamic, 4)
    for (int j = 0; j < H; j++)
        for (int i = 0; i < W; i++) {
            float *o = out + ((size_t)j * W + i) * 4;
            o[0] = o[1] = o[2] = o[3] = 0.0f;
            float u = ((i + 0.5f) / W * 2.0f - 1.0f) * tanx, v = (1.0f - (j + 0.5f) / H * 2.0f) * tany;
            float dx = basis[6] + u * basis[0] + v * basis[3];
            float dy = basis[7] + u * basis[1] + v * basis[4];
            float dz = basis[8] + u * basis[2] + v * basis[5];
            float il = 1.0f / sqrtf(dx * dx + dy * dy + dz * dz);
            dx *= il; dy *= il; dz *= il;
            float b = r0 * dy;
            float dg = b * b - (r0 * r0 - Rg * Rg);
            if (dy < 0.0f && dg > 0.0f) continue;
            float t = -b + sqrtf(fmaxf(b * b - (r0 * r0 - Rc * Rc), 0.0f));
            if (t > 420.0f) continue;
            float x = dx * t, z = dz * t;
            float xr = (x * cr - z * sr) * prm[1], zr = (x * sr + z * cr) * prm[1] * prm[8];
            float n = tex2c(xr + prm[2], zr + prm[3]);
            float n2 = tex2c(xr * 0.23f + 0.37f + prm[2] * 0.4f, zr * 0.23f + 0.11f);
            float a = smooth(1.0f - prm[4] - prm[5], 1.0f - prm[4] + prm[5], n * 0.65f + n2 * 0.35f);
            a *= prm[6] * (1.0f - smooth(70.0f, 160.0f, t)) * smooth(0.07f, 0.26f, dy);
            // caminho mais longo em angulos rasantes
            float slant = fminf(1.0f / fmaxf(dy + 0.06f, 0.08f), 3.0f);
            float al = 1.0f - expf(-a * slant);
            float cv = dx * sun[0] + dy * sun[1] + dz * sun[2];
            float ph = 0.6f * phaseHG(cv, 0.75f) + 0.4f * phaseHG(cv, -0.2f) + 0.05f;
            for (int c = 0; c < 3; c++) o[c] = al * (sunCol[c] * ph * prm[9] + amb[c] * prm[10]);
            o[3] = al;
        }
}
