// Convolution and pooling kernels over contiguous NCHW tensors (float types; one module per type).
//
//   im2col     x [N, C, H, W]            -> cols [N, C*kh*kw, oh*ow]        (zero padding)
//   col2im     cols [N, C*kh*kw, oh*ow]  -> img [N, C, H, W]                (sums overlapping windows; gather form, so the
//                                                                            result is deterministic and needs no atomics)
//   maxpool_fwd / maxpool_bwd: max pooling with the position of each window's winner, like the CPU backend (padding never
//   wins, a NaN wins, the first of equal values wins); the backward pass is also a gather.
#include "common.cuh"

struct Geo {
    int n, c, h, w;
    int kh, kw, sh, sw, ph, pw, dh, dw;
    int oh, ow;
};

#if IS_FLOAT

extern "C" __global__ void im2col(const T* x, T* cols, u64 total, Geo g) {
    const u64 L = (u64)g.oh * g.ow, K = (u64)g.kh * g.kw;
    GRID_STRIDE_LOOP(i, total) {
        const u64 l = i % L;
        const u64 ck = (i / L) % ((u64)g.c * K);
        const u64 b = i / (L * g.c * K);
        const int cc = (int)(ck / K), rem = (int)(ck % K);
        const int r = rem / g.kw, s = rem % g.kw;
        const int oh = (int)(l / g.ow), ow = (int)(l % g.ow);
        const int ih = oh * g.sh + r * g.dh - g.ph, iw = ow * g.sw + s * g.dw - g.pw;
        cols[i] = (ih >= 0 && ih < g.h && iw >= 0 && iw < g.w) ? x[(((u64)b * g.c + cc) * g.h + ih) * g.w + iw] : (T)0;
    }
}

extern "C" __global__ void col2im(const T* cols, T* img, u64 total, Geo g) {
    const u64 L = (u64)g.oh * g.ow, K = (u64)g.kh * g.kw;
    GRID_STRIDE_LOOP(i, total) {
        const int iw = (int)(i % g.w);
        const int ih = (int)((i / g.w) % g.h);
        const int cc = (int)((i / ((u64)g.w * g.h)) % g.c);
        const u64 b = i / ((u64)g.w * g.h * g.c);
        C acc = (C)0;
        for (int r = 0; r < g.kh; ++r) {
            const int t = ih + g.ph - r * g.dh;
            if (t < 0 || t % g.sh) continue;
            const int oh = t / g.sh;
            if (oh >= g.oh) continue;
            for (int s = 0; s < g.kw; ++s) {
                const int u = iw + g.pw - s * g.dw;
                if (u < 0 || u % g.sw) continue;
                const int ow = u / g.sw;
                if (ow >= g.ow) continue;
                acc += ld(cols[(b * g.c * K + ((u64)cc * g.kh + r) * g.kw + s) * L + (u64)oh * g.ow + ow]);
            }
        }
        img[i] = st(acc);
    }
}

extern "C" __global__ void maxpool_fwd(const T* x, T* y, i64* arg, u64 total, Geo g) {
    GRID_STRIDE_LOOP(i, total) {
        const int ow = (int)(i % g.ow);
        const int oh = (int)((i / g.ow) % g.oh);
        const u64 p = i / ((u64)g.ow * g.oh);  // plane = n * c + channel
        const int h0 = max(oh * g.sh - g.ph, 0), h1 = min(oh * g.sh + g.kh - g.ph, g.h);
        const int w0 = max(ow * g.sw - g.pw, 0), w1 = min(ow * g.sw + g.kw - g.pw, g.w);
        const T* plane = x + p * g.h * g.w;
        C best = (C)(-__int_as_float(0x7f800000));
        i64 at = (i64)h0 * g.w + w0;
        for (int ih = h0; ih < h1; ++ih) {
            for (int iw = w0; iw < w1; ++iw) {
                const C v = ld(plane[(u64)ih * g.w + iw]);
                if (v > best || (v != v && best == best)) {
                    best = v;
                    at = (i64)ih * g.w + iw;
                }
            }
        }
        y[i] = st(best);
        if (arg) arg[i] = at;
    }
}

extern "C" __global__ void maxpool_bwd(const T* gy, const i64* arg, T* gx, u64 total, Geo g) {
    GRID_STRIDE_LOOP(i, total) {
        const int iw = (int)(i % g.w);
        const int ih = (int)((i / g.w) % g.h);
        const u64 p = i / ((u64)g.w * g.h);
        const i64 here = (i64)ih * g.w + iw;
        const int nh = ih + g.ph - g.kh + 1, nw = iw + g.pw - g.kw + 1;
        const int oh_lo = nh <= 0 ? 0 : (nh + g.sh - 1) / g.sh, oh_hi = min((ih + g.ph) / g.sh, g.oh - 1);
        const int ow_lo = nw <= 0 ? 0 : (nw + g.sw - 1) / g.sw, ow_hi = min((iw + g.pw) / g.sw, g.ow - 1);
        C acc = (C)0;
        for (int oh = oh_lo; oh <= oh_hi; ++oh) {
            for (int ow = ow_lo; ow <= ow_hi; ++ow) {
                const u64 o = p * g.oh * g.ow + (u64)oh * g.ow + ow;
                if (arg[o] == here) acc += ld(gy[o]);
            }
        }
        gx[i] = st(acc);
    }
}

#endif  // IS_FLOAT
