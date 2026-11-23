// Fused optimizer updates, in place on device buffers (f32 / f64 modules). The arithmetic is the CPU backend's, term by term.
#include "common.cuh"

#if DTYPE == DT_F32 || DTYPE == DT_F64

// Adam / AdamW. `decoupled` selects the AdamW weight decay (scale the parameter) over the L2 form (add wd * p to the gradient);
// ibc1 / ibc2 are the reciprocal bias corrections 1 / (1 - beta^t).
extern "C" __global__ void adam_step(T* p, const T* g, T* m, T* v, u64 n, double lr_, double b1_, double b2_, double eps_, double wd_, int decoupled, double ibc1_, double ibc2_) {
    const C lr = (C)lr_, b1 = (C)b1_, b2 = (C)b2_, eps = (C)eps_, wd = (C)wd_, ibc1 = (C)ibc1_, ibc2 = (C)ibc2_;
    GRID_STRIDE_LOOP(i, n) {
        C pv = p[i], gv = g[i];
        if (wd != (C)0) {
            if (decoupled) pv *= (C)1 - lr * wd;
            else gv += pv * wd;
        }
        const C mv = b1 * m[i] + ((C)1 - b1) * gv;
        const C vv = b2 * v[i] + ((C)1 - b2) * gv * gv;
        m[i] = mv;
        v[i] = vv;
        p[i] = pv - lr * (mv * ibc1) / (m_sqrt(vv * ibc2) + eps);
    }
}

// SGD with optional momentum (buffer `buf`, updated in place), weight decay and Nesterov correction.
extern "C" __global__ void sgd_step(T* p, const T* g, T* buf, u64 n, double lr_, double wd_, double momentum_, double dampening_, int nesterov, int first) {
    const C lr = (C)lr_, wd = (C)wd_, mom = (C)momentum_, damp = (C)dampening_;
    GRID_STRIDE_LOOP(i, n) {
        C gv = g[i];
        const C pv = p[i];
        if (wd != (C)0) gv += pv * wd;
        C upd = gv;
        if (mom != (C)0) {
            const C b = first ? gv : mom * buf[i] + ((C)1 - damp) * gv;
            buf[i] = b;
            upd = nesterov ? gv + mom * b : b;
        }
        p[i] = pv - lr * upd;
    }
}

// sum of squares of a contiguous buffer, one partial per block (for gradient-norm clipping: the host adds the partials)
extern "C" __global__ void sumsq_partial(const T* x, double* out, u64 n) {
    __shared__ double sm[1024];
    double acc = 0.0;
    GRID_STRIDE_LOOP(i, n) {
        const double v = (double)x[i];
        acc += v * v;
    }
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = sm[0];
}

#endif
