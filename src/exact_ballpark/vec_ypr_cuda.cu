// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Eidosoma
// The C library's double sin, cos, asin and atan2 on a CUDA device, bit for bit glibc 2.34 aarch64's, and the yaw,
// pitch and roll's three calls (Ball::CalculateYawPitchRoll without dynamical orientation, MOV-729: atan2, asin, and
// asin of sin) as elementwise launches, so exact_ballpark.vec_orient.VecOrientPark(..., ypr="cuda") computes the yaw,
// pitch and roll on the device with no host round trip (a tick can be captured as a CUDA graph). Built and loaded by
// exact_ballpark.vec_ypr_cuda; held to exact_ballpark.dlibm, exact_ballpark.dlibm_inv and the C library itself bit for bit by
// tests/test_movement_vec_ypr_cuda.py.
//
// Provenance. sin and cos are IBM's Accurate Mathematical Library (glibc's sysdeps/ieee754/dbl-64/s_sin.c and
// branred.c), copied from exact_ballpark.vec_modes_cuda's device code (itself transcribed from exact_ballpark.dlibm) with only the constants' names prefixed YL_ for this file's header and two long lines wrapped;
// asin and atan2 are IBM's too
// (e_asin.c with asncs.tbl, root.tbl and powtwo.tbl; e_atan2.c with uatan2.tbl), transcribed from
// exact_ballpark.dlibm_inv (read from libm.so.6's disassembly). All four are LGPL 2.1 inside glibc.
//
// Exactness. Every operation's rounding is pinned as in exact_ballpark.vec_orient_cuda: __dadd_rn, __dsub_rn,
// __dmul_rn, __ddiv_rn in the transcription's order; a fused multiply-add (__fma_rn) exactly where the C library's
// compiled code has one (fmadd, fmsub = a - n m, fnmsub = n m - a, with the disassembly's operands) and nowhere else;
// negation and the sign moves on the bits. No function of CUDA's math library is called; every constant comes from
// the generated header (exact_ballpark.vec_ypr_cuda.tables_header, from dlibm's and dlibm_inv's tables, read out of the
// library), never typed. The build adds -fmad=false, -prec-div=true, -prec-sqrt=true, -ftz=false and no fast math,
// and checks the PTX. A NaN result is NaN, not necessarily with the C library's payload.
//
// The file includes nothing of the package's other kernels and defines everything in namespace xbpy (its header's
// names prefixed YL_ and YI_, include-guarded), so another kernel may include it beside them.

#ifndef EXACT_BALLPARK_VEC_YPR_CUDA_CU
#define EXACT_BALLPARK_VEC_YPR_CUDA_CU

#include <cstdint>
#include <cuda_runtime.h>

#include "vec_ypr_cuda_tables.h"

#define XBPY_ABI 1

namespace xbpy {

// ======================================================================================== arithmetic, rounding pinned

static __device__ __forceinline__ double dadd(double a, double b) { return __dadd_rn(a, b); }
static __device__ __forceinline__ double dsub(double a, double b) { return __dsub_rn(a, b); }
static __device__ __forceinline__ double dmul(double a, double b) { return __dmul_rn(a, b); }
static __device__ __forceinline__ double ddiv(double a, double b) { return __ddiv_rn(a, b); }
static __device__ __forceinline__ double dfma(double a, double b, double c) { return __fma_rn(a, b, c); }
static __device__ __forceinline__ double dfmsub(double n, double m, double a) { return dfma(-n, m, a); }  // a - n m
static __device__ __forceinline__ double dfnmsub(double n, double m, double a) { return dfma(n, m, -a); } // n m - a
static __device__ __forceinline__ uint64_t dbits(double x) { return (uint64_t)__double_as_longlong(x); }
static __device__ __forceinline__ double dfrom(uint64_t u) { return __longlong_as_double((long long)u); }
static __device__ __forceinline__ double dabs_(double x) { return dfrom(dbits(x) & 0x7fffffffffffffffULL); }
static __device__ __forceinline__ double dnan() { return dfrom(0x7ff8000000000000ULL); }
static __device__ __forceinline__ double dcopysign(double a, double s) {
    return dfrom((dbits(a) & 0x7fffffffffffffffULL) | (dbits(s) & 0x8000000000000000ULL));
}

#define YKD(name) dfrom(name)       // a double constant from its bit pattern

#include "glibc_ypr.cuh"   // LGPL-2.1-or-later: the C library's sin, cos, asin and atan2
// ================================================================================================ the launches

// out = f(a) or f(a, b): 0 sin, 1 cos, 2 asin, 3 atan2(a, b) (a = y, b = x), 4 asin(sin(a)) (the roll's yaw delta),
// 5 and 6 sincos's sine and cosine
__global__ void dlibm_kernel(int func, int64_t n, const double* __restrict__ a, const double* __restrict__ b,
                             double* __restrict__ out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    switch (func) {
        case 0: out[i] = g_sin(a[i]); break;
        case 1: out[i] = g_cos(a[i]); break;
        case 2: out[i] = g_asin(a[i]); break;
        case 3: out[i] = g_atan2(a[i], b[i]); break;
        case 4: out[i] = g_asin(g_sin(a[i])); break;
        case 5:
        case 6: {
            double s, c;
            g_sincos(a[i], &s, &c);
            out[i] = func == 5 ? s : c;
            break;
        }
        default: out[i] = dnan(); break;
    }
}

static int launch_check(void) {
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int)e;
}

}  // namespace xbpy

#undef YKD

// ====================================================================================== the C interface (ctypes)

extern "C" {

int xbpy_abi(void) { return XBPY_ABI; }

const char* xbpy_error_string(int code) { return cudaGetErrorString((cudaError_t)code); }

// out = f(a[, b]) for n values on `stream` (torch's current stream): no allocation and no host sync, so a launch can be
// captured in a CUDA graph. b is read by atan2 only (it may be a for the others).
int xbpy_dlibm(int func, int64_t n, const double* a, const double* b, double* out, void* stream, int block) {
    if (n <= 0) return 0;
    if (block <= 0 || block > 1024 || block % 32 != 0 || func < 0 || func > 6) return -1;
    const unsigned grid = (unsigned)((n + block - 1) / block);
    xbpy::dlibm_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(func, n, a, b, out);
    return xbpy::launch_check();
}

// The kernel as loaded on the current device: registers, local memory, the binary and PTX versions.
int xbpy_info(int* regs, int* local_bytes, int* binary_version, int* ptx_version, int* max_threads) {
    cudaFuncAttributes fa;
    const cudaError_t e = cudaFuncGetAttributes(&fa, xbpy::dlibm_kernel);
    if (e != cudaSuccess) return (int)e;
    *regs = fa.numRegs;
    *local_bytes = (int)fa.localSizeBytes;
    *binary_version = fa.binaryVersion;
    *ptx_version = fa.ptxVersion;
    *max_threads = fa.maxThreadsPerBlock;
    return 0;
}

}  // extern "C"

#endif  // EXACT_BALLPARK_VEC_YPR_CUDA_CU
