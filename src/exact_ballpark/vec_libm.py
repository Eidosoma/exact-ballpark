# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (c) 2026 Eidosoma
# Transcribed from the GNU C Library 2.34; see THIRD_PARTY_NOTICES.md.
"""The C library's float functions (expf, logf, sinf, cosf, acosf, atanf, atan2f) and pow on float64 tensors, bit for
bit on the CPU and the GPU: what Destiny's dynamical orientation (MOV-728) and DirectXMath's rotation (MOV-725,
MOV-726) call, for the batched turning of exact_ballpark.vec_orient.

The scalar reference (exact_ballpark.full) calls these through ctypes from the C library itself, as Destiny's compiled
code does: glibc 2.34 on this machine (aarch64, /lib64/libm.so.6, SHA-256 `LIBM_SHA256`). Each function here is that
library's algorithm transcribed from its disassembly, operation for operation, so every multiply-add the compiler fused
is fused here too, and only those (the addresses are in the comments):

- expf, logf, sinf, cosf and pow are ARM's optimized-routines (glibc's sysdeps/ieee754/flt-32/e_expf.c, e_logf.c,
  s_sinf.c, s_cosf.c and dbl-64/e_pow.c, MIT-licensed upstream): computed in double, the float functions rounded to
  float at the end. Their tables (`__exp2f_data`, `__logf_data`, `__sincosf_table`, `__inv_pio4`, `__pow_log_data`)
  were read out of the loaded library at the addresses the code references (below, generated, not typed); pow's exp
  part uses `__exp_data`, which exact_ballpark.vec_exp generates and checks.
- acosf, atanf and atan2f are fdlibm's (flt-32/e_acosf.c, s_atanf.c, e_atan2f.c; Sun's permissive licence), computed
  in float, with GCC's fused multiply-adds on float registers (fmadd, fmsub).

A float value is held in a float64 tensor, as the reference holds it in a Python float; `f32` rounds a double to the
nearest float. Only correctly rounded operations are used: +, -, *, /, sqrt and integer steps on the bits. The double
fused multiply-add is exact_ballpark.vec_exp's exact emulation (Boldo and Melquiond); the float one, `fmaf`, rounds the
exact product plus the addend to odd in double, then to float, which is the correctly rounded result because
53 >= 24 + 2 (the same paper). The 64-bit integer steps of sinf's large-argument reduction are done in 32-bit halves,
so no int64 operation overflows. Divisions have tensor divisors (CUDA divides by a host scalar through its reciprocal).
Nothing is compiled or fused (no torch.compile).

Checked (tests/test_movement_vec_libm.py): the one-argument float functions against the C library over all 2^32 float
bit patterns once, and on samples with every branch in the suite; atan2f and
pow on samples over every branch; the CPU against the GPU. A NaN result is NaN, not necessarily with the C library's
payload; errno and the floating-point flags are not modelled.
"""

from __future__ import annotations

import struct

import torch

from .vec_exp import C2, C3, C4, C5, F64, I64, INVLN2N, NEGLN2HIN, NEGLN2LON, _table, add_odd, fma, split

F32, I32 = torch.float32, torch.int32
LIBM_SHA256 = "b4fa1652bf3a7267d0fa4256fade78f0df2954b7b39ee842ec410e0e7b0f6c86"   # /usr/lib64/libm.so.6
_H = float.fromhex
INF, NAN = float("inf"), float("nan")
SIGN_BIT = -(1 << 63)                                   # 0x8000000000000000 as an int64


def _fh(u: int) -> float:
    """The float with bit pattern u, as a Python float."""
    return struct.unpack("<f", struct.pack("<I", u))[0]


# ---- generated from this machine's libm.so.6: tables read at the
# addresses the disassembly references, fdlibm constants assembled from their mov/movk pairs; not typed ----

EXP2F_TAB = (0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f, 0x3fef9301d0125b51, 0x3fef72b83c7d517b,
             0x3fef54873168b9aa, 0x3fef387a6e756238, 0x3fef1e9df51fdee1, 0x3fef06fe0a31b715, 0x3feef1a7373aa9cb,
             0x3feedea64c123422, 0x3feece086061892d, 0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429,
             0x3feea47eb03a5585, 0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74, 0x3feea11473eb0187, 0x3feea589994cce13,
             0x3feeace5422aa0db, 0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d, 0x3feee89f995ad3ad,
             0x3feeff76f2fb5e47, 0x3fef199bdd85529c, 0x3fef3720dcef9069, 0x3fef5818dcfba487, 0x3fef7c97337b9b5f,
             0x3fefa4afa2a490da, 0x3fefd0765b6e4540)
EXPF_INVLN2 = _H("0x1.71547652b82fep+5")
EXPF_C = (_H("0x1.c6af84b912394p-20"), _H("0x1.ebfce50fac4f3p-13"), _H("0x1.62e42ff0c52d6p-6"))
LOGF_TAB = ((_H("0x1.661ec79f8f3bep+0"), _H("-0x1.57bf7808caadep-2")),
            (_H("0x1.571ed4aaf883dp+0"), _H("-0x1.2bef0a7c06ddbp-2")),
            (_H("0x1.49539f0f010b0p+0"), _H("-0x1.01eae7f513a67p-2")),
            (_H("0x1.3c995b0b80385p+0"), _H("-0x1.b31d8a68224e9p-3")),
            (_H("0x1.30d190c8864a5p+0"), _H("-0x1.6574f0ac07758p-3")),
            (_H("0x1.25e227b0b8ea0p+0"), _H("-0x1.1aa2bc79c8100p-3")),
            (_H("0x1.1bb4a4a1a343fp+0"), _H("-0x1.a4e76ce8c0e5ep-4")),
            (_H("0x1.12358f08ae5bap+0"), _H("-0x1.1973c5a611cccp-4")),
            (_H("0x1.0953f419900a7p+0"), _H("-0x1.252f438e10c1ep-5")), (_H("0x1.0000000000000p+0"), _H("0x0.0p+0")),
            (_H("0x1.e608cfd9a47acp-1"), _H("0x1.aa5aa5df25984p-5")),
            (_H("0x1.ca4b31f026aa0p-1"), _H("0x1.c5e53aa362eb4p-4")),
            (_H("0x1.b2036576afce6p-1"), _H("0x1.526e57720db08p-3")),
            (_H("0x1.9c2d163a1aa2dp-1"), _H("0x1.bc2860d224770p-3")),
            (_H("0x1.886e6037841edp-1"), _H("0x1.1058bc8a07ee1p-2")),
            (_H("0x1.767dcf5534862p-1"), _H("0x1.4043057b6ee09p-2")))
LOGF_LN2 = _H("0x1.62e42fefa39efp-1")
LOGF_A = (_H("-0x1.00ea348b88334p-2"), _H("0x1.5575b0be00b6ap-2"), _H("-0x1.ffffef20a4123p-2"))
SINCOSF_SIGN = (_H("0x1.0000000000000p+0"), _H("-0x1.0000000000000p+0"), _H("-0x1.0000000000000p+0"),
                _H("0x1.0000000000000p+0"))
HPI_INV, HPI = _H("0x1.45f306dc9c883p-1"), _H("0x1.921fb54442d18p+0")
COSF_C = (_H("0x1.0000000000000p+0"), _H("-0x1.ffffffd0c621cp-2"), _H("0x1.55553e1068f19p-5"),
          _H("-0x1.6c087e89a359dp-10"), _H("0x1.99343027bf8c3p-16"))
SINF_S = (_H("-0x1.555545995a603p-3"), _H("0x1.1107605230bc4p-7"), _H("-0x1.994eb3774cf24p-13"))
COSF_C_TABLE1 = (_H("-0x1.0000000000000p+0"), _H("0x1.ffffffd0c621cp-2"), _H("-0x1.55553e1068f19p-5"),
                 _H("0x1.6c087e89a359dp-10"), _H("-0x1.99343027bf8c3p-16"))
INV_PIO4 = (0xa2, 0xa2f9, 0xa2f983, 0xa2f9836e, 0xf9836e4e, 0x836e4e44, 0x6e4e4415, 0x4e441529, 0x441529fc,
            0x1529fc27, 0x29fc2757, 0xfc2757d1, 0x2757d1f5, 0x57d1f534, 0xd1f534dd, 0xf534ddc0, 0x34ddc0db,
            0xddc0db62, 0xc0db6295, 0xdb629599, 0x6295993c, 0x95993c43, 0x993c4390, 0x3c439041)
PI63 = _H("0x1.921fb54442d18p-62")
ATANF_HI = (_H("0x1.dac6700000000p-2"), _H("0x1.921fb40000000p-1"), _H("0x1.f730bc0000000p-1"),
            _H("0x1.921fb40000000p+0"))
ATANF_LO = (_H("0x1.586ed20000000p-28"), _H("0x1.4442d00000000p-25"), _H("0x1.281f680000000p-25"),
            _H("0x1.4442d00000000p-24"))
EXPF_OFLOW = _fh(0x42b17217)
EXPF_UFLOW = _fh(0xc2cff1b4)
EXPF_MAY_UFLOW = _fh(0xc2ce8ecf)
ACOSF_PS0 = _fh(0x3e2aaaab)
ACOSF_PS1 = _fh(0xbea6b090)
ACOSF_PS2 = _fh(0x3e4e0aa8)
ACOSF_PS3 = _fh(0xbd241146)
ACOSF_PS4 = _fh(0x3a4f7f04)
ACOSF_PS5 = _fh(0x3811ef08)
ACOSF_QS1 = _fh(0xc019d139)
ACOSF_QS2 = _fh(0x4001572d)
ACOSF_QS3 = _fh(0xbf303361)
ACOSF_QS4 = _fh(0x3d9dc62e)
ACOSF_PI = _fh(0x40490fda)
ACOSF_PIO2_HI = _fh(0x3fc90fda)
ACOSF_PIO2_LO = _fh(0x33a22168)
ACOSF_NEG_PIO2_LO = _fh(0xb3a22168)
ACOSF_TWO_PIO2_LO = _fh(0x34222168)
ATANF_T0 = _fh(0x3eaaaaab)
ATANF_T1 = _fh(0xbe4ccccd)
ATANF_T2 = _fh(0x3e124925)
ATANF_T3 = _fh(0xbde38e38)
ATANF_T4 = _fh(0x3dba2e6e)
ATANF_T5 = _fh(0xbd9d8795)
ATANF_T6 = _fh(0x3d886b35)
ATANF_T7 = _fh(0xbd6ef16b)
ATANF_T8 = _fh(0x3d4bda59)
ATANF_T9 = _fh(0xbd15a221)
ATANF_T10 = _fh(0x3c8569d7)
ATANF_HUGE = _fh(0x7149f2ca)
ATANF_HI3 = _fh(0x3fc90fda)
ATANF_LO3 = _fh(0x33a22168)
ATAN2F_PI = _fh(0x40490fdb)
ATAN2F_NEG_PI = _fh(0xc0490fdb)
ATAN2F_TINY = _fh(0x0da24260)
ATAN2F_NEG_TINY = _fh(0x8da24260)
ATAN2F_PIO2 = _fh(0x3fc90fdb)
ATAN2F_NEG_PIO2 = _fh(0xbfc90fdb)
ATAN2F_PIO4 = _fh(0x3f490fdb)
ATAN2F_NEG_PIO4 = _fh(0xbf490fdb)
ATAN2F_NEG_PI_LO = _fh(0x33bbbd2e)
ATAN2F_NEG_HALF_PI_LO = _fh(0x333bbd2e)
POW_OFF = 0x3fe6955500000000                 # pow's log_inline offset (0x338e8: -OFF)
POW_LN2HI, POW_LN2LO = _H("0x1.62e42fefa3800p-1"), _H("0x1.ef35793c76730p-45")
POW_A = (_H("-0x1.0000000000000p-1"), _H("-0x1.5555555555560p-1"), _H("0x1.0000000000006p-1"),
         _H("0x1.999999959554ep-1"), _H("-0x1.555555529a47ap-1"), _H("-0x1.2495b9b4845e9p+0"),
         _H("0x1.0002b8b263fc3p+0"))
# __pow_log_data.tab: (1/c, log c, its tail) for each of the 128 cells (the pad word left out)
POW_TAB = ((_H("0x1.6a00000000000p+0"), _H("-0x1.62c82f2b9c800p-2"), _H("0x1.ab42428375680p-48")),
           (_H("0x1.6800000000000p+0"), _H("-0x1.5d1bdbf580800p-2"), _H("-0x1.ca508d8e0f720p-46")),
           (_H("0x1.6600000000000p+0"), _H("-0x1.5767717455800p-2"), _H("-0x1.362a4d5b6506dp-45")),
           (_H("0x1.6400000000000p+0"), _H("-0x1.51aad872df800p-2"), _H("-0x1.684e49eb067d5p-49")),
           (_H("0x1.6200000000000p+0"), _H("-0x1.4be5f95777800p-2"), _H("-0x1.41b6993293ee0p-47")),
           (_H("0x1.6000000000000p+0"), _H("-0x1.4618bc21c6000p-2"), _H("0x1.3d82f484c84ccp-46")),
           (_H("0x1.5e00000000000p+0"), _H("-0x1.404308686a800p-2"), _H("0x1.c42f3ed820b3ap-50")),
           (_H("0x1.5c00000000000p+0"), _H("-0x1.3a64c55694800p-2"), _H("0x1.0b1c686519460p-45")),
           (_H("0x1.5a00000000000p+0"), _H("-0x1.347dd9a988000p-2"), _H("0x1.5594dd4c58092p-45")),
           (_H("0x1.5800000000000p+0"), _H("-0x1.2e8e2bae12000p-2"), _H("0x1.67b1e99b72bd8p-45")),
           (_H("0x1.5600000000000p+0"), _H("-0x1.2895a13de8800p-2"), _H("0x1.5ca14b6cfb03fp-46")),
           (_H("0x1.5600000000000p+0"), _H("-0x1.2895a13de8800p-2"), _H("0x1.5ca14b6cfb03fp-46")),
           (_H("0x1.5400000000000p+0"), _H("-0x1.22941fbcf7800p-2"), _H("-0x1.65a242853da76p-46")),
           (_H("0x1.5200000000000p+0"), _H("-0x1.1c898c1699800p-2"), _H("-0x1.fafbc68e75404p-46")),
           (_H("0x1.5000000000000p+0"), _H("-0x1.1675cababa800p-2"), _H("0x1.f1fc63382a8f0p-46")),
           (_H("0x1.4e00000000000p+0"), _H("-0x1.1058bf9ae4800p-2"), _H("-0x1.6a8c4fd055a66p-45")),
           (_H("0x1.4c00000000000p+0"), _H("-0x1.0a324e2739000p-2"), _H("-0x1.c6bee7ef4030ep-47")),
           (_H("0x1.4a00000000000p+0"), _H("-0x1.0402594b4d000p-2"), _H("-0x1.036b89ef42d7fp-48")),
           (_H("0x1.4a00000000000p+0"), _H("-0x1.0402594b4d000p-2"), _H("-0x1.036b89ef42d7fp-48")),
           (_H("0x1.4800000000000p+0"), _H("-0x1.fb9186d5e4000p-3"), _H("0x1.d572aab993c87p-47")),
           (_H("0x1.4600000000000p+0"), _H("-0x1.ef0adcbdc6000p-3"), _H("0x1.b26b79c86af24p-45")),
           (_H("0x1.4400000000000p+0"), _H("-0x1.e27076e2af000p-3"), _H("-0x1.72f4f543fff10p-46")),
           (_H("0x1.4200000000000p+0"), _H("-0x1.d5c216b4fc000p-3"), _H("0x1.1ba91bbca681bp-45")),
           (_H("0x1.4000000000000p+0"), _H("-0x1.c8ff7c79aa000p-3"), _H("0x1.7794f689f8434p-45")),
           (_H("0x1.4000000000000p+0"), _H("-0x1.c8ff7c79aa000p-3"), _H("0x1.7794f689f8434p-45")),
           (_H("0x1.3e00000000000p+0"), _H("-0x1.bc286742d9000p-3"), _H("0x1.94eb0318bb78fp-46")),
           (_H("0x1.3c00000000000p+0"), _H("-0x1.af3c94e80c000p-3"), _H("0x1.a4e633fcd9066p-52")),
           (_H("0x1.3a00000000000p+0"), _H("-0x1.a23bc1fe2b000p-3"), _H("-0x1.58c64dc46c1eap-45")),
           (_H("0x1.3a00000000000p+0"), _H("-0x1.a23bc1fe2b000p-3"), _H("-0x1.58c64dc46c1eap-45")),
           (_H("0x1.3800000000000p+0"), _H("-0x1.9525a9cf45000p-3"), _H("-0x1.ad1d904c1d4e3p-45")),
           (_H("0x1.3600000000000p+0"), _H("-0x1.87fa06520d000p-3"), _H("0x1.bbdbf7fdbfa09p-45")),
           (_H("0x1.3400000000000p+0"), _H("-0x1.7ab890210e000p-3"), _H("0x1.bdb9072534a58p-45")),
           (_H("0x1.3400000000000p+0"), _H("-0x1.7ab890210e000p-3"), _H("0x1.bdb9072534a58p-45")),
           (_H("0x1.3200000000000p+0"), _H("-0x1.6d60fe719d000p-3"), _H("-0x1.0e46aa3b2e266p-46")),
           (_H("0x1.3000000000000p+0"), _H("-0x1.5ff3070a79000p-3"), _H("-0x1.e9e439f105039p-46")),
           (_H("0x1.3000000000000p+0"), _H("-0x1.5ff3070a79000p-3"), _H("-0x1.e9e439f105039p-46")),
           (_H("0x1.2e00000000000p+0"), _H("-0x1.526e5e3a1b000p-3"), _H("-0x1.0de8b90075b8fp-45")),
           (_H("0x1.2c00000000000p+0"), _H("-0x1.44d2b6ccb8000p-3"), _H("0x1.70cc16135783cp-46")),
           (_H("0x1.2c00000000000p+0"), _H("-0x1.44d2b6ccb8000p-3"), _H("0x1.70cc16135783cp-46")),
           (_H("0x1.2a00000000000p+0"), _H("-0x1.371fc201e9000p-3"), _H("0x1.178864d27543ap-48")),
           (_H("0x1.2800000000000p+0"), _H("-0x1.29552f81ff000p-3"), _H("-0x1.48d301771c408p-45")),
           (_H("0x1.2600000000000p+0"), _H("-0x1.1b72ad52f6000p-3"), _H("-0x1.e80a41811a396p-45")),
           (_H("0x1.2600000000000p+0"), _H("-0x1.1b72ad52f6000p-3"), _H("-0x1.e80a41811a396p-45")),
           (_H("0x1.2400000000000p+0"), _H("-0x1.0d77e7cd09000p-3"), _H("0x1.a699688e85bf4p-47")),
           (_H("0x1.2400000000000p+0"), _H("-0x1.0d77e7cd09000p-3"), _H("0x1.a699688e85bf4p-47")),
           (_H("0x1.2200000000000p+0"), _H("-0x1.fec9131dbe000p-4"), _H("-0x1.575545ca333f2p-45")),
           (_H("0x1.2000000000000p+0"), _H("-0x1.e27076e2b0000p-4"), _H("0x1.a342c2af0003cp-45")),
           (_H("0x1.2000000000000p+0"), _H("-0x1.e27076e2b0000p-4"), _H("0x1.a342c2af0003cp-45")),
           (_H("0x1.1e00000000000p+0"), _H("-0x1.c5e548f5bc000p-4"), _H("-0x1.d0c57585fbe06p-46")),
           (_H("0x1.1c00000000000p+0"), _H("-0x1.a926d3a4ae000p-4"), _H("0x1.53935e85baac8p-45")),
           (_H("0x1.1c00000000000p+0"), _H("-0x1.a926d3a4ae000p-4"), _H("0x1.53935e85baac8p-45")),
           (_H("0x1.1a00000000000p+0"), _H("-0x1.8c345d631a000p-4"), _H("0x1.37c294d2f5668p-46")),
           (_H("0x1.1a00000000000p+0"), _H("-0x1.8c345d631a000p-4"), _H("0x1.37c294d2f5668p-46")),
           (_H("0x1.1800000000000p+0"), _H("-0x1.6f0d28ae56000p-4"), _H("-0x1.69737c93373dap-45")),
           (_H("0x1.1600000000000p+0"), _H("-0x1.51b073f062000p-4"), _H("0x1.f025b61c65e57p-46")),
           (_H("0x1.1600000000000p+0"), _H("-0x1.51b073f062000p-4"), _H("0x1.f025b61c65e57p-46")),
           (_H("0x1.1400000000000p+0"), _H("-0x1.341d7961be000p-4"), _H("0x1.c5edaccf913dfp-45")),
           (_H("0x1.1400000000000p+0"), _H("-0x1.341d7961be000p-4"), _H("0x1.c5edaccf913dfp-45")),
           (_H("0x1.1200000000000p+0"), _H("-0x1.16536eea38000p-4"), _H("0x1.47c5e768fa309p-46")),
           (_H("0x1.1000000000000p+0"), _H("-0x1.f0a30c0118000p-5"), _H("0x1.d599e83368e91p-45")),
           (_H("0x1.1000000000000p+0"), _H("-0x1.f0a30c0118000p-5"), _H("0x1.d599e83368e91p-45")),
           (_H("0x1.0e00000000000p+0"), _H("-0x1.b42dd71198000p-5"), _H("0x1.c827ae5d6704cp-46")),
           (_H("0x1.0e00000000000p+0"), _H("-0x1.b42dd71198000p-5"), _H("0x1.c827ae5d6704cp-46")),
           (_H("0x1.0c00000000000p+0"), _H("-0x1.77458f632c000p-5"), _H("-0x1.cfc4634f2a1eep-45")),
           (_H("0x1.0c00000000000p+0"), _H("-0x1.77458f632c000p-5"), _H("-0x1.cfc4634f2a1eep-45")),
           (_H("0x1.0a00000000000p+0"), _H("-0x1.39e87b9fec000p-5"), _H("0x1.502b7f526feaap-48")),
           (_H("0x1.0a00000000000p+0"), _H("-0x1.39e87b9fec000p-5"), _H("0x1.502b7f526feaap-48")),
           (_H("0x1.0800000000000p+0"), _H("-0x1.f829b0e780000p-6"), _H("-0x1.980267c7e09e4p-45")),
           (_H("0x1.0800000000000p+0"), _H("-0x1.f829b0e780000p-6"), _H("-0x1.980267c7e09e4p-45")),
           (_H("0x1.0600000000000p+0"), _H("-0x1.7b91b07d58000p-6"), _H("-0x1.88d5493faa639p-45")),
           (_H("0x1.0400000000000p+0"), _H("-0x1.fc0a8b0fc0000p-7"), _H("-0x1.f1e7cf6d3a69cp-50")),
           (_H("0x1.0400000000000p+0"), _H("-0x1.fc0a8b0fc0000p-7"), _H("-0x1.f1e7cf6d3a69cp-50")),
           (_H("0x1.0200000000000p+0"), _H("-0x1.fe02a6b100000p-8"), _H("-0x1.9e23f0dda40e4p-46")),
           (_H("0x1.0200000000000p+0"), _H("-0x1.fe02a6b100000p-8"), _H("-0x1.9e23f0dda40e4p-46")),
           (_H("0x1.0000000000000p+0"), _H("0x0.0p+0"), _H("0x0.0p+0")),
           (_H("0x1.0000000000000p+0"), _H("0x0.0p+0"), _H("0x0.0p+0")),
           (_H("0x1.fc00000000000p-1"), _H("0x1.0101575890000p-7"), _H("-0x1.0c76b999d2be8p-46")),
           (_H("0x1.f800000000000p-1"), _H("0x1.0205658938000p-6"), _H("-0x1.3dc5b06e2f7d2p-45")),
           (_H("0x1.f400000000000p-1"), _H("0x1.8492528c90000p-6"), _H("-0x1.aa0ba325a0c34p-45")),
           (_H("0x1.f000000000000p-1"), _H("0x1.0415d89e74000p-5"), _H("0x1.111c05cf1d753p-47")),
           (_H("0x1.ec00000000000p-1"), _H("0x1.466aed42e0000p-5"), _H("-0x1.c167375bdfd28p-45")),
           (_H("0x1.e800000000000p-1"), _H("0x1.894aa149fc000p-5"), _H("-0x1.97995d05a267dp-46")),
           (_H("0x1.e400000000000p-1"), _H("0x1.ccb73cdddc000p-5"), _H("-0x1.a68f247d82807p-46")),
           (_H("0x1.e200000000000p-1"), _H("0x1.eea31c006c000p-5"), _H("-0x1.e113e4fc93b7bp-47")),
           (_H("0x1.de00000000000p-1"), _H("0x1.1973bd1466000p-4"), _H("-0x1.5325d560d9e9bp-45")),
           (_H("0x1.da00000000000p-1"), _H("0x1.3bdf5a7d1e000p-4"), _H("0x1.cc85ea5db4ed7p-45")),
           (_H("0x1.d600000000000p-1"), _H("0x1.5e95a4d97a000p-4"), _H("-0x1.c69063c5d1d1ep-45")),
           (_H("0x1.d400000000000p-1"), _H("0x1.700d30aeac000p-4"), _H("0x1.c1e8da99ded32p-49")),
           (_H("0x1.d000000000000p-1"), _H("0x1.9335e5d594000p-4"), _H("0x1.3115c3abd47dap-45")),
           (_H("0x1.cc00000000000p-1"), _H("0x1.b6ac88dad6000p-4"), _H("-0x1.390802bf768e5p-46")),
           (_H("0x1.ca00000000000p-1"), _H("0x1.c885801bc4000p-4"), _H("0x1.646d1c65aacd3p-45")),
           (_H("0x1.c600000000000p-1"), _H("0x1.ec739830a2000p-4"), _H("-0x1.dc068afe645e0p-45")),
           (_H("0x1.c400000000000p-1"), _H("0x1.fe89139dbe000p-4"), _H("-0x1.534d64fa10afdp-45")),
           (_H("0x1.c000000000000p-1"), _H("0x1.1178e8227e000p-3"), _H("0x1.1ef78ce2d07f2p-45")),
           (_H("0x1.be00000000000p-1"), _H("0x1.1aa2b7e23f000p-3"), _H("0x1.ca78e44389934p-45")),
           (_H("0x1.ba00000000000p-1"), _H("0x1.2d1610c868000p-3"), _H("0x1.39d6ccb81b4a1p-47")),
           (_H("0x1.b800000000000p-1"), _H("0x1.365fcb0159000p-3"), _H("0x1.62fa8234b7289p-51")),
           (_H("0x1.b400000000000p-1"), _H("0x1.4913d8333b000p-3"), _H("0x1.5837954fdb678p-45")),
           (_H("0x1.b200000000000p-1"), _H("0x1.527e5e4a1b000p-3"), _H("0x1.633e8e5697dc7p-45")),
           (_H("0x1.ae00000000000p-1"), _H("0x1.6574ebe8c1000p-3"), _H("0x1.9cf8b2c3c2e78p-46")),
           (_H("0x1.ac00000000000p-1"), _H("0x1.6f0128b757000p-3"), _H("-0x1.5118de59c21e1p-45")),
           (_H("0x1.aa00000000000p-1"), _H("0x1.7898d85445000p-3"), _H("-0x1.c661070914305p-46")),
           (_H("0x1.a600000000000p-1"), _H("0x1.8beafeb390000p-3"), _H("-0x1.73d54aae92cd1p-47")),
           (_H("0x1.a400000000000p-1"), _H("0x1.95a5adcf70000p-3"), _H("0x1.7f22858a0ff6fp-47")),
           (_H("0x1.a000000000000p-1"), _H("0x1.a93ed3c8ae000p-3"), _H("-0x1.8724350562169p-45")),
           (_H("0x1.9e00000000000p-1"), _H("0x1.b31d8575bd000p-3"), _H("-0x1.c358d4eace1aap-47")),
           (_H("0x1.9c00000000000p-1"), _H("0x1.bd087383be000p-3"), _H("-0x1.d4bc4595412b6p-45")),
           (_H("0x1.9a00000000000p-1"), _H("0x1.c6ffbc6f01000p-3"), _H("-0x1.1ec72c5962bd2p-48")),
           (_H("0x1.9600000000000p-1"), _H("0x1.db13db0d49000p-3"), _H("-0x1.aff2af715b035p-45")),
           (_H("0x1.9400000000000p-1"), _H("0x1.e530effe71000p-3"), _H("0x1.212276041f430p-51")),
           (_H("0x1.9200000000000p-1"), _H("0x1.ef5ade4dd0000p-3"), _H("-0x1.a211565bb8e11p-51")),
           (_H("0x1.9000000000000p-1"), _H("0x1.f991c6cb3b000p-3"), _H("0x1.bcbecca0cdf30p-46")),
           (_H("0x1.8c00000000000p-1"), _H("0x1.07138604d5800p-2"), _H("0x1.89cdb16ed4e91p-48")),
           (_H("0x1.8a00000000000p-1"), _H("0x1.0c42d67616000p-2"), _H("0x1.7188b163ceae9p-45")),
           (_H("0x1.8800000000000p-1"), _H("0x1.1178e8227e800p-2"), _H("-0x1.c210e63a5f01cp-45")),
           (_H("0x1.8600000000000p-1"), _H("0x1.16b5ccbacf800p-2"), _H("0x1.b9acdf7a51681p-45")),
           (_H("0x1.8400000000000p-1"), _H("0x1.1bf99635a6800p-2"), _H("0x1.ca6ed5147bdb7p-45")),
           (_H("0x1.8200000000000p-1"), _H("0x1.214456d0eb800p-2"), _H("0x1.a87deba46baeap-47")),
           (_H("0x1.7e00000000000p-1"), _H("0x1.2bef07cdc9000p-2"), _H("0x1.a9cfa4a5004f4p-45")),
           (_H("0x1.7c00000000000p-1"), _H("0x1.314f1e1d36000p-2"), _H("-0x1.8e27ad3213cb8p-45")),
           (_H("0x1.7a00000000000p-1"), _H("0x1.36b6776be1000p-2"), _H("0x1.16ecdb0f177c8p-46")),
           (_H("0x1.7800000000000p-1"), _H("0x1.3c25277333000p-2"), _H("0x1.83b54b606bd5cp-46")),
           (_H("0x1.7600000000000p-1"), _H("0x1.419b423d5e800p-2"), _H("0x1.8e436ec90e09dp-47")),
           (_H("0x1.7400000000000p-1"), _H("0x1.4718dc271c800p-2"), _H("-0x1.f27ce0967d675p-45")),
           (_H("0x1.7200000000000p-1"), _H("0x1.4c9e09e173000p-2"), _H("-0x1.e20891b0ad8a4p-45")),
           (_H("0x1.7000000000000p-1"), _H("0x1.522ae0738a000p-2"), _H("0x1.ebe708164c759p-45")),
           (_H("0x1.6e00000000000p-1"), _H("0x1.57bf753c8d000p-2"), _H("0x1.fadedee5d40efp-46")),
           (_H("0x1.6c00000000000p-1"), _H("0x1.5d5bddf596000p-2"), _H("-0x1.a0b2a08a465dcp-47")))

MAY_UFLOWF = _fh(0x00000001)                            # __math_may_uflowf: 0x1.4p-75f squared, rounded: 2^-149
SIGN_BIAS = 0x800 << 7                                  # pow's sign of a negative base to an odd power


# ================================================================================================ float arithmetic

def f32(x: torch.Tensor) -> torch.Tensor:
    """A double rounded to the nearest float, ties to even (a C float operation's result, or a conversion to float);
    beyond the float range an infinity, as the hardware gives."""
    return x.to(F32).to(F64)


def fbits(x: torch.Tensor) -> torch.Tensor:
    """A float's bit pattern (asuint), as int64 in [0, 2^32)."""
    return x.to(F32).view(I32).to(I64) & 0xFFFFFFFF


def from_fbits(u: torch.Tensor) -> torch.Tensor:
    """The float with bit pattern u (asfloat), held in float64."""
    return (((u & 0xFFFFFFFF) ^ 0x80000000) - 0x80000000).to(I32).view(F32).to(F64)


def fmaf(a: torch.Tensor, b, c) -> torch.Tensor:
    """fmaf(a, b, c) of floats, rounded once: the exact product (24 + 24 bits fit a double) plus c rounded to odd in
    double, then rounded to float (Boldo and Melquiond: 53 >= 24 + 2). Finite operands."""
    return f32(add_odd(a * b, c if torch.is_tensor(c) else torch.full_like(a, c)))


def frinta(z: torch.Tensor) -> torch.Tensor:
    """aarch64's frinta: z rounded to the nearest integer, ties away from zero (z - trunc(z) is exact)."""
    t = torch.trunc(z)
    d = z - t
    return torch.where(d >= 0.5, t + 1.0, torch.where(d <= -0.5, t - 1.0, t))


_TABS: dict = {}


def _tab(name: str, device) -> torch.Tensor:
    """A table as a tensor on a device, made once."""
    key = (name, str(device))
    if key not in _TABS:
        if name == "exp2f":
            t = torch.tensor(EXP2F_TAB, dtype=I64, device=device)
        elif name == "logf":
            t = torch.tensor(LOGF_TAB, dtype=F64, device=device)
        elif name == "inv_pio4":
            t = torch.tensor(INV_PIO4, dtype=I64, device=device)
        elif name == "sign":
            t = torch.tensor(SINCOSF_SIGN, dtype=F64, device=device)
        elif name == "atanf":
            t = torch.tensor([ATANF_HI, ATANF_LO], dtype=F64, device=device)
        elif name == "pow":
            t = torch.tensor(POW_TAB, dtype=F64, device=device)
        else:
            raise KeyError(name)
        _TABS[key] = t
    return _TABS[key]


_EXPF_SPLIT = tuple(split(c) for c in EXPF_C)
_LOGF_SPLIT = split(LOGF_LN2), split(LOGF_A[0]), split(LOGF_A[1])
_SIN_SPLIT = tuple(split(c) for c in SINF_S)
_COS_SPLIT = tuple(split(c) for c in COSF_C)
_NEG_HPI_SPLIT = split(-HPI)
_POW_SPLIT = (split(POW_LN2HI), split(POW_LN2LO)) + tuple(split(a) for a in POW_A)
_EXP_SPLIT = split(NEGLN2HIN), split(NEGLN2LON), split(C3), split(C5)


# ============================================================================================ expf, logf (ARM's)

def expf(x: torch.Tensor) -> torch.Tensor:
    """glibc's expf (e_expf.c, __expf at 0x3dd80): k = round(x 32/ln2) ties away (frinta), r = x 32/ln2 - k, a cubic
    in r fused three times around the table's 2^(i/32), rounded to float; above 88.72 an infinity, below -103.97 zero,
    between that and -103.28 the smallest subnormal (__math_may_uflowf)."""
    ix = fbits(x)
    abstop = (ix >> 20) & 0x7FF
    big = abstop > 0x42A                                # |x| >= 88 or NaN (0x3dd8c)
    through = (x <= EXPF_OFLOW) & (x >= EXPF_MAY_UFLOW)  # big but none of the special results: the main path
    special = big & ~through
    xs = torch.where(special, 0.0, x)
    z = xs * EXPF_INVLN2                                # 0x3dda8
    kd = frinta(z)
    ki = kd.to(I64)                                     # fcvtas: the same integer
    r = z - kd
    s = (_tab("exp2f", x.device)[ki & 31] + ki * (1 << 47)).view(F64)   # t += ki << 47
    zc = fma(r, EXPF_C[0], EXPF_C[1], _EXPF_SPLIT[0])  # 0x3ddc0 fmadd: C0 r + C1
    y = fma(r, EXPF_C[2], 1.0, _EXPF_SPLIT[2])          # 0x3ddc4 fmadd: C2 r + 1
    r2 = r * r
    y = fma(zc, r2, y)                                  # 0x3ddd4 fmadd: z r2 + y
    out = f32(y * s)
    sp = torch.where(x < EXPF_UFLOW, 0.0, MAY_UFLOWF)   # __math_uflowf (0x1p-95f squared: 0), __math_may_uflowf
    sp = torch.where(x > EXPF_OFLOW, INF, sp)           # __math_oflowf
    sp = torch.where(abstop >= 0x7F8, x + x, sp)        # infinity or NaN
    sp = torch.where(x == -INF, 0.0, sp)
    return torch.where(special, sp, out)


def logf(x: torch.Tensor) -> torch.Tensor:
    """glibc's logf (e_logf.c, __logf at 0x41020): x = 2^k z with z in [0.7, 1.4), r = z/c - 1 fused, a cubic in r,
    log c + k ln2 fused; subnormals normalised by 2^23; log 1 = +0, log 0 = -inf, of a negative or NaN NaN."""
    ix = fbits(x)
    special = ((ix - 0x00800000) & 0xFFFFFFFF) >= 0x7F000000   # below 2^-126, negative, infinity or NaN (0x41040)
    sub = special & (ix > 0) & (ix < 0x00800000)               # a positive subnormal: x 2^23, exponent less 23
    xn = f32(torch.where(sub, x, 0.0) * 8388608.0)
    ixn = torch.where(sub, (fbits(xn) - (23 << 23)) & 0xFFFFFFFF, ix)
    tmp = (ixn - 0x3F330000) & 0xFFFFFFFF                      # ix - OFF (0x41044)
    i = (tmp >> 19) & 15
    k = ((tmp ^ 0x80000000) - 0x80000000) >> 23                # (int32_t) tmp >> 23
    z = from_fbits((ixn - (tmp & 0xFF800000)) & 0xFFFFFFFF)
    t = _tab("logf", x.device)[i]
    invc, logc = t[..., 0], t[..., 1]
    r = fma(z, invc, -1.0)                              # 0x41088 fmadd: invc z - 1
    y0 = fma(k.to(F64), LOGF_LN2, logc, _LOGF_SPLIT[0])  # 0x41090 fmadd: logc + k ln2
    y = fma(r, LOGF_A[1], LOGF_A[2], _LOGF_SPLIT[2])    # 0x41094 fmadd: A1 r + A2
    r2 = r * r
    ry0 = r + y0                                        # 0x4109c
    y = fma(r2, LOGF_A[0], y, _LOGF_SPLIT[1])           # 0x410a0 fmadd: A0 r2 + y
    out = f32(fma(y, r2, ry0))                          # 0x410a4 fmadd: y r2 + (r + y0)
    out = torch.where(ix == 0x3F800000, 0.0, out)
    sp = torch.where((ix & 0x7FFFFFFF) == 0, -INF, torch.where(ix == 0x7F800000, INF, NAN))
    return torch.where(special & ~sub, sp, out)


# ================================================================================================ sinf, cosf (ARM's)

def _sin_poly(xv: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """sinf_poly's sine branch: x3 = x x2, s1 = s2 + x2 s3, x7 = x2 x3, (x + x3 s1) + s1 x7, three fused."""
    x3 = xv * x2
    s1 = fma(x2, SINF_S[2], SINF_S[1], _SIN_SPLIT[2])
    x7 = x2 * x3
    s = fma(x3, SINF_S[0], xv, _SIN_SPLIT[0])
    return fma(s1, x7, s)


def _cos_poly(x2: torch.Tensor) -> torch.Tensor:
    """sinf_poly's cosine branch: c1 = c0 + x2 c1, c2 = c3 + x2 c4, x4 = x2 x2, x6 = x2 x4, (c1 + x4 c2) + c2 x6."""
    c1 = fma(x2, COSF_C[1], COSF_C[0], _COS_SPLIT[1])
    c2 = fma(x2, COSF_C[4], COSF_C[3], _COS_SPLIT[4])
    x4 = x2 * x2
    x6 = x2 * x4
    c = fma(x4, COSF_C[2], c1, _COS_SPLIT[2])
    return fma(c2, x6, c)


def _reduce_large(ix: torch.Tensor) -> tuple:
    """reduce_large (sincosf.h): x 4/pi in 62-bit fixed point from __inv_pio4, with 32-bit halves for the 64-bit
    steps: (n, the reduced x in double)."""
    idx = (ix >> 26) & 15
    m = ((ix & 0x7FFFFF) | 0x800000) << ((ix >> 23) & 7)      # below 2^31
    tab = _tab("inv_pio4", ix.device)
    hi = (m * tab[idx]) & 0xFFFFFFFF                            # res0 = xi arr[0], a 32-bit product, shifted up 32
    p1, p2 = m * tab[idx + 4], m * tab[idx + 8]                 # res1, res2: below 2^63
    lo = (p2 >> 32) + (p1 & 0xFFFFFFFF)                         # res0 = (res2 >> 32) | (res0 << 32); res0 += res1
    hi = (hi + (p1 >> 32) + (lo >> 32)) & 0xFFFFFFFF
    lo = lo & 0xFFFFFFFF
    n = ((hi + 0x20000000) & 0xFFFFFFFF) >> 30                  # (res0 + 2^61) >> 62
    hi = (hi - (n << 30)) & 0xFFFFFFFF                          # res0 -= n << 62
    v = ((hi ^ 0x80000000) - 0x80000000) * 4294967296 + lo      # (int64_t) res0
    return n, v.to(F64) * PI63


def _sincosf(x: torch.Tensor, cosine: bool) -> torch.Tensor:
    """glibc's sinf (s_sinf.c, 0x43a40) or cosf (s_cosf.c, 0x42430): below pi/4 the polynomial at x (below 2^-12, x or
    1); below 120 reduce_fast (x - round(x 2/pi) pi/2, fused); else reduce_large; the quadrant's sign on x, the
    polynomial by n's parity, the second table's negated cosine as the result's sign (exact)."""
    ix = fbits(x)
    abstop = (ix >> 20) & 0x7FF
    small, tiny, mid = abstop < 0x3F4, abstop < 0x398, abstop < 0x42F
    finite = abstop < 0x7F8
    xf = torch.where(mid, x, 0.0)
    kd = frinta(xf * HPI_INV)
    xr_fast = fma(kd, -HPI, xf, _NEG_HPI_SPLIT)          # fmsub: x - round(r) hpi
    n_large, xr_large = _reduce_large(torch.where(finite & ~mid, ix, 0x43000000))
    n = torch.where(mid, kd.to(I64), n_large)
    xr = torch.where(mid, xr_fast, xr_large)
    q = torch.where(mid, n, n + (ix >> 31))             # the sign's quadrant: n, or n + the sign of x when large
    s = _tab("sign", x.device)[q & 3]
    x2 = xr * xr
    odd = ((n & 1) == 0) if cosine else ((n & 1) == 1)  # cosf uses n ^ 1
    reduced = torch.where(odd, _cos_poly(x2) * torch.where((q & 2) != 0, -1.0, 1.0), _sin_poly(xr * s, x2))
    xx = x * x
    near = _cos_poly(xx) if cosine else _sin_poly(x, xx)
    out = f32(torch.where(small, near, reduced))
    if cosine:
        out = torch.where(tiny, 1.0, out)
    else:
        out = torch.where(tiny, x, out)
    return torch.where(finite, out, NAN)                # __math_invalidf


def sinf(x: torch.Tensor) -> torch.Tensor:
    """glibc's sinf of floats held in float64."""
    return _sincosf(x, False)


def cosf(x: torch.Tensor) -> torch.Tensor:
    """glibc's cosf of floats held in float64."""
    return _sincosf(x, True)


# ======================================================================================= acosf, atanf (fdlibm's)

def _acosf_r(z: torch.Tensor) -> torch.Tensor:
    """e_acosf.c's r = p/q: p = z (pS0 + z (pS1 + ...)), q = 1 + z (qS1 + ...), every step fused as compiled."""
    p4 = fmaf(z, ACOSF_PS5, ACOSF_PS4)
    q3 = fmaf(z, ACOSF_QS4, ACOSF_QS3)
    p3 = fmaf(p4, z, ACOSF_PS3)
    q2 = fmaf(q3, z, ACOSF_QS2)
    p2 = fmaf(p3, z, ACOSF_PS2)
    q1 = fmaf(q2, z, ACOSF_QS1)
    p1 = fmaf(p2, z, ACOSF_PS1)
    q = fmaf(q1, z, 1.0)
    p0 = fmaf(p1, z, ACOSF_PS0)
    p = f32(p0 * z)
    return f32(p / q)


def acosf(x: torch.Tensor) -> torch.Tensor:
    """glibc's acosf (e_acosf.c, __ieee754_acosf at 0x3d2e0, behind the errno wrapper): |x| < 0.5, x > 0.5 and
    x < -0.5 with their fused steps, the exact values at +-1 and below 2^-26, NaN beyond 1."""
    hx = fbits(x)
    ix = hx & 0x7FFFFFFF
    xs = torch.where(ix > 0x3F800000, 0.0, x)
    # |x| < 0.5 (0x3d448)
    z = f32(xs * xs)
    t = fmaf(-xs, _acosf_r(z), ACOSF_PIO2_LO)            # fmsub: pio2_lo - x r
    small = f32(ACOSF_PIO2_HI - f32(xs - t))
    # x > 0.5 (0x3d368)
    z = f32(f32(1.0 - xs) * 0.5)
    s = f32(torch.sqrt(z))
    df = from_fbits(fbits(s) & 0xFFFFF000)
    c = f32(fmaf(-df, df, z) / f32(s + df))             # fmsub: z - df df
    w = fmaf(s, _acosf_r(z), c)                         # fmadd: c + s r
    pos = f32(f32(df + w) + f32(df + w))
    # x < -0.5 (0x3d51c)
    z = f32(f32(xs + 1.0) * 0.5)
    r = _acosf_r(z)
    s = f32(torch.sqrt(z))
    w = fmaf(s, r, ACOSF_NEG_PIO2_LO)                   # fmadd: -pio2_lo + s r
    neg = fmaf(-f32(s + w), 2.0, ACOSF_PI)              # fmsub: pi - (s + w) 2
    out = torch.where(ix < 0x3F000000, small, torch.where(hx >> 31 == 1, neg, pos))
    out = torch.where(ix <= 0x32800000, f32(torch.full_like(x, ACOSF_PIO2_HI) + ACOSF_PIO2_LO), out)
    one = torch.where(hx >> 31 == 1, f32(torch.full_like(x, ACOSF_PI) + ACOSF_TWO_PIO2_LO), 0.0)
    out = torch.where(ix == 0x3F800000, one, out)
    return torch.where(ix > 0x3F800000, NAN, out)


def atanf(x: torch.Tensor) -> torch.Tensor:
    """glibc's atanf (s_atanf.c, 0x42090): reduced by |x| against 7/16, 11/16, 19/16 and 39/16 (fused where
    compiled), the odd and even polynomials in w = z^2 fused, the table's atan(c) in two parts; x itself below 2^-29,
    +-(atanhi[3] + atanlo[3]) at 2^25 and beyond."""
    hx = fbits(x)
    ix = hx & 0x7FFFFFFF
    xs = torch.where(ix > 0x4BFFFFFF, 0.0, x)
    ax = xs.abs()
    x0 = f32(fmaf(ax, 2.0, -1.0) / f32(ax + 2.0))       # id 0 (fmadd: 2x - 1)
    x1 = f32(f32(ax - 1.0) / f32(ax + 1.0))             # id 1
    x2 = f32(f32(ax - 1.5) / fmaf(ax, 1.5, 1.0))        # id 2 (fmadd: 1 + 1.5 x)
    x3 = f32(torch.full_like(ax, -1.0) / torch.where(ax == 0.0, 1.0, ax))   # id 3
    below = ix <= 0x3EDFFFFF
    k = torch.where(ix <= 0x3F97FFFF, torch.where(ix <= 0x3F2FFFFF, 0, 1), torch.where(ix <= 0x401BFFFF, 2, 3))
    xr = torch.where(below, xs, torch.where(k == 0, x0, torch.where(k == 1, x1, torch.where(k == 2, x2, x3))))
    z = f32(xr * xr)
    w = f32(z * z)
    t8 = fmaf(w, ATANF_T10, ATANF_T8)
    t7 = fmaf(w, ATANF_T9, ATANF_T7)
    t6 = fmaf(t8, w, ATANF_T6)
    t5 = fmaf(t7, w, ATANF_T5)
    t4 = fmaf(t6, w, ATANF_T4)
    t3 = fmaf(t5, w, ATANF_T3)
    t2 = fmaf(t4, w, ATANF_T2)
    t1 = fmaf(t3, w, ATANF_T1)
    t0 = fmaf(t2, w, ATANF_T0)
    s12 = fmaf(t0, z, f32(t1 * w))                      # fmadd: s1 + s2 = z (aT0 + ...) + w (aT1 + ...)
    xs12 = f32(xr * s12)
    hl = _tab("atanf", x.device)[:, k]
    zz = f32(hl[0] - f32(f32(xs12 - hl[1]) - xr))
    out = torch.where(below, f32(xr - xs12), torch.where(hx >> 31 == 1, -zz, zz))
    out = torch.where(ix <= 0x30FFFFFF, x, out)         # |x| < 2^-29: x (huge + x > 1)
    big = torch.where(hx >> 31 == 1, f32(torch.full_like(x, -ATANF_HI3) - ATANF_LO3),
                      f32(torch.full_like(x, ATANF_HI3) + ATANF_LO3))
    out = torch.where(ix > 0x4BFFFFFF, big, out)
    return torch.where(ix > 0x7F800000, x + x, out)


def atan2f(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """glibc's atan2f (e_atan2f.c, __ieee754_atan2f at 0x3d930): atanf(y) for x = 1; the quadrant m = 2 sign(x) +
    sign(y); zeros and infinities with their exact values (+-3 pi/4 fused); |y/x| beyond 2^60 or below 2^-60;
    otherwise atanf(|y/x|) moved to the quadrant with pi_lo."""
    hy, hx = fbits(y), fbits(x)
    iy, ix = hy & 0x7FFFFFFF, hx & 0x7FFFFFFF
    m = ((hx >> 30) & 2) | (hy >> 31)
    k = (iy - ix) >> 23
    q = f32(y / torch.where(ix == 0, 1.0, x))
    z = atanf(q.abs())
    z = torch.where((hx >> 31 == 1) & (k < -60), 0.0, z)
    z = torch.where(k > 60, f32(torch.full_like(x, ATAN2F_PIO2) - ATAN2F_NEG_HALF_PI_LO), z)
    zl = f32(z + ATAN2F_NEG_PI_LO)                      # z - pi_lo
    out = torch.where(m == 0, z, torch.where(m == 1, -z, torch.where(m == 2, f32(ATAN2F_PI - zl),
                                                                      f32(zl - ATAN2F_PI))))
    full = lambda c: torch.full_like(x, c)
    pi_t, mpi_t = f32(full(ATAN2F_PI) + ATAN2F_TINY), f32(full(ATAN2F_NEG_PI) - ATAN2F_TINY)
    pio2_t = torch.where(hy >> 31 == 1, f32(full(ATAN2F_NEG_PIO2) - ATAN2F_TINY), f32(full(ATAN2F_PIO2) + ATAN2F_TINY))
    out = torch.where(iy == 0x7F800000, pio2_t, out)    # y infinite (x finite)
    inf_inf = torch.where(m == 0, f32(full(ATAN2F_PIO4) + ATAN2F_TINY),
                          torch.where(m == 1, f32(full(ATAN2F_NEG_PIO4) - ATAN2F_TINY),
                                      torch.where(m == 2, fmaf(full(3.0), ATAN2F_PIO4, ATAN2F_TINY),
                                                  fmaf(full(-3.0), ATAN2F_PIO4, ATAN2F_NEG_TINY))))
    inf_fin = torch.where(m == 0, 0.0, torch.where(m == 1, -0.0, torch.where(m == 2, pi_t, mpi_t)))
    out = torch.where(ix == 0x7F800000, torch.where(iy == 0x7F800000, inf_inf, inf_fin), out)
    out = torch.where(ix == 0, pio2_t, out)             # x zero
    out = torch.where(iy == 0, torch.where(m <= 1, y, torch.where(m == 2, pi_t, mpi_t)), out)   # y zero
    out = torch.where(hx == 0x3F800000, atanf(y), out)
    return torch.where((ix > 0x7F800000) | (iy > 0x7F800000), x + y, out)


# ================================================================================================== pow (ARM's)

def _checkint(iy: torch.Tensor) -> torch.Tensor:
    """checkint: 0 if y is not an integer, 1 if odd, 2 if even (y finite and nonzero)."""
    e = (iy >> 52) & 0x7FF
    shift = torch.clamp(0x3FF + 52 - e, 0, 52)
    low = iy & ((torch.ones_like(iy) << shift) - 1)
    unit = (iy >> shift) & 1
    out = torch.where(low != 0, 0, torch.where(unit == 1, 1, 2))
    return torch.where(e < 0x3FF, 0, torch.where(e > 0x3FF + 52, 2, out))


def _log_inline(ix: torch.Tensor) -> tuple:
    """pow's log_inline (0x338e8 to 0x339b0): log x as y + tail, 15 bits beyond double, every step as compiled."""
    tmp = ix - POW_OFF
    i = (tmp >> 45) & 127
    kd = (tmp >> 52).to(F64)
    z = (ix - (tmp & -(1 << 52))).view(F64)
    t = _tab("pow", ix.device)[i]
    invc, logc, logctail = t[..., 0], t[..., 1], t[..., 2]
    sh, sl = _POW_SPLIT[0], _POW_SPLIT[1]
    a = POW_A
    sa = _POW_SPLIT[2:]
    t1 = fma(kd, POW_LN2HI, logc, sh)                  # 0x3392c: logc + kd ln2hi
    r = fma(z, invc, -1.0)                              # 0x33934: z invc - 1
    lo1 = fma(kd, POW_LN2LO, logctail, sl)              # 0x33944: logctail + kd ln2lo
    ar = r * a[0]                                       # 0x33950
    a56 = fma(r, a[6], a[5], sa[6])                     # 0x33954: A5 + r A6
    t2 = r + t1                                         # 0x33958
    a34 = fma(r, a[4], a[3], sa[4])                     # 0x3395c: A3 + r A4
    a12 = fma(r, a[2], a[1], sa[2])                     # 0x33960: A1 + r A2
    ar2 = r * ar                                        # 0x33964
    d = t1 - t2                                         # 0x33968
    lo3 = fma(ar, r, -ar2)                              # 0x3396c fnmsub: ar r - ar2
    a3456 = fma(ar2, a56, a34)                          # 0x33970: a34 + ar2 a56
    hi = t2 + ar2                                       # 0x33974
    lo2 = d + r                                         # 0x33978
    ar3 = r * ar2                                       # 0x3397c
    e = t2 - hi                                         # 0x33980
    s = lo1 + lo2                                       # 0x33984
    poly = fma(ar2, a3456, a12)                         # 0x33988: a12 + ar2 a3456
    lo4 = e + ar2                                       # 0x3398c
    s = s + lo3                                         # 0x33990
    s = s + lo4                                         # 0x33994
    lo = fma(ar3, poly, s)                              # 0x33998: s + ar3 poly
    y = hi + lo                                         # 0x3399c
    return y, (hi - y) + lo                             # 0x339a4, 0x339b0


def _exp_inline(ehi: torch.Tensor, elo: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """pow's exp_inline (0x339b4 to 0x33a3c, its special cases at 0x33aa4): sign exp(ehi + elo), the sign through
    sign_bias (the scale's sign bit), as glibc's exp otherwise (exact_ballpark.vec_exp)."""
    abstop = (ehi.view(I64) >> 52) & 0x7FF
    tiny, edge, huge = abstop < 0x3C9, abstop == 0x408, abstop > 0x408
    x = torch.where(tiny | huge, 0.0, ehi)
    kd = frinta(x * INVLN2N)
    ki = kd.to(I64)
    r = fma(kd, NEGLN2HIN, x, _EXP_SPLIT[0])            # 0x339ec: x + kd negln2hiN
    r = fma(kd, NEGLN2LON, r, _EXP_SPLIT[1])            # 0x33a00: + kd negln2loN
    tab = _table(ehi.device)
    idx = (ki & 0x7F) * 2
    tail = tab[idx].view(F64)
    base = tab[idx + 1] + ki * (1 << 45)                # T[idx + 1] + (ki << 45), within int64
    sgn = torch.where(negative, SIGN_BIT, 0)
    r = elo + r                                         # 0x33a0c: r += xtail
    p23 = fma(r, C3, C2, _EXP_SPLIT[2])                 # 0x33a14
    p45 = fma(r, C5, C4, _EXP_SPLIT[3])                 # 0x33a18
    r2 = r * r
    t = r + tail                                        # 0x33a20
    r4 = r2 * r2
    tmp = fma(r2, p23, t)                               # 0x33a28
    tmp = fma(r4, p45, tmp)                             # 0x33a2c
    scale = (base ^ sgn).view(F64)
    main = fma(tmp, scale, scale)                       # 0x33a38: scale + tmp scale
    # specialcase (abstop forced to 0 at 0x33b50): k > 0 may overflow, k < 0 may be subnormal
    sc_hi = ((base - (1009 << 52)) ^ sgn).view(F64)
    y_hi = fma(tmp, sc_hi, sc_hi) * _H("0x1p1009")      # 0x33cb0, 0x33cb8
    sb_lo = (base + (1022 << 52)) ^ sgn
    sc_lo = sb_lo.view(F64)
    st = tmp * sc_lo                                    # 0x33acc: not fused
    y = sc_lo + st
    one = torch.where(y < 0.0, -1.0, 1.0)
    lo = (sc_lo - y) + st
    hi = y + one
    lo = ((one - hi) + y) + lo
    y2 = (lo + hi) - one
    y2 = torch.where(y2 == 0.0, (sb_lo & SIGN_BIT).view(F64), y2)
    y_lo = torch.where(y.abs() < 1.0, y2, y) * _H("0x1p-1022")
    out = torch.where(edge, torch.where((ki & 0x80000000) == 0, y_hi, y_lo), main)
    out = torch.where(tiny, torch.where(negative, -(ehi + 1.0), ehi + 1.0), out)     # 1 + x, with the sign
    flow = torch.where(ehi < 0.0, 0.0, INF)             # __math_uflow, __math_oflow: (+-2^-767)^2, (+-2^769)^2
    return torch.where(huge, torch.where(negative, -flow, flow), out)


def pow(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """glibc's pow (e_pow.c, __pow at 0x338b0) of float64 tensors: log_inline, then y log x in two parts (fused),
    then exp_inline; the special cases in glibc's order (zeros, infinities and NaNs of y, then of x; a negative base
    with an integer y; |y| below 2^-65 or at 2^63 and beyond; a subnormal base normalised by 2^52)."""
    ix, iy = x.view(I64), y.view(I64)
    topx, topy = (ix >> 52) & 0xFFF, (iy >> 52) & 0xFFF
    ay = topy & 0x7FF
    # zeroinfnan(y): 2 iy - 1 >= 2 asuint64(inf) - 1, unsigned: y is a zero, an infinity or NaN
    y_zin = (ay == 0x7FF) | ((iy & ~SIGN_BIT) == 0)
    x_zin = ((topx & 0x7FF) == 0x7FF) | ((ix & ~SIGN_BIT) == 0)
    ax_bits = ix & ~SIGN_BIT
    one_bits = 0x3FF0000000000000
    x_nan = ax_bits > 0x7FF0000000000000
    y_nan = (iy & ~SIGN_BIT) > 0x7FF0000000000000
    # y zero, infinite or NaN
    r_y = torch.where((ax_bits == one_bits), 1.0, torch.where((ax_bits < one_bits) == (iy >= 0), 0.0, y * y))
    r_y = torch.where(x_nan | y_nan, x + y, r_y)
    r_y = torch.where(ix == one_bits, 1.0, r_y)
    r_y = torch.where((iy & ~SIGN_BIT) == 0, 1.0, r_y)
    # x zero, infinite or NaN (y finite, nonzero)
    yint = _checkint(iy)
    x2 = x * x
    odd_neg = (ix < 0) & (yint == 1)
    x2 = torch.where(odd_neg, -x2, x2)
    r_x = torch.where(iy < 0, torch.ones_like(x2) / x2, x2)                 # 1/x2 for a negative y
    # x negative, finite, nonzero
    neg_ok = (ix < 0) & (yint != 0)
    negative = (ix < 0) & (yint == 1)
    axm = torch.where(ix < 0, ax_bits, ix)
    topxm = torch.where(ix < 0, topx & 0x7FF, topx)
    # |y| tiny or huge
    r_big = torch.where(axm == one_bits, 1.0,
                        torch.where(ay < 0x3BE, torch.where(axm > one_bits, 1.0 + y, 1.0 - y),
                                    torch.where((axm > one_bits) == (topy < 0x800), INF, 0.0)))
    y_far = ((ay - 0x3BE) & 0xFFFFFFFF) >= 0x80
    # the main path, a subnormal base normalised
    sub = topxm == 0
    xn = (torch.where(sub, axm, 0).view(F64) * 4503599627370496.0).view(I64) & ~SIGN_BIT
    ixm = torch.where(sub, xn - (52 << 52), axm)
    ok = ~y_zin & ~x_zin & ~((ix < 0) & ~neg_ok) & ~y_far
    ixm = torch.where(ok, ixm, one_bits)
    hi, lo = _log_inline(ixm)
    ys = torch.where(ok, y, 0.0)
    ehi = ys * hi                                       # 0x339a0
    elo = fma(ys, lo, fma(ys, hi, -ehi))                # 0x339a8 fnmsub: y hi - ehi; 0x339b8 fmadd: + y lo
    out = _exp_inline(ehi, elo, negative & ok)
    out = torch.where(y_far, r_big, out)
    out = torch.where((ix < 0) & ~neg_ok, NAN, out)     # __math_invalid
    out = torch.where(x_zin & ~y_zin, r_x, out)
    return torch.where(y_zin, r_y, out)
