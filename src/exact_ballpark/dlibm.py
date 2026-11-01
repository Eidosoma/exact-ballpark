# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (c) 2026 Eidosoma
# Transcribed from the GNU C Library 2.34; see THIRD_PARTY_NOTICES.md.
"""The C library's double sin, cos and log as scalar Python, operation for operation: what Destiny's FOLLOW, ORBIT and
WARP modes call in double (exact_ballpark.modes calls math.sin, math.cos and math.log, the C library's own), transcribed
here so that exact_ballpark.vec_modes_cuda's CUDA kernel can follow them line for line on the GPU. (exp, the fourth, is
exact_ballpark.vec_exp's; sqrt is IEEE.)

The C library is glibc 2.34 on this machine (aarch64, /usr/lib64/libm.so.6, SHA-256 `LIBM_SHA256`), and each function
is its compiled code read from the disassembly, every fused multiply-add GCC made fused here too (math.fma, rounded
once) and only those:

- sin and cos are IBM's Accurate Mathematical Library (sysdeps/ieee754/dbl-64/s_sin.c, without its slow paths since
  glibc 2.28): below 2^-26 (sin) or 2^-27 (cos) the argument or 1; below 0.855469 a Taylor polynomial (under 0.126) or
  the 440-entry table of sines and cosines (`__sincostab`) with short polynomials; below 2.426265 the same around pi/2;
  below 105414350 a three-part reduction by pi/2 (fused); beyond, `branred`; infinities and NaN give NaN. __sin is at
  0x362b0, __cos at 0x369f0.
- branred (sysdeps/ieee754/dbl-64/branred.c, at 0x49dd0) reduces huge arguments with 2/pi in 24-bit pieces
  (`toverp`); glibc compiles it without contraction (it splits numbers Dekker's way), so nothing here is fused.
- log is ARM's optimized-routines (dbl-64/e_log.c, MIT-licensed upstream; at 0x33610): near 1 a degree-11 polynomial
  with r split in two, elsewhere x = 2^k z, z/c - 1 by one fused multiply-add from a 128-cell table (`__log_data`) and a
  degree-5 polynomial.

Constants and tables were read out of the library at the addresses the disassembly references, or assembled from its
mov/movk immediates (the generated block below); a test
holds them to the library's bytes when it is that library. A NaN result is NaN, not necessarily with the C library's
payload; errno and the floating-point flags are not modelled (Python's math.sin and math.log raise where C returns a NaN
or an infinity: these return the C value). Checked by tests/test_movement_dlibm.py against the C library itself.
`sincos` is glibc's sincos from the same pieces (tests/test_movement_dlibm_sincos.py): it is not
always sin and cos (see its docstring).
"""

from __future__ import annotations

import math
import struct

_H = float.fromhex
LIBM_SHA256 = "b4fa1652bf3a7267d0fa4256fade78f0df2954b7b39ee842ec410e0e7b0f6c86"   # /usr/lib64/libm.so.6
M64 = (1 << 64) - 1
fma = math.fma

# ---- generated from this machine's libm.so.6 (SHA-256
# b4fa1652bf3a7267...): read at the addresses the disassembly of __sin, __cos, __branred and __log
# references, or assembled from their mov/movk immediates; not typed ----

TAYLOR_LIMIT = _H("0x1.020c49ba5e354p-3")
BIG = _H("0x1.8000000000000p+45")
TOINT = _H("0x1.8000000000000p+52")
S1 = _H("-0x1.5555555555555p-3")
S5 = _H("-0x1.addffc2fcdf59p-26")
S4 = _H("0x1.71de27b9a7ed9p-19")
S3 = _H("-0x1.a01a019db08b8p-13")
S2 = _H("0x1.1111111110ecep-7")
CS6 = _H("0x1.6c16bedd9e239p-10")
SN5 = _H("0x1.11110e829872fp-7")
CS4 = _H("-0x1.5555555555535p-5")
SN3 = _H("-0x1.5555555555515p-3")
HPINV = _H("0x1.45f306dc9c883p-1")
MP1 = _H("0x1.921fb58000000p+0")
MP2 = _H("-0x1.dde973c000000p-27")
PP3 = _H("-0x1.cb3b398000000p-55")
PP4 = _H("-0x1.d747f23e32ed7p-83")
HP0 = _H("0x1.921fb54442d18p+0")
HP1 = _H("0x1.1a62633145c07p-54")
NEG_HP1 = _H("-0x1.1a62633145c07p-54")
CS2 = _H("0x1.0000000000000p-1")
DBL_MIN = _H("0x1.0000000000000p-1022")
SINCOSTAB = (_H("0x0.0p+0"), _H("0x0.0p+0"), _H("0x1.0000000000000p+0"), _H("0x0.0p+0"), _H("0x1.fffeaaaaeeeefp-8"),
             _H("-0x1.e45e2ec67b77cp-62"), _H("0x1.fffc000155552p-1"), _H("0x1.f4a01a0196daep-55"),
             _H("0x1.fffaaaaeeeed5p-7"), _H("-0x1.2ab639a9f0777p-63"), _H("0x1.fff000155549fp-1"),
             _H("0x1.28a28a03a5ef3p-55"), _H("0x1.7ff7001033255p-6"), _H("0x1.efe2b51527336p-64"),
             _H("0x1.ffdc006bff7e6p-1"), _H("0x1.ae6dae86977bdp-55"), _H("0x1.ffeaaaeeee86fp-6"),
             _H("-0x1.cd406fb224ae2p-60"), _H("0x1.ffc00155527d3p-1"), _H("-0x1.3b54492d89b5bp-55"),
             _H("0x1.3feb2b12d45d5p-5"), _H("0x1.4ec54203d1c11p-60"), _H("0x1.ff9c03414a7bap-1"),
             _H("0x1.991f4be6c59bfp-57"), _H("0x1.7fdc01032fba9p-5"), _H("-0x1.599bdf46e997ap-59"),
             _H("0x1.ff7006bfdf99fp-1"), _H("-0x1.8b3b560648d5fp-56"), _H("0x1.bfc6d78586dacp-5"),
             _H("0x1.8e4fd03dbf236p-62"), _H("0x1.ff3c0c8103a31p-1"), _H("0x1.4856dbddc0e66p-56"),
             _H("0x1.ffaaaeeed4edbp-5"), _H("-0x1.2d16d32684b69p-59"), _H("0x1.ff0015549f4d3p-1"),
             _H("0x1.328387b99426fp-55"), _H("0x1.1fc343d808befp-4"), _H("-0x1.f3d32e6f3be4fp-58"),
             _H("0x1.febc222a8ef9fp-1"), _H("0x1.7934934f54c77p-58"), _H("0x1.3facb12d1755bp-4"),
             _H("-0x1.921915299468cp-58"), _H("0x1.fe7034129ef6fp-1"), _H("-0x1.cbf4337c96f97p-57"),
             _H("0x1.5f911fd10b737p-4"), _H("-0x1.0184f02be9102p-58"), _H("0x1.fe1c4c3c873ebp-1"),
             _H("-0x1.5a9c9057c4a02p-60"), _H("0x1.7f701032550e4p-4"), _H("0x1.afc2d1800501ap-60"),
             _H("0x1.fdc06bf7e6b9bp-1"), _H("0x1.31902b535f8dbp-55"), _H("0x1.9f4902d55d1f9p-4"),
             _H("0x1.2696d7eac1dc1p-58"), _H("0x1.fd5c94b43e000p-1"), _H("-0x1.2e768cb4f92f9p-57"),
             _H("0x1.bf1b78568391dp-4"), _H("0x1.e91841dea4cc8p-58"), _H("0x1.fcf0c800e99b1p-1"),
             _H("0x1.ea3d786d186acp-57"), _H("0x1.dee6f16c1cce6p-4"), _H("-0x1.50f8e2fb71673p-59"),
             _H("0x1.fc7d078d1bc88p-1"), _H("0x1.075d2447db685p-55"), _H("0x1.feaaeee86ee36p-4"),
             _H("-0x1.afcb2bcc6f03bp-59"), _H("0x1.fc015527d5bd3p-1"), _H("0x1.b68f35094efb8p-55"),
             _H("0x1.0f3378ddd71d1p-3"), _H("0x1.d8468724f0f9ep-57"), _H("0x1.fb7db2bfe0695p-1"),
             _H("0x1.21dadf4f65ab1p-55"), _H("0x1.1f0d3d7afceafp-3"), _H("-0x1.6ef95099769a5p-57"),
             _H("0x1.faf22263c4bd3p-1"), _H("-0x1.52ace133a2769p-58"), _H("0x1.2ee285e4ab88fp-3"),
             _H("-0x1.e4d0f05dee058p-57"), _H("0x1.fa5ea641c36f2p-1"), _H("0x1.04da6ed17cc7cp-59"),
             _H("0x1.3eb312c5d66cbp-3"), _H("0x1.47d666b66cb91p-57"), _H("0x1.f9c340a7cc428p-1"),
             _H("0x1.c5b6b063b7462p-55"), _H("0x1.4e7ea4dc5f27bp-3"), _H("0x1.949db2ac072fcp-58"),
             _H("0x1.f91ff40374d01p-1"), _H("-0x1.7d03f4d3a9e4cp-57"), _H("0x1.5e44fcfa126f3p-3"),
             _H("-0x1.6f443063f89b6p-57"), _H("0x1.f874c2e1eecf6p-1"), _H("-0x1.c6514e1332b16p-55"),
             _H("0x1.6e05dc05a4d4cp-3"), _H("-0x1.32c5c8b81c940p-66"), _H("0x1.f7c1afeffde24p-1"),
             _H("-0x1.8f55bc47540b1p-56"), _H("0x1.7dc102fbaf2b5p-3"), _H("0x1.5ab50e23c97c3p-59"),
             _H("0x1.f706bdf9ece1cp-1"), _H("-0x1.698c80c36dcb4p-55"), _H("0x1.8d7632efaa944p-3"),
             _H("-0x1.20fa262cbb953p-57"), _H("0x1.f643efeb82acdp-1"), _H("0x1.6b00ac1fe28acp-56"),
             _H("0x1.9d252d0cec312p-3"), _H("0x1.9c43d80b1137dp-58"), _H("0x1.f57948cff6797p-1"),
             _H("0x1.e3a0d3e03b1d5p-57"), _H("0x1.accdb297a0765p-3"), _H("-0x1.9883b57d6cdebp-58"),
             _H("0x1.f4a6cbd1e3a79p-1"), _H("0x1.13df0edaebb57p-55"), _H("0x1.bc6f84edc6199p-3"),
             _H("0x1.9c1a56a7b0cabp-57"), _H("0x1.f3cc7c3b3d16ep-1"), _H("-0x1.21a3ad28a3494p-57"),
             _H("0x1.cc0a6588289a3p-3"), _H("-0x1.868d09bc87c6bp-57"), _H("0x1.f2ea5d753ffedp-1"),
             _H("0x1.cc4215f56d583p-55"), _H("0x1.db9e15fb5a5d0p-3"), _H("-0x1.32e20d6cc6fc2p-57"),
             _H("0x1.f20073086649fp-1"), _H("0x1.b940416c1984bp-56"), _H("0x1.eb2a57f8ae5a3p-3"),
             _H("-0x1.0be06af572cebp-57"), _H("0x1.f10ec09c5873bp-1"), _H("0x1.d9072762c1283p-55"),
             _H("0x1.faaeed4f31577p-3"), _H("-0x1.15d88508e32b8p-57"), _H("0x1.f01549f7deea1p-1"),
             _H("0x1.d3c1e99e5cafdp-55"), _H("0x1.0515cbf65155cp-2"), _H("-0x1.9b8c29dfd8ec8p-56"),
             _H("0x1.ef141300d2f26p-1"), _H("-0x1.2aa1b08ded372p-55"), _H("0x1.0cd00cef36436p-2"),
             _H("-0x1.9fb0a0c93e2b5p-56"), _H("0x1.ee0b1fbc0f11cp-1"), _H("-0x1.bfd2380bbc3b1p-59"),
             _H("0x1.14861aa94ddebp-2"), _H("-0x1.be881b5b615a4p-57"), _H("0x1.ecfa744d5efa1p-1"),
             _H("-0x1.56d0a4af541d0p-58"), _H("0x1.1c37d64c6b876p-2"), _H("0x1.46076fe0dcff5p-56"),
             _H("0x1.ebe214f76efa8p-1"), _H("-0x1.02f9f12ba543ep-55"), _H("0x1.23e52111aaf36p-2"),
             _H("-0x1.4f080334eff18p-56"), _H("0x1.eac2061bbaf4fp-1"), _H("0x1.2c1d53e94658dp-57"),
             _H("0x1.2b8ddc43eb49fp-2"), _H("0x1.1553899f2d807p-57"), _H("0x1.e99a4c3a7cd83p-1"),
             _H("-0x1.2264b1bc53ce8p-55"), _H("0x1.3331e94049f87p-2"), _H("0x1.e0cb6b40c302cp-56"),
             _H("0x1.e86aebf29a9edp-1"), _H("0x1.9397afdbb58a7p-55"), _H("0x1.3ad129769d3d8p-2"),
             _H("0x1.03d5504878398p-63"), _H("0x1.e733ea0193d40p-1"), _H("-0x1.6428b3546ce13p-55"),
             _H("0x1.426b7e69ee697p-2"), _H("-0x1.f09c75705c59fp-56"), _H("0x1.e5f54b436e9d0p-1"),
             _H("0x1.7eb0fd02fc8bcp-55"), _H("0x1.4a00c9b0f3d20p-2"), _H("0x1.823ba6bb08eadp-56"),
             _H("0x1.e4af14b2a449cp-1"), _H("-0x1.68ca02e8a6833p-55"), _H("0x1.5190ecf68a77ap-2"),
             _H("0x1.b357155eef0f3p-56"), _H("0x1.e3614b680d6a5p-1"), _H("-0x1.27793aa015237p-56"),
             _H("0x1.591bc9fa2f597p-2"), _H("0x1.7c74bac3fe0cbp-57"), _H("0x1.e20bf49acd6c1p-1"),
             _H("-0x1.660aec7ef636cp-58"), _H("0x1.60a1429078775p-2"), _H("0x1.b1fd80ba89133p-58"),
             _H("0x1.e0af15a03dbcep-1"), _H("0x1.fe8e702771ae6p-58"), _H("0x1.682138a38d7f7p-2"),
             _H("-0x1.d889202444aadp-56"), _H("0x1.df4ab3ebd875ep-1"), _H("-0x1.e2d8a7e6736c4p-55"),
             _H("0x1.6f9b8e33a0255p-2"), _H("0x1.42bc14ee9da0dp-56"), _H("0x1.ddded50f228d6p-1"),
             _H("-0x1.e80c8d42ba2bfp-57"), _H("0x1.7710255764214p-2"), _H("-0x1.6ead7314bb6cep-57"),
             _H("0x1.dc6b7eb995912p-1"), _H("0x1.4b364776dcd35p-58"), _H("0x1.7e7ee03c86d4ep-2"),
             _H("-0x1.b63bcdabf5af2p-56"), _H("0x1.daf0b6b888e83p-1"), _H("0x1.a249e2b5e5ceap-55"),
             _H("0x1.85e7a12826949p-2"), _H("0x1.8a40e9b5face0p-56"), _H("0x1.d96e82f71a9dcp-1"),
             _H("0x1.ff61bd5d2039dp-55"), _H("0x1.8d4a4a774992fp-2"), _H("0x1.44a02ea766326p-56"),
             _H("0x1.d7e4e97e17b4ap-1"), _H("-0x1.3b770352bed94p-57"), _H("0x1.94a6be9f546c5p-2"),
             _H("-0x1.69ce13e683f58p-56"), _H("0x1.d653f073e4040p-1"), _H("-0x1.76236434bec37p-55"),
             _H("0x1.9bfce02e80510p-2"), _H("0x1.09e39a320b0a4p-56"), _H("0x1.d4bb9e1c619e0p-1"),
             _H("0x1.f34bb77858f61p-55"), _H("0x1.a34c91cc50ccap-2"), _H("-0x1.a310e3b50cecdp-58"),
             _H("0x1.d31bf8d8d7c06p-1"), _H("0x1.e60dd3089cbddp-56"), _H("0x1.aa95b63a09277p-2"),
             _H("-0x1.6293eb13c0381p-57"), _H("0x1.d1750727d94f0p-1"), _H("0x1.0d52b1ec1a48ep-55"),
             _H("0x1.b1d8305321617p-2"), _H("-0x1.ae242cb99f519p-56"), _H("0x1.cfc6cfa52ad9fp-1"),
             _H("0x1.8b5b5508f2a0dp-55"), _H("0x1.b913e30dbac43p-2"), _H("-0x1.e38ad2f6c3ff1p-56"),
             _H("0x1.ce115909a82e5p-1"), _H("0x1.1f139bb31109ap-55"), _H("0x1.c048b17b140a3p-2"),
             _H("0x1.19fe6757e9fa7p-57"), _H("0x1.cc54aa2b2972ep-1"), _H("0x1.4ee162ba83a98p-57"),
             _H("0x1.c7767ec7fd19ep-2"), _H("-0x1.eb14d1a3d5826p-58"), _H("0x1.ca90c9fc67d0bp-1"),
             _H("-0x1.46a81485e3462p-57"), _H("0x1.ce9d2e3d4a51fp-2"), _H("-0x1.2fc8a12dae298p-57"),
             _H("0x1.c8c5bf8ce1a84p-1"), _H("0x1.ab3d1a1590123p-56"), _H("0x1.d5bca34047661p-2"),
             _H("0x1.28a44a75fc29cp-56"), _H("0x1.c6f39208be53bp-1"), _H("-0x1.741dbfbaadb42p-55"),
             _H("0x1.dcd4c15329c9ap-2"), _H("0x1.0d4c6e171fd9ap-56"), _H("0x1.c51a48b8b175ep-1"),
             _H("-0x1.1bbb43b9aa880p-57"), _H("0x1.e3e56c1582a69p-2"), _H("-0x1.0a4821099f88fp-58"),
             _H("0x1.c339eb01ddd81p-1"), _H("-0x1.caaf5ee82c5c0p-55"), _H("0x1.eaee8744b05f0p-2"),
             _H("-0x1.789b43c9b027dp-58"), _H("0x1.c1528065b7d50p-1"), _H("-0x1.892111312e828p-55"),
             _H("0x1.f1eff6bc4f97bp-2"), _H("0x1.17212f8a7525cp-56"), _H("0x1.bf641081e7536p-1"),
             _H("0x1.b7bd71628a9a1p-55"), _H("0x1.f8e99e76abc97p-2"), _H("0x1.9d950af2d00a3p-58"),
             _H("0x1.bd6ea310294f5p-1"), _H("0x1.31bbcc88c109dp-56"), _H("0x1.ffdb628d2f57ap-2"),
             _H("0x1.f4a992e905b6ap-57"), _H("0x1.bb723fe630f32p-1"), _H("0x1.72bd2452d0a39p-56"),
             _H("0x1.0362939c69955p-1"), _H("-0x1.2d8cd78397b01p-55"), _H("0x1.b96eeef58840ep-1"),
             _H("0x1.45a3cc78fade0p-58"), _H("0x1.06d3686946e5bp-1"), _H("0x1.3f5ae4538ff1bp-55"),
             _H("0x1.b764b84b704c2p-1"), _H("-0x1.f5848c21b389bp-55"), _H("0x1.0a4021e9e1001p-1"),
             _H("-0x1.6f643a13914f6p-55"), _H("0x1.b553a410c104ep-1"), _H("0x1.8ff7947027a16p-58"),
             _H("0x1.0da8b26b5672ep-1"), _H("-0x1.a58def0bee909p-55"), _H("0x1.b33bba89c8948p-1"),
             _H("0x1.ea6a51d1f6ca9p-55"), _H("0x1.110d0c4b69c3bp-1"), _H("0x1.d918998809981p-55"),
             _H("0x1.b11d04162a4c6p-1"), _H("0x1.1dd561efbc0c2p-56"), _H("0x1.146d21f8b7f82p-1"),
             _H("0x1.bf9535e2739a8p-56"), _H("0x1.aef78930bd275p-1"), _H("-0x1.f836279746f94p-56"),
             _H("0x1.17c8e5f2eedb0p-1"), _H("0x1.35e57102e2488p-57"), _H("0x1.accb526f69de5p-1"),
             _H("0x1.8fb6a8dd6b6ccp-55"), _H("0x1.1b204acb02fddp-1"), _H("-0x1.f190c70cbb5ffp-58"),
             _H("0x1.aa98688308913p-1"), _H("-0x1.b83d607cd5070p-63"), _H("0x1.1e7343236574cp-1"),
             _H("0x1.22a3fa4f41d5ap-56"), _H("0x1.a85ed4373e02dp-1"), _H("0x1.9be06385ec792p-57"),
             _H("0x1.21c1c1b0394cfp-1"), _H("0x1.e5b324b23aa31p-58"), _H("0x1.a61e9e72586afp-1"),
             _H("0x1.58330e2fd453fp-55"), _H("0x1.250bb93788bbbp-1"), _H("0x1.ea3d02457bccep-56"),
             _H("0x1.a3d7d0352bdcfp-1"), _H("-0x1.68dbaeca19669p-55"), _H("0x1.28511c917a067p-1"),
             _H("-0x1.01df1d9a16b70p-55"), _H("0x1.a18a729aee445p-1"), _H("0x1.95e25736c0358p-60"),
             _H("0x1.2b91dea88421ep-1"), _H("-0x1.fa371db216ab0p-55"), _H("0x1.9f368ed912f85p-1"),
             _H("-0x1.1d200c5791606p-55"), _H("0x1.2ecdf279a3082p-1"), _H("0x1.d3557e0e7e37ep-55"),
             _H("0x1.9cdc2e3f25e5cp-1"), _H("0x1.3f99112993f62p-55"), _H("0x1.32054b148bc4fp-1"),
             _H("0x1.f6b42095a135bp-55"), _H("0x1.9a7b5a36a6514p-1"), _H("0x1.722cfcc9fa7a9p-55"),
             _H("0x1.3537db9be0367p-1"), _H("0x1.b327e7af040f0p-57"), _H("0x1.98141c42e1310p-1"),
             _H("0x1.d1ff80488f08dp-55"), _H("0x1.386597456282bp-1"), _H("-0x1.10fada93b07a8p-56"),
             _H("0x1.95a67e00cb1fdp-1"), _H("-0x1.0befda21f862dp-55"), _H("0x1.3b8e715a2840ap-1"),
             _H("-0x1.97653a7d2f07bp-56"), _H("0x1.93328926d9e92p-1"), _H("-0x1.bb77003600cdap-55"),
             _H("0x1.3eb25d36cd53ap-1"), _H("-0x1.be570e1570fc0p-58"), _H("0x1.90b84784ddaf7p-1"),
             _H("-0x1.0feb10ab93b87p-56"), _H("0x1.41d14e4ba6790p-1"), _H("0x1.4608fd287ecf5p-55"),
             _H("0x1.8e37c303d9ad1p-1"), _H("-0x1.463a4b53d4bf8p-57"), _H("0x1.44eb381cf386bp-1"),
             _H("-0x1.3ed6c1e6a5505p-55"), _H("0x1.8bb105a5dc900p-1"), _H("0x1.863e03e9474c1p-55"),
             _H("0x1.48000e431159fp-1"), _H("-0x1.b194a7463ed10p-55"), _H("0x1.89241985d871fp-1"),
             _H("0x1.c48d9c413ed84p-55"), _H("0x1.4b0fc46aab761p-1"), _H("0x1.0da05738cc59ap-61"),
             _H("0x1.869108d77a6c6p-1"), _H("0x1.338ffe2bfe9ddp-56"), _H("0x1.4e1a4e54ed51bp-1"),
             _H("-0x1.a492f89b7c76ap-55"), _H("0x1.83f7dde701ca0p-1"), _H("-0x1.152cf609bc6e8p-59"),
             _H("0x1.511f9fd7b351cp-1"), _H("-0x1.5c0e861c48831p-55"), _H("0x1.8158a31916d5dp-1"),
             _H("-0x1.de8b90b8228dep-57"), _H("0x1.541facddbb724p-1"), _H("0x1.232c28520d391p-56"),
             _H("0x1.7eb362eaa1488p-1"), _H("0x1.a1d65a4a5959fp-58"), _H("0x1.571a6966d59b3p-1"),
             _H("0x1.c843b4d0fb198p-58"), _H("0x1.7c0827f09e54fp-1"), _H("-0x1.c73d6d72aee68p-57"),
             _H("0x1.5a0fc98813a12p-1"), _H("-0x1.d82e2b7d4227bp-55"), _H("0x1.7956fcd7f6543p-1"),
             _H("-0x1.ab276e9d45ae4p-55"), _H("0x1.5cffc16bf8f0dp-1"), _H("0x1.96cb370eb578ap-55"),
             _H("0x1.769fec655211fp-1"), _H("-0x1.827d5cf8c68c5p-57"), _H("0x1.5fea4552a9e57p-1"),
             _H("0x1.0b6cef7ee20b7p-55"), _H("0x1.73e30174efba1p-1"), _H("-0x1.5d3ae3d94ad5fp-57"),
             _H("0x1.62cf49921ac79p-1"), _H("-0x1.edd9855b6241ap-55"), _H("0x1.712046fa77678p-1"),
             _H("0x1.425b0a5029c81p-55"), _H("0x1.65aec2963e755p-1"), _H("0x1.126f96b71053cp-55"),
             _H("0x1.6e57c800cf55ep-1"), _H("0x1.60286dedbd0a6p-55"), _H("0x1.6888a4e134b2fp-1"),
             _H("-0x1.6b7d37644d5e6p-55"), _H("0x1.6b898fa9efb5dp-1"), _H("0x1.15ac786ccf4b2p-56"),
             _H("0x1.6b5ce50b7821ap-1"), _H("-0x1.5d5158f702e0fp-57"), _H("0x1.68b5a92eb6253p-1"),
             _H("-0x1.9a91ad985f89cp-55"), _H("0x1.6e2b77c40bde1p-1"), _H("-0x1.0e729857fad53p-56"),
             _H("0x1.65dc1fdeb8cbap-1"), _H("-0x1.97c1b47337c77p-58"), _H("0x1.70f451d0a8c40p-1"),
             _H("0x1.97ede3885770dp-57"), _H("0x1.62fcff20191c7p-1"), _H("0x1.d9143895756efp-57"),
             _H("0x1.73b7680dea578p-1"), _H("-0x1.2248306dc12a2p-56"), _H("0x1.6018526f563dfp-1"),
             _H("0x1.46ca5e0e432d0p-55"), _H("0x1.7674af6f7b524p-1"), _H("0x1.e9d3f94ac84a8p-56"),
             _H("0x1.5d2e255f1f17ap-1"), _H("0x1.0314104c8892bp-55"), _H("0x1.792c1d0041d52p-1"),
             _H("-0x1.abf05eeb354ebp-55"), _H("0x1.5a3e839824077p-1"), _H("0x1.428aa2759be62p-55"),
             _H("0x1.7bdda5e28b3c2p-1"), _H("0x1.ad1197ccd0393p-59"), _H("0x1.574978d8e83f2p-1"),
             _H("0x1.f4714af282d23p-55"), _H("0x1.7e893f5037959p-1"), _H("0x1.0eefbaa650c4cp-55"),
             _H("0x1.544f10f592ca5p-1"), _H("-0x1.e7ae8e6c7a62fp-55"), _H("0x1.812ede9ae4ba4p-1"),
             _H("-0x1.7830adf402ddap-55"), _H("0x1.514f57d7bf3dap-1"), _H("0x1.47a108073c259p-56"))
K_TINY_SIN = 0x3e500000
K_TINY_COS = 0x3e400000
K_DO_SIN = 0x3feb6000
K_MID = 0x400368fd
K_REDUCE = 0x419921fb
K_HUGE = 0x7ff00000
BR_TM600 = _H("0x1.0000000000000p-600")
BR_SPLIT = _H("0x1.0000002000000p+27")
BR_T576_HI = 0x63f00000
BR_TM24 = _H("0x1.0000000000000p-24")
BR_BIG = _H("0x1.8000000000000p+52")
BR_BIG1 = _H("0x1.8000000000000p+54")
BR_EXP_LIMIT = 0x1aa
BR_EXP_BIAS = 0x1c2
BR_TOVERP = (_H("0x1.45f3060000000p+23"), _H("0x1.b939100000000p+22"), _H("0x1.529fc00000000p+20"),
             _H("0x1.3abe880000000p+21"), _H("0x1.ea69ba0000000p+23"), _H("0x1.81b6c40000000p+23"),
             _H("0x1.2b32780000000p+23"), _H("0x1.0e41040000000p+22"), _H("0x1.fca2c60000000p+23"),
             _H("0x1.57bd760000000p+23"), _H("0x1.8ac36e0000000p+23"), _H("0x1.2371d00000000p+21"),
             _H("0x1.0937480000000p+22"), _H("0x1.c00c920000000p+23"), _H("0x1.7750480000000p+21"),
             _H("0x1.a324380000000p+23"), _H("0x1.fc3bd60000000p+23"), _H("0x1.cb12900000000p+20"),
             _H("0x1.4e7dd00000000p+23"), _H("0x1.046bea0000000p+23"), _H("0x1.75da200000000p+21"),
             _H("0x1.09d3380000000p+23"), _H("0x1.c09ad00000000p+22"), _H("0x1.7df9040000000p+22"),
             _H("0x1.cc8eb00000000p+21"), _H("0x1.cc1a980000000p+21"), _H("0x1.cfa4e00000000p+21"),
             _H("0x1.08bf160000000p+23"), _H("0x1.7bf2500000000p+23"), _H("0x1.d8ffc00000000p+21"),
             _H("0x1.2fffbc0000000p+23"), _H("0x1.6603c00000000p+18"), _H("0x1.de5e220000000p+23"),
             _H("0x1.16b4140000000p+23"), _H("0x1.b47db40000000p+22"), _H("0x1.b3f6780000000p+21"),
             _H("0x1.3e58480000000p+21"), _H("0x1.6e9e8c0000000p+23"), _H("0x1.fb34f00000000p+21"),
             _H("0x1.7fa8b40000000p+22"), _H("0x1.d49ee80000000p+22"), _H("0x1.8fd7ca0000000p+23"),
             _H("0x1.e2f67a0000000p+23"), _H("0x1.ce7dc00000000p+18"), _H("0x1.14a5240000000p+23"),
             _H("0x1.d4d7f60000000p+23"), _H("0x1.7ec47c0000000p+22"), _H("0x1.1aba100000000p+23"),
             _H("0x1.580cc00000000p+22"), _H("0x1.1bf1ec0000000p+22"), _H("0x1.aeafc00000000p+22"),
             _H("0x1.9f78400000000p+23"), _H("0x1.35e86c0000000p+23"), _H("0x1.da9e300000000p+20"),
             _H("0x1.22c2bc0000000p+23"), _H("0x1.cc36100000000p+23"), _H("0x1.9666140000000p+22"),
             _H("0x1.7c52800000000p+22"), _H("0x1.a102340000000p+22"), _H("0x1.ffb1000000000p+23"),
             _H("0x1.35cc9c0000000p+22"), _H("0x1.8830300000000p+21"), _H("0x1.556ca00000000p+20"),
             _H("0x1.cea3240000000p+22"), _H("0x1.8389ec0000000p+22"), _H("0x1.8118d60000000p+23"),
             _H("0x1.1f10640000000p+22"), _H("0x1.86cf9a0000000p+23"), _H("0x1.b9d0120000000p+23"),
             _H("0x1.541ac80000000p+21"), _H("0x1.88ed160000000p+23"), _H("0x1.2c394c0000000p+23"),
             _H("0x1.bb5e880000000p+23"), _H("0x1.a2ae320000000p+23"), _H("0x1.4fa9400000000p+18"))
BR_TOVERP0 = _H("0x1.45f3060000000p+23")
BR_HP0 = _H("0x1.921fb54442d18p+0")
BR_MP1 = _H("0x1.921fb58000000p+0")
BR_MP2 = _H("-0x1.dde9740000000p-27")
BR_HP1 = _H("0x1.1a62633145c07p-54")
LOG_NEG_LO = 0xc012000000000000
LOG_RANGE = 0x308ffffffffff
LOG_NEG_OFF = 0xc01a000000000000
LOG_TWO27 = _H("0x1.0000000000000p+27")
LOG_TWO52 = _H("0x1.0000000000000p+52")
LOG_NEG_52 = 0xfcc0000000000000
LOG_LN2HI = _H("0x1.62e42fefa3800p-1")
LOG_LN2LO = _H("0x1.ef35793c76730p-45")
LOG_A = (_H("-0x1.0000000000001p-1"), _H("0x1.555555551305bp-2"), _H("-0x1.fffffffeb4590p-3"),
         _H("0x1.999b324f10111p-3"), _H("-0x1.55575e506c89fp-3"))
LOG_B = (_H("-0x1.0000000000000p-1"), _H("0x1.5555555555577p-2"), _H("-0x1.ffffffffffdcbp-3"),
         _H("0x1.999999995dd0cp-3"), _H("-0x1.55555556745a7p-3"), _H("0x1.24924a344de30p-3"),
         _H("-0x1.fffffa4423d65p-4"), _H("0x1.c7184282ad6cap-4"), _H("-0x1.999eb43b068ffp-4"),
         _H("0x1.78182f7afd085p-4"), _H("-0x1.5521375d145cdp-4"))
LOG_TAB = ((_H("0x1.734f0c3e0de9fp+0"), _H("-0x1.7cc7f79e69000p-2")),
           (_H("0x1.713786a2ce91fp+0"), _H("-0x1.76feec20d0000p-2")),
           (_H("0x1.6f26008fab5a0p+0"), _H("-0x1.713e31351e000p-2")),
           (_H("0x1.6d1a61f138c7dp+0"), _H("-0x1.6b85b38287800p-2")),
           (_H("0x1.6b1490bc5b4d1p+0"), _H("-0x1.65d5590807800p-2")),
           (_H("0x1.69147332f0cbap+0"), _H("-0x1.602d076180000p-2")),
           (_H("0x1.6719f18224223p+0"), _H("-0x1.5a8ca86909000p-2")),
           (_H("0x1.6524f99a51ed9p+0"), _H("-0x1.54f4356035000p-2")),
           (_H("0x1.63356aa8f24c4p+0"), _H("-0x1.4f637c36b4000p-2")),
           (_H("0x1.614b36b9ddc14p+0"), _H("-0x1.49da7fda85000p-2")),
           (_H("0x1.5f66452c65c4cp+0"), _H("-0x1.445923989a800p-2")),
           (_H("0x1.5d867b5912c4fp+0"), _H("-0x1.3edf439b0b800p-2")),
           (_H("0x1.5babccb5b90dep+0"), _H("-0x1.396ce448f7000p-2")),
           (_H("0x1.59d61f2d91a78p+0"), _H("-0x1.3401e17bda000p-2")),
           (_H("0x1.5805612465687p+0"), _H("-0x1.2e9e2ef468000p-2")),
           (_H("0x1.56397cee76bd3p+0"), _H("-0x1.2941b3830e000p-2")),
           (_H("0x1.54725e2a77f93p+0"), _H("-0x1.23ec58cda8800p-2")),
           (_H("0x1.52aff42064583p+0"), _H("-0x1.1e9e129279000p-2")),
           (_H("0x1.50f22dbb2bddfp+0"), _H("-0x1.1956d2b48f800p-2")),
           (_H("0x1.4f38f4734ded7p+0"), _H("-0x1.141679ab9f800p-2")),
           (_H("0x1.4d843cfde2840p+0"), _H("-0x1.0edd094ef9800p-2")),
           (_H("0x1.4bd3ec078a3c8p+0"), _H("-0x1.09aa518db1000p-2")),
           (_H("0x1.4a27fc3e0258ap+0"), _H("-0x1.047e65263b800p-2")),
           (_H("0x1.4880524d48434p+0"), _H("-0x1.feb224586f000p-3")),
           (_H("0x1.46dce1b192d0bp+0"), _H("-0x1.f474a7517b000p-3")),
           (_H("0x1.453d9d3391854p+0"), _H("-0x1.ea4443d103000p-3")),
           (_H("0x1.43a2744b4845ap+0"), _H("-0x1.e020d44e9b000p-3")),
           (_H("0x1.420b54115f8fbp+0"), _H("-0x1.d60a22977f000p-3")),
           (_H("0x1.40782da3ef4b1p+0"), _H("-0x1.cc00104959000p-3")),
           (_H("0x1.3ee8f5d57fe8fp+0"), _H("-0x1.c202956891000p-3")),
           (_H("0x1.3d5d9a00b4ce9p+0"), _H("-0x1.b81178d811000p-3")),
           (_H("0x1.3bd60c010c12bp+0"), _H("-0x1.ae2c9ccd3d000p-3")),
           (_H("0x1.3a5242b75dab8p+0"), _H("-0x1.a45402e129000p-3")),
           (_H("0x1.38d22cd9fd002p+0"), _H("-0x1.9a877681df000p-3")),
           (_H("0x1.3755bc5847a1cp+0"), _H("-0x1.90c6d69483000p-3")),
           (_H("0x1.35dce49ad36e2p+0"), _H("-0x1.87120a645c000p-3")),
           (_H("0x1.34679984dd440p+0"), _H("-0x1.7d68fb4143000p-3")),
           (_H("0x1.32f5cceffcb24p+0"), _H("-0x1.73cb83c627000p-3")),
           (_H("0x1.3187775a10d49p+0"), _H("-0x1.6a39a9b376000p-3")),
           (_H("0x1.301c8373e3990p+0"), _H("-0x1.60b3154b7a000p-3")),
           (_H("0x1.2eb4ebb95f841p+0"), _H("-0x1.5737d76243000p-3")),
           (_H("0x1.2d50a0219a9d1p+0"), _H("-0x1.4dc7b8fc23000p-3")),
           (_H("0x1.2bef9a8b7fd2ap+0"), _H("-0x1.4462c51d20000p-3")),
           (_H("0x1.2a91c7a0c1babp+0"), _H("-0x1.3b08abc830000p-3")),
           (_H("0x1.293726014b530p+0"), _H("-0x1.31b996b490000p-3")),
           (_H("0x1.27dfa5757a1f5p+0"), _H("-0x1.2875490a44000p-3")),
           (_H("0x1.268b39b1d3bbfp+0"), _H("-0x1.1f3b9f879a000p-3")),
           (_H("0x1.2539d838ff5bdp+0"), _H("-0x1.160c8252ca000p-3")),
           (_H("0x1.23eb7aac9083bp+0"), _H("-0x1.0ce7f57f72000p-3")),
           (_H("0x1.22a012ba940b6p+0"), _H("-0x1.03cdc49fea000p-3")),
           (_H("0x1.2157996cc4132p+0"), _H("-0x1.f57bdbc4b8000p-4")),
           (_H("0x1.201201dd2fc9bp+0"), _H("-0x1.e370896404000p-4")),
           (_H("0x1.1ecf4494d480bp+0"), _H("-0x1.d17983ef94000p-4")),
           (_H("0x1.1d8f5528f6569p+0"), _H("-0x1.bf9674ed8a000p-4")),
           (_H("0x1.1c52311577e7cp+0"), _H("-0x1.adc79202f6000p-4")),
           (_H("0x1.1b17c74cb26e9p+0"), _H("-0x1.9c0c3e7288000p-4")),
           (_H("0x1.19e010c2c1ab6p+0"), _H("-0x1.8a646b372c000p-4")),
           (_H("0x1.18ab07bb670bdp+0"), _H("-0x1.78d01b3ac0000p-4")),
           (_H("0x1.1778a25efbcb6p+0"), _H("-0x1.674f145380000p-4")),
           (_H("0x1.1648d354c31dap+0"), _H("-0x1.55e0e6d878000p-4")),
           (_H("0x1.151b990275fddp+0"), _H("-0x1.4485cdea1e000p-4")),
           (_H("0x1.13f0ea432d24cp+0"), _H("-0x1.333d94d6aa000p-4")),
           (_H("0x1.12c8b7210f9dap+0"), _H("-0x1.22079f8c56000p-4")),
           (_H("0x1.11a3028ecb531p+0"), _H("-0x1.10e4698622000p-4")),
           (_H("0x1.107fbda8434afp+0"), _H("-0x1.ffa6c6ad20000p-5")),
           (_H("0x1.0f5ee0f4e6bb3p+0"), _H("-0x1.dda8d4a774000p-5")),
           (_H("0x1.0e4065d2a9fcep+0"), _H("-0x1.bbcece4850000p-5")),
           (_H("0x1.0d244632ca521p+0"), _H("-0x1.9a1894012c000p-5")),
           (_H("0x1.0c0a77ce2981ap+0"), _H("-0x1.788583302c000p-5")),
           (_H("0x1.0af2f83c636d1p+0"), _H("-0x1.5715e67d68000p-5")),
           (_H("0x1.09ddb98a01339p+0"), _H("-0x1.35c8a49658000p-5")),
           (_H("0x1.08cabaf52e7dfp+0"), _H("-0x1.149e364154000p-5")),
           (_H("0x1.07b9f2f4e28fbp+0"), _H("-0x1.e72c082eb8000p-6")),
           (_H("0x1.06ab58c358f19p+0"), _H("-0x1.a55f152528000p-6")),
           (_H("0x1.059eea5ecf92cp+0"), _H("-0x1.63d62cf818000p-6")),
           (_H("0x1.04949cdd12c90p+0"), _H("-0x1.228fb8caa0000p-6")),
           (_H("0x1.038c6c6f0ada9p+0"), _H("-0x1.c317b20f90000p-7")),
           (_H("0x1.02865137932a9p+0"), _H("-0x1.419355daa0000p-7")),
           (_H("0x1.0182427ea7348p+0"), _H("-0x1.81203c2ec0000p-8")),
           (_H("0x1.008040614b195p+0"), _H("-0x1.0040979240000p-9")),
           (_H("0x1.fe01ff726fa1ap-1"), _H("0x1.feff384900000p-9")),
           (_H("0x1.fa11cc261ea74p-1"), _H("0x1.7dc41353d0000p-7")),
           (_H("0x1.f6310b081992ep-1"), _H("0x1.3cea3c4c28000p-6")),
           (_H("0x1.f25f63ceeadcdp-1"), _H("0x1.b9fc114890000p-6")),
           (_H("0x1.ee9c8039113e7p-1"), _H("0x1.1b0d8ce110000p-5")),
           (_H("0x1.eae8078cbb1abp-1"), _H("0x1.58a5bd001c000p-5")),
           (_H("0x1.e741aa29d0c9bp-1"), _H("0x1.95c8340d88000p-5")),
           (_H("0x1.e3a91830a99b5p-1"), _H("0x1.d276aef578000p-5")),
           (_H("0x1.e01e009609a56p-1"), _H("0x1.07598e598c000p-4")),
           (_H("0x1.dca01e577bb98p-1"), _H("0x1.253f5e30d2000p-4")),
           (_H("0x1.d92f20b7c9103p-1"), _H("0x1.42edd8b380000p-4")),
           (_H("0x1.d5cac66fb5ccep-1"), _H("0x1.606598757c000p-4")),
           (_H("0x1.d272caa5ede9dp-1"), _H("0x1.7da76356a0000p-4")),
           (_H("0x1.cf26e3e6b2ccdp-1"), _H("0x1.9ab434e1c6000p-4")),
           (_H("0x1.cbe6da2a77902p-1"), _H("0x1.b78c7bb0d6000p-4")),
           (_H("0x1.c8b266d37086dp-1"), _H("0x1.d431332e72000p-4")),
           (_H("0x1.c5894bd5d5804p-1"), _H("0x1.f0a3171de6000p-4")),
           (_H("0x1.c26b533bb9f8cp-1"), _H("0x1.067152b914000p-3")),
           (_H("0x1.bf583eeece73fp-1"), _H("0x1.147858292b000p-3")),
           (_H("0x1.bc4fd75db96c1p-1"), _H("0x1.2266ecdca3000p-3")),
           (_H("0x1.b951e0c864a28p-1"), _H("0x1.303d7a6c55000p-3")),
           (_H("0x1.b65e2c5ef3e2cp-1"), _H("0x1.3dfc33c331000p-3")),
           (_H("0x1.b374867c9888bp-1"), _H("0x1.4ba366b7a8000p-3")),
           (_H("0x1.b094b211d304ap-1"), _H("0x1.5933928d1f000p-3")),
           (_H("0x1.adbe885f2ef7ep-1"), _H("0x1.66acd2418f000p-3")),
           (_H("0x1.aaf1d31603da2p-1"), _H("0x1.740f8ec669000p-3")),
           (_H("0x1.a82e63fd358a7p-1"), _H("0x1.815c0f51af000p-3")),
           (_H("0x1.a5740ef09738bp-1"), _H("0x1.8e92954f68000p-3")),
           (_H("0x1.a2c2a90ab4b27p-1"), _H("0x1.9bb3602f84000p-3")),
           (_H("0x1.a01a01393f2d1p-1"), _H("0x1.a8bed1c2c0000p-3")),
           (_H("0x1.9d79f24db3c1bp-1"), _H("0x1.b5b515c01d000p-3")),
           (_H("0x1.9ae2505c7b190p-1"), _H("0x1.c2967ccbcc000p-3")),
           (_H("0x1.9852ef297ce2fp-1"), _H("0x1.cf635d5486000p-3")),
           (_H("0x1.95cbaeea44b75p-1"), _H("0x1.dc1bd3446c000p-3")),
           (_H("0x1.934c69de74838p-1"), _H("0x1.e8c01b8cfe000p-3")),
           (_H("0x1.90d4f2f6752e6p-1"), _H("0x1.f5509c0179000p-3")),
           (_H("0x1.8e6528effd79dp-1"), _H("0x1.00e6c121fb800p-2")),
           (_H("0x1.8bfce9fcc007cp-1"), _H("0x1.071b80e93d000p-2")),
           (_H("0x1.899c0dabec30ep-1"), _H("0x1.0d46b9e867000p-2")),
           (_H("0x1.87427aa2317fbp-1"), _H("0x1.13687334bd000p-2")),
           (_H("0x1.84f00acb39a08p-1"), _H("0x1.1980d67234800p-2")),
           (_H("0x1.82a49e8653e55p-1"), _H("0x1.1f8ffe0cc8000p-2")),
           (_H("0x1.8060195f40260p-1"), _H("0x1.2595fd7636800p-2")),
           (_H("0x1.7e22563e0a329p-1"), _H("0x1.2b9300914a800p-2")),
           (_H("0x1.7beb377dcb5adp-1"), _H("0x1.3187210436000p-2")),
           (_H("0x1.79baa679725c2p-1"), _H("0x1.377266dec1800p-2")),
           (_H("0x1.77907f2170657p-1"), _H("0x1.3d54ffbaf3000p-2")),
           (_H("0x1.756cadbd6130cp-1"), _H("0x1.432eee32fe000p-2")))
# the addresses in libm.so.6 the tables and the constants loaded from memory were read at
ADDRESSES = {"SINCOSTAB": 0x734d8, "BR_TOVERP": 0x73228, "LOG_DATA": 0x74b08, "TAYLOR_LIMIT": 0x6cec8, "S5": 0x6ced0,
             "S4": 0x6ced8, "S3": 0x6cee0, "S2": 0x6cee8, "CS6": 0x6cef0, "SN5": 0x6cef8, "CS4": 0x6cf00,
             "SN3": 0x6cf08, "HPINV": 0x69148, "MP1": 0x6cf18, "MP2": 0x6cf20, "PP3": 0x6cf28, "PP4": 0x6cf30,
             "HP0": 0x603e0, "HP1": 0x603d8, "NEG_HP1": 0x6cf10, "BR_TOVERP0": 0x73218, "BR_HP0": 0x603e0,
             "BR_MP1": 0x6cf18, "BR_MP2": 0x73220, "BR_HP1": 0x603d8}


def bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def dbl(u: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", u & M64))[0]


def _fmsub(n: float, m: float, a: float) -> float:
    """aarch64's fmsub: a - n m, rounded once."""
    return fma(-n, m, a)


def _fnmsub(n: float, m: float, a: float) -> float:
    """aarch64's fnmsub: n m - a, rounded once."""
    return fma(n, m, -a)


# ================================================================================================= sin and cos

def taylor_sin(a: float, da: float) -> float:
    """TAYLOR_SIN(a a, a, da): a + ((POLY(a a) a - da / 2) a a + da), the polynomial and both steps fused."""
    xx = a * a
    poly = fma(xx, fma(xx, fma(xx, fma(xx, S5, S4), S3), S2), S1)
    t = _fnmsub(poly, a, da * 0.5)
    return fma(xx, t, da) + a


def _table(x: float) -> tuple:
    """u = big + |x| and the table row of its low word: (|x| - (u - big), sn, ssn, cs, ccs)."""
    u = BIG + abs(x)
    k = (bits(u) & 0xFFFFFFFF) << 2
    return abs(x) - (u - BIG), SINCOSTAB[k], SINCOSTAB[k + 1], SINCOSTAB[k + 2], SINCOSTAB[k + 3]


def do_sin(x: float, dx: float) -> float:
    """do_sin: the Taylor polynomial below 0.126, else the table's sin and cos of the nearest 1/128 combined with
    short polynomials in the rest; the sign of x."""
    if abs(x) < TAYLOR_LIMIT:
        return taylor_sin(x, dx)
    if x <= 0.0:
        dx = -dx
    r, sn, ssn, cs, ccs = _table(x)
    xx = r * r
    pc = fma(xx, CS6, CS4)
    ps = fma(xx, SN5, SN3)
    r3 = r * xx
    c1 = fma(xx, pc, CS2)
    s = r + fma(r3, ps, dx)
    c = fma(r, dx, xx * c1)
    cor = fma(s, cs, _fmsub(c, sn, fma(s, ccs, ssn)))
    return math.copysign(sn + cor, x)


def do_cos(x: float, dx: float) -> float:
    """do_cos: the table's cos and sin of the nearest 1/128 combined with short polynomials in the rest."""
    if x < 0.0:
        dx = -dx
    r, sn, ssn, cs, ccs = _table(x)
    r = r + dx
    xx = r * r
    pc = fma(xx, CS6, CS4)
    ps = fma(xx, SN5, SN3)
    r3 = r * xx
    c1 = fma(xx, pc, CS2)
    s = fma(r3, ps, r)
    c = xx * c1
    cor = _fmsub(s, sn, _fmsub(c, cs, _fmsub(s, ssn, ccs)))
    return cs + cor


def reduce_sincos(x: float) -> tuple:
    """reduce_sincos, fused as compiled: (n, a, da) with x = n pi/2 + a + da, n the low bits of round(x 2/pi)."""
    t = fma(x, HPINV, TOINT)
    xn = t - TOINT
    y = _fmsub(xn, MP2, _fmsub(xn, MP1, x))
    t2 = _fmsub(xn, PP3, y)
    b = _fmsub(xn, PP4, t2)
    db = _fmsub(xn, PP3, y - t2)
    db = db + _fmsub(xn, PP4, t2 - b)
    return bits(t) & 3, b, db


def _branred_half(xh: float) -> tuple:
    """One half of branred: xh 2/pi from six 24-bit pieces, the integer part split off; (sum, b, bb)."""
    e = (bits(xh) >> 52) & 0x7FF
    if e <= BR_EXP_LIMIT:
        k, first = 0, BR_TOVERP0
    else:
        q = abs(e - BR_EXP_BIAS) // 24
        k = q if e >= BR_EXP_BIAS else -q
        first = BR_TOVERP[k]
    gor = dbl(((BR_T576_HI - ((3 * k) << 23)) & 0xFFFFFFFF) << 32)
    r = []
    for i in range(6):
        r.append((xh * (first if i == 0 else BR_TOVERP[k + i])) * gor)
        gor = gor * BR_TM24
    total = 0.0
    for i in range(3):
        s = (r[i] + BR_BIG) - BR_BIG
        total = total + s
        r[i] = r[i] - s
    t = 0.0
    for i in range(6):
        t = t + r[5 - i]
    bb = (((((r[0] - t) + r[1]) + r[2]) + r[3]) + r[4]) + r[5]
    s = (t + BR_BIG) - BR_BIG
    total = s + total
    t = t - s
    s = (total + BR_BIG1) - BR_BIG1
    total = total - s
    b = bb + t
    bb = (t - b) + bb
    return total, b, bb


def branred(x: float) -> tuple:
    """__branred: (n, a, da) with x = n pi/2 + a + da for huge x (the argument scaled by 2^-600 and split in two)."""
    x = x * BR_TM600
    t = x * BR_SPLIT
    x1 = t - (t - x)
    x2 = x - x1
    sum1, b1, bb1 = _branred_half(x1)
    sum2, b2, bb2 = _branred_half(x2)
    total = sum1 + sum2
    b = b1 + b2
    bb = (b1 - b) + b2 if abs(b1) > abs(b2) else (b2 - b) + b1
    if b > 0.5:
        b = b - 1.0
        total = total + 1.0
    elif b < -0.5:
        b = b + 1.0
        total = total - 1.0
    s = ((bb + bb1) + bb2) + b
    t = ((b - s) + bb) + (bb1 + bb2)
    sp = s * BR_SPLIT
    t1 = sp - (sp - s)
    t2 = s - t1
    b = s * BR_HP0
    bb = (((t1 * BR_MP1 - b) + t1 * BR_MP2) + t2 * BR_MP1) + ((t2 * BR_MP2 + s * BR_HP1) + t * BR_HP0)
    s = b + bb
    return int(total) & 3, s, (b - s) + bb


def _sincos_n(a: float, da: float, n: int) -> float:
    """do_sincos: do_cos for odd n, else do_sin; negated when n & 2."""
    v = do_cos(a, da) if n & 1 else do_sin(a, da)
    return -v if n & 2 else v


def sin(x: float) -> float:
    """glibc 2.34 aarch64's sin (__sin, 0x362b0), bit for bit; NaN for infinities and NaN."""
    k = (bits(x) >> 32) & 0x7FFFFFFF
    if k < K_TINY_SIN:
        return x
    if k < K_DO_SIN:
        return do_sin(x, 0.0)
    if k < K_MID:
        return math.copysign(do_cos(HP0 - abs(x), HP1), x)
    if k < K_REDUCE:
        n, a, da = reduce_sincos(x)
        return _sincos_n(a, da, n)
    if k < K_HUGE:
        n, a, da = branred(x)
        return _sincos_n(a, da, n)
    return math.nan


def cos(x: float) -> float:
    """glibc 2.34 aarch64's cos (__cos, 0x369f0), bit for bit; NaN for infinities and NaN."""
    k = (bits(x) >> 32) & 0x7FFFFFFF
    if k < K_TINY_COS:
        return 1.0
    if k < K_DO_SIN:
        return do_cos(x, 0.0)
    if k < K_MID:
        y = HP0 - abs(x)
        a = y + HP1
        return do_sin(a, (y - a) + HP1)
    if k < K_REDUCE:
        n, a, da = reduce_sincos(x)
        return _sincos_n(a, da, n + 1)
    if k < K_HUGE:
        n, a, da = branred(x)
        return _sincos_n(a, da, n + 1)
    return math.nan


# ======================================================================================================= sincos

def sincos(x: float) -> tuple:
    """glibc 2.34 aarch64's sincos (__sincos, 0x37ca0), bit for bit: (sin x, cos x) as Destiny's compiled code gets
    them (g++ -O2 turns a sin and a cos of one argument into one sincos call). Its branches are __sin's and __cos's, at
    the same bounds (its immediates 0x3e3fffff, 0x3feb5fff, 0x400368fc, 0x419921fa and 0x7fefffff are K_TINY_COS,
    K_DO_SIN, K_MID, K_REDUCE and K_HUGE less one), with one difference: between 0.855469 and 2.426265 its sine is
    do_cos(a, da) after the two-sum of hp0 - |x| and hp1, as __cos's cosine there is do_sin(a, da), where __sin calls
    do_cos(hp0 - |x|, hp1). The two meet the same table cell and the same reduced argument except where |hp0 - |x|| is
    within hp1 of a cell's boundary (an odd multiple of 1/256); on every argument within 2^20 patterns of each such
    point glibc's sin and sincos agree but at x = +-0x1.ad1fb54442d18p+0 (|x| = hp0 + 27/256, a tie of the cell's
    rounding), where they differ in the sine's last bit. Below 2^-27 (x, 1);
    infinities and NaN give NaN."""
    k = (bits(x) >> 32) & 0x7FFFFFFF
    if k < K_TINY_COS:
        return x, 1.0
    if k < K_DO_SIN:
        return do_sin(x, 0.0), do_cos(x, 0.0)
    if k < K_MID:
        y = HP0 - abs(x)
        a = y + HP1
        da = (y - a) + HP1
        return math.copysign(do_cos(a, da), x), do_sin(a, da)
    if k < K_REDUCE:
        n, a, da = reduce_sincos(x)
        return _sincos_n(a, da, n), _sincos_n(a, da, n + 1)
    if k < K_HUGE:
        n, a, da = branred(x)
        return _sincos_n(a, da, n), _sincos_n(a, da, n + 1)
    return math.nan, math.nan


# ========================================================================================================== log

def log(x: float) -> float:
    """glibc 2.34 aarch64's log (__log, 0x33610), bit for bit: -inf at zero, NaN below zero and for NaN."""
    ix = bits(x)
    if (ix + LOG_NEG_LO) & M64 <= LOG_RANGE:            # 1 - 2^-4 <= x < 1 + 0x1.09p-4
        if ix == 0x3FF0000000000000:
            return 0.0
        b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10 = LOG_B
        r = x - 1.0
        rhi = fma(r, LOG_TWO27, r)                      # r + w, w = r 2^27
        r2 = r * r
        p78 = fma(b8, r, b7)
        rhi = _fmsub(r, LOG_TWO27, rhi)                 # (r + w) - w
        r3 = r * r2
        p789 = fma(b9, r2, p78)
        p45 = fma(b5, r, b4)
        p710 = fma(b10, r3, p789)
        rhi2 = rhi * rhi
        p456 = fma(b6, r2, p45)
        hi = fma(rhi2, b0, r)                           # r + rhi rhi B0
        p12 = fma(b2, r, b1)
        p410 = fma(p710, r3, p456)
        rlo = r - rhi
        rr = r + rhi
        p123 = fma(b3, r2, p12)
        b0rlo = b0 * rlo
        lo = r - hi
        lo = fma(rhi2, b0, lo)
        p = fma(p410, r3, p123)
        lo = fma(b0rlo, rr, lo)
        y = fma(p, r3, lo)
        return hi + y
    top = ix >> 48
    if (top - 0x0010) & 0xFFFFFFFF >= 0x7FE0:           # below 2^-1022, negative, infinite or NaN
        if (ix << 1) & M64 == 0:
            return -math.inf
        if ix == 0x7FF0000000000000:
            return x
        if (top & 0x8000) or (top & 0x7FF0) == 0x7FF0:
            return math.nan
        ix = (bits(x * LOG_TWO52) + LOG_NEG_52) & M64   # a subnormal x normalised by 2^52
    tmp = (ix + LOG_NEG_OFF) & M64
    i = (tmp >> 45) & 127
    k = (tmp - (1 << 64) if tmp >> 63 else tmp) >> 52
    iz = (ix - (tmp & 0xFFF0000000000000)) & M64
    kd = float(k)
    invc, logc = LOG_TAB[i]
    a0, a1, a2, a3, a4 = LOG_A
    w = fma(LOG_LN2HI, kd, logc)
    r = fma(dbl(iz), invc, -1.0)
    hi = r + w
    p34 = fma(a4, r, a3)
    p12 = fma(a2, r, a1)
    r2 = r * r
    lo = w - hi
    r3 = r * r2
    lo = lo + r
    p = fma(p34, r2, p12)
    lo = fma(LOG_LN2LO, kd, lo)
    lo = fma(a0, r2, lo)
    lo = fma(r3, p, lo)
    return lo + hi
