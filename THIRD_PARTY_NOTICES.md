# Third-party notices

exact-ballpark is licensed under the MIT licence (`LICENSE`), except the files marked
`SPDX-License-Identifier: LGPL-2.1-or-later`, which are licensed under the GNU Lesser General Public License, version 2.1
or later (`LICENSES/LGPL-2.1-or-later.txt`). It contains work derived from the projects below. Their notices follow.

## destiny (CCP Games), MIT

The scalar reference, the batched parks and the CUDA kernels port the behaviour of `destiny`
(https://github.com/carbonengine/destiny), checked against commit `114e89fa584f0e1a81eb0a679aabf41af8870072`.

```text
MIT License

Copyright (c) 2026 CCP Games

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## carbonengine/math (CCP Games), MIT

The vector, matrix and quaternion operations destiny calls are ported from `carbonengine/math`
(https://github.com/carbonengine/math), commit `8273322093eb1f80168ae60a35891f2d454369f9`.

```text
MIT License

Copyright (c) 2026 CCP Games

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## DirectXMath (Microsoft), MIT

The quaternion from yaw, pitch and roll (`XMQuaternionRotationRollPitchYaw`'s scalar path, which carbonengine/math's
`RotationQuaternion` calls) is ported from DirectXMath (https://github.com/microsoft/DirectXMath), commit `f5d82574b03417d905fdd2f85bfa9bf7a2f1deac`.

```text
MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE
```

## The GNU C Library's maths, LGPL-2.1-or-later

destiny's results depend on the C library's elementary functions. The files that transcribe them, operation for
operation, from the GNU C Library 2.34 are licensed under the GNU Lesser General Public License, version 2.1 or (at your
option) any later version, as the GNU C Library is. The full text is in `LICENSES/LGPL-2.1-or-later.txt`, and each such
file says so in its header. The functions come from:
- **the IBM Accurate Mathematical Library**, written by International Business Machines Corp. and part of the GNU C
  Library, copyright the Free Software Foundation, Inc.: the double `sin` and `cos` with their argument reduction, and
  the double `asin` and `atan2`;
- **Arm's optimized-routines**, contributed to the GNU C Library by Arm Ltd: `exp`, `log`, `pow`, `expf`, `logf`,
  `sinf`, `cosf` and `powf`;
- **Sun's fdlibm**, as carried in the GNU C Library: `acosf`, `atanf` and `atan2f`, under this notice:

```text
Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.

Developed at SunPro, a Sun Microsystems, Inc. business.
Permission to use, copy, modify, and distribute this
software is freely granted, provided that this notice
is preserved.
```

The port also reproduces the C library's `rand()`: the GNU C Library's additive feedback generator (`random_r`,
TYPE_3), the source of destiny's `floatrand` for shuffled sensors. It falls under the same LGPL-2.1-or-later rule. The GNU C Library
distributes the code it incorporates from 4.4 BSD under this notice:

```text
Copyright (C) 1991 Regents of the University of California.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. [This condition was removed.]
4. Neither the name of the University nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.
```

## Trademarks

CCP Games is a trademark of CCP ehf. exact-ballpark is not affiliated with or endorsed by CCP Games, and nothing in this
project grants any rights to CCP Games' trademarks or game content.
