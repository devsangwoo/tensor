; RUN: %S/../../mlir-opt %s -o - | FileCheck %s

; CHECK: #{{[0-9]+}} = (d0, d1) -> (d0, d1)
#hello_world0 = (i, j) -> (i, j)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> (d0, d1)
#hello_world1 = (i, j) [s0] -> (i, j)

; CHECK: #{{[0-9]+}} = () -> (0)
#hello_world2 = () -> (0)

; CHECK: #{{[0-9]+}} = (d0, d1) -> ((d0 + 1), d1)
#hello_world3 = (i, j) -> (i+1, j)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> ((d0 + s0), d1)
#hello_world4 = (i, j) [s0] -> (i + s0, j)

; CHECK: #{{[0-9]+}} = (d0, d1) -> ((1 + d0), d1)
#hello_world5 = (i, j) -> (1+i, j)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> ((d0 + s0), (d1 + 5))
#hello_world6 = (i, j) [s0] -> (i + s0, j + 5)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> (((d0 + d1) + s0), d1)
#hello_world7 = (i, j) [s0] -> (i + j + s0, j)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> ((((5 + d0) + d1) + s0), d1)
#hello_world8 = (i, j) [s0] -> (5 + i + j + s0, j)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> (((d0 + d1) + 5), d1)
#hello_world9 = (i, j) [s0] -> ((i + j) + 5, j)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> ((d0 + (d1 + 5)), d1)
#hello_world10 = (i, j) [s0] -> (i + (j + 5), j)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> ((2 * d0), (3 * d1))
#hello_world11 = (i, j) [s0] -> (2*i, 3*j)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> (((d0 + (2 * 6)) + (5 * (d1 + (s0 * 3)))), d1)
#hello_world12 = (i, j) [s0] -> (i + 2*6 + 5*(j+s0*3), j)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> (((5 * d0) + d1), d1)
#hello_world13 = (i, j) [s0] -> (5*i + j, j)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> ((d0 + d1), d1)
#hello_world14 = (i, j) [s0] -> ((i + j), (j))

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> (((d0 + d1) + 5), (d1 + 3))
#hello_world15 = (i, j) [s0] -> ((i + j)+5, (j)+3)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> (d0, 0)
#hello_world16 = (i, j) [s1] -> (i, 0)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> (d0, (s0 * d1))
#hello_world17 = (i, j) [s0] -> (i, s0*j)

; CHECK: #{{[0-9]+}} = (d0, d1) -> (d0, ((3 * d0) + d1))
#hello_world19 = (i, j) -> (i, 3*i + j)

; CHECK: #{{[0-9]+}} = (d0, d1) -> (d0, (d0 + (3 * d1)))
#hello_world20 = (i, j)  -> (i, i + 3*j)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> (d0, ((2 + (((s0 * s0) * 9) * d0)) + 1))
#hello_world18 = (i, j) [N] -> (i, 2 + N*N*9*i + 1)

; CHECK: #{{[0-9]+}} = (d0, d1) -> (1, ((d0 + (3 * d1)) + 5))
#hello_world21 = (i, j)  -> (1, i + 3*j + 5)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> ((5 * s0), ((d0 + (3 * d1)) + (5 * d0)))
#hello_world22 = (i, j) [s0] -> (5*s0, i + 3*j + 5*i)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0, s1] -> ((d0 * (s0 * s1)), d1)
#hello_world23 = (i, j) [s0, s1] -> (i*(s0*s1), j)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0, s1] -> (d0, (d1 mod 5))
#hello_world24 = (i, j) [s0, s1] -> (i, j mod 5)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0, s1] -> (d0, (d1 floordiv 5))
#hello_world25 = (i, j) [s0, s1] -> (i, j floordiv 5)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0, s1] -> (d0, (d1 ceildiv 5))
#hello_world26 = (i, j) [s0, s1] -> (i, j ceildiv 5)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0, s1] -> (d0, ((d0 - d1) - 5))
#hello_world29 = (i, j) [s0, s1] -> (i, i - j - 5)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0, s1] -> (d0, ((d0 - (s1 * d1)) + 2))
#hello_world30 = (i, j) [M, N] -> (i, i - N*j + 2)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0, s1] -> (((-1 * 5) * d0), ((-1 * 3) * d1), (-1 * 2), ((-1 * 1) * (d0 + d1)), ((-1 * 1) * s0))
#hello_world32 = (i, j) [s0, s1] -> (-5*i, -3*j, -2, -1*(i+j), -1*s0)

; CHECK: #{{[0-9]+}} = (d0, d1) -> ((((-1 * 2) + (-1 * 5)) - (-1 * 3)), ((-1 * 1) * d0))
#hello_world33 = (i, j) -> (-2+-5-(-3), -1*i)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0, s1] -> (d0, (d1 floordiv s0), (d1 mod s0))
#hello_world34 = (i, j) [s0, s1] -> (i, j floordiv s0, j mod s0)

; CHECK: #{{[0-9]+}} = (d0, d1, d2) [s0, s1, s2] -> (((((d0 * s1) * s2) + (d1 * s1)) + d2))
#hello_world35 = (i, j, k) [s0, s1, s2] -> (i*s1*s2 + j*s1 + k)

; CHECK: #{{[0-9]+}} = (d0, d1) -> (2, 8)
#hello_world36 = (i, j) -> (1+1, 5+3)
