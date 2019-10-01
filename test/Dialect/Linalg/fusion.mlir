// RUN: mlir-opt %s -linalg-fusion | FileCheck %s

#map0 = (d0) -> (d0 + 2)
#map1 = (d0) -> (d0 + 4)
#map2 = (d0) -> (d0 + 3)

// CHECK-DAG: #[[strided2D:.*]] = (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)
#strided2D = (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)

func @f1(%A: memref<?x?xf32, #strided2D>, %B: memref<?x?xf32, #strided2D>, %C: memref<?x?xf32, #strided2D>, %D: memref<?x?xf32, #strided2D>, %E: memref<?x?xf32, #strided2D>) -> memref<?x?xf32, #strided2D> {
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  %0 = dim %A, 0 : memref<?x?xf32, #strided2D>
  %1 = dim %A, 1 : memref<?x?xf32, #strided2D>
  %2 = dim %B, 1 : memref<?x?xf32, #strided2D>
  linalg.matmul(%A, %B, %C) : memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>
  %c1 = constant 1 : index
  loop.for %arg5 = %c0 to %0 step %c2 {
    loop.for %arg6 = %c0 to %2 step %c3 {
      loop.for %arg7 = %c0 to %1 step %c4 {
        %3 = affine.apply #map0(%arg5)
        %4 = affine.apply #map1(%arg7)
        %5 = linalg.subview %A[%arg5, %3, %c1, %arg7, %4, %c1] : memref<?x?xf32, #strided2D>
        %6 = affine.apply #map2(%arg6)
        %7 = linalg.subview %B[%arg7, %4, %c1, %arg6, %6, %c1] : memref<?x?xf32, #strided2D>
        %8 = linalg.subview %C[%arg5, %3, %c1, %arg6, %6, %c1] : memref<?x?xf32, #strided2D>
        linalg.matmul(%5, %7, %8) : memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>
      }
    }
  }
  return %E : memref<?x?xf32, #strided2D>
}
// CHECK-LABEL: func @f1
//       CHECK:   (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
// No RAW dependences, the pass does not fuse RAR atm.
//      CHECK: linalg.matmul
//      CHECK: loop.for
//      CHECK:   loop.for
//      CHECK:     loop.for
//      CHECK:       linalg.matmul

func @f2(%A: memref<?x?xf32, #strided2D>, %B: memref<?x?xf32, #strided2D>, %C: memref<?x?xf32, #strided2D>, %D: memref<?x?xf32, #strided2D>, %E: memref<?x?xf32, #strided2D>) -> memref<?x?xf32, #strided2D> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  linalg.matmul(%A, %B, %C) : memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>
  %0 = dim %C, 0 : memref<?x?xf32, #strided2D>
  %1 = dim %C, 1 : memref<?x?xf32, #strided2D>
  %2 = dim %D, 1 : memref<?x?xf32, #strided2D>
  loop.for %arg5 = %c0 to %0 step %c2 {
    loop.for %arg6 = %c0 to %2 step %c3 {
      loop.for %arg7 = %c0 to %1 step %c4 {
        %3 = affine.apply #map0(%arg5)
        %4 = affine.apply #map1(%arg7)
        %5 = linalg.subview %C[%arg5, %3, %c1, %arg7, %4, %c1] : memref<?x?xf32, #strided2D>
        %6 = affine.apply #map2(%arg6)
        %7 = linalg.subview %D[%arg7, %4, %c1, %arg6, %6, %c1] : memref<?x?xf32, #strided2D>
        %8 = linalg.subview %E[%arg5, %3, %c1, %arg6, %6, %c1] : memref<?x?xf32, #strided2D>
        linalg.matmul(%5, %7, %8) : memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>
      }
    }
  }
  return %E : memref<?x?xf32, #strided2D>
}
// CHECK-LABEL: func @f2
//       CHECK:   (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
//   CHECK-DAG:   %[[C_0:.*]] = dim %[[C]], 0 : memref<?x?xf32, #[[strided2D]]>
//   CHECK-DAG:   %[[C_1:.*]] = dim %[[C]], 1 : memref<?x?xf32, #[[strided2D]]>
//   CHECK-DAG:   %[[D_1:.*]] = dim %[[D]], 1 : memref<?x?xf32, #[[strided2D]]>
//       CHECK:   loop.for %{{.*}} = %{{.*}} to %[[C_0]] step %{{.*}} {
//       CHECK:     loop.for %{{.*}} = %{{.*}} to %[[D_1]] step %{{.*}} {
//       CHECK:       loop.for %{{.*}} = %{{.*}} to %[[C_1]] step %{{.*}} {
//       CHECK:         linalg.matmul
//       CHECK:         linalg.matmul

func @f3(%A: memref<?x?xf32, #strided2D>, %B: memref<?x?xf32, #strided2D>, %C: memref<?x?xf32, #strided2D>, %D: memref<?x?xf32, #strided2D>, %E: memref<?x?xf32, #strided2D>) -> memref<?x?xf32, #strided2D> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  linalg.matmul(%A, %B, %C) : memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>
  %0 = dim %D, 0 : memref<?x?xf32, #strided2D>
  %1 = dim %D, 1 : memref<?x?xf32, #strided2D>
  %2 = dim %C, 1 : memref<?x?xf32, #strided2D>
  loop.for %arg5 = %c0 to %0 step %c2 {
    loop.for %arg6 = %c0 to %2 step %c3 {
      loop.for %arg7 = %c0 to %1 step %c4 {
        %3 = affine.apply #map0(%arg5)
        %4 = affine.apply #map1(%arg7)
        %5 = linalg.subview %D[%arg5, %3, %c1, %arg7, %4, %c1] : memref<?x?xf32, #strided2D>
        %6 = affine.apply #map2(%arg6)
        %7 = linalg.subview %C[%arg7, %4, %c1, %arg6, %6, %c1] : memref<?x?xf32, #strided2D>
        %8 = linalg.subview %E[%arg5, %3, %c1, %arg6, %6, %c1] : memref<?x?xf32, #strided2D>
        linalg.matmul(%5, %7, %8) : memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>
      }
    }
  }
  return %E : memref<?x?xf32, #strided2D>
}
// CHECK-LABEL: func @f3
//       CHECK:   (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
//          CHECK:   %[[D_0:.*]] = dim %[[D]], 0 : memref<?x?xf32, #[[strided2D]]>
//          CHECK:   %[[D_1:.*]] = dim %[[D]], 1 : memref<?x?xf32, #[[strided2D]]>
//          CHECK:   %[[C_1:.*]] = dim %[[C]], 1 : memref<?x?xf32, #[[strided2D]]>
//          CHECK:   loop.for %{{.*}} = %{{.*}} to %[[D_0]] step %{{.*}} {
//          CHECK:     loop.for %{{.*}} = %{{.*}} to %[[C_1]] step %{{.*}} {
//          CHECK:       loop.for %{{.*}} = %{{.*}} to %[[D_1]] step %{{.*}} {
//          CHECK:         linalg.matmul
//          CHECK:         linalg.matmul

func @f4(%A: memref<?x?xf32, #strided2D>, %B: memref<?x?xf32, #strided2D>, %C: memref<?x?xf32, #strided2D>, %D: memref<?x?xf32, #strided2D>, %E: memref<?x?xf32, #strided2D>) -> memref<?x?xf32, #strided2D> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  linalg.matmul(%A, %B, %C) : memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>
  linalg.matmul(%A, %B, %D) : memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>
  %0 = dim %C, 0 : memref<?x?xf32, #strided2D>
  %1 = dim %C, 1 : memref<?x?xf32, #strided2D>
  %2 = dim %D, 1 : memref<?x?xf32, #strided2D>
  loop.for %arg5 = %c0 to %0 step %c2 {
    loop.for %arg6 = %c0 to %2 step %c3 {
      loop.for %arg7 = %c0 to %1 step %c4 {
        %3 = affine.apply #map0(%arg5)
        %4 = affine.apply #map1(%arg7)
        %5 = linalg.subview %C[%arg5, %3, %c1, %arg7, %4, %c1] : memref<?x?xf32, #strided2D>
        %6 = affine.apply #map2(%arg6)
        %7 = linalg.subview %D[%arg7, %4, %c1, %arg6, %6, %c1] : memref<?x?xf32, #strided2D>
        %8 = linalg.subview %E[%arg5, %3, %c1, %arg6, %6, %c1] : memref<?x?xf32, #strided2D>
        linalg.matmul(%5, %7, %8) : memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>
      }
    }
  }
  return %E : memref<?x?xf32, #strided2D>
}
// CHECK-LABEL: func @f4
//       CHECK:   (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
//          CHECK:   %[[C_0:.*]] = dim %[[C]], 0 : memref<?x?xf32, #[[strided2D]]>
//          CHECK:   %[[C_1:.*]] = dim %[[C]], 1 : memref<?x?xf32, #[[strided2D]]>
//          CHECK:   %[[D_1:.*]] = dim %[[D]], 1 : memref<?x?xf32, #[[strided2D]]>
//          CHECK:   loop.for %{{.*}} = %{{.*}} to %[[C_0]] step %{{.*}} {
//          CHECK:     loop.for %{{.*}} = %{{.*}} to %[[D_1]] step %{{.*}} {
//          CHECK:       loop.for %{{.*}} = %{{.*}} to %[[C_1]] step %{{.*}} {
// Fuse D then fuse C, no false dependence prevent it.
//          CHECK:         linalg.matmul
//          CHECK:         linalg.matmul
//          CHECK:         linalg.matmul

func @f5(%A: memref<?x?xf32, #strided2D>, %B: memref<?x?xf32, #strided2D>, %C: memref<?x?xf32, #strided2D>, %D: memref<?x?xf32, #strided2D>, %E: memref<?x?xf32, #strided2D>) -> memref<?x?xf32, #strided2D> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  %0 = dim %B, 1 : memref<?x?xf32, #strided2D>
  %1 = dim %D, 0 : memref<?x?xf32, #strided2D>
  %2 = dim %D, 1 : memref<?x?xf32, #strided2D>
  linalg.matmul(%A, %B, %C) : memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>
  linalg.matmul(%C, %B, %D) : memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>
  loop.for %arg5 = %c0 to %1 step %c2 {
    loop.for %arg6 = %c0 to %0 step %c3 {
      loop.for %arg7 = %c0 to %2 step %c4 {
        %3 = affine.apply #map0(%arg5)
        %4 = affine.apply #map1(%arg7)
        %5 = linalg.subview %D[%arg5, %3, %c1, %arg7, %4, %c1] : memref<?x?xf32, #strided2D>
        %6 = affine.apply #map2(%arg6)
        %7 = linalg.subview %B[%arg7, %4, %c1, %arg6, %6, %c1] : memref<?x?xf32, #strided2D>
        %8 = linalg.subview %E[%arg5, %3, %c1, %arg6, %6, %c1] : memref<?x?xf32, #strided2D>
        linalg.matmul(%5, %7, %8) : memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>
      }
    }
  }
  return %E : memref<?x?xf32, #strided2D>
}
// CHECK-LABEL: func @f5
//       CHECK:   (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
//      CHECK-DAG:   %[[B_1:.*]] = dim %[[B]], 1 : memref<?x?xf32, #[[strided2D]]>
//      CHECK-DAG:   %[[D_0:.*]] = dim %[[D]], 0 : memref<?x?xf32, #[[strided2D]]>
//      CHECK-DAG:   %[[D_1:.*]] = dim %[[D]], 1 : memref<?x?xf32, #[[strided2D]]>
// Don't fuse C due to false dependence, note that this is too conservative though.
//          CHECK:   linalg.matmul(%{{.*}}, %{{.*}}, %{{.*}})
//          CHECK:   loop.for %{{.*}} = %{{.*}} to %[[D_0]] step %{{.*}} {
//          CHECK:     loop.for %{{.*}} = %{{.*}} to %[[B_1]] step %{{.*}} {
//          CHECK:       loop.for %{{.*}} = %{{.*}} to %[[D_1]] step %{{.*}} {
//          CHECK:         linalg.matmul
//          CHECK:         linalg.matmul

func @f6(%A: memref<?x?xf32, #strided2D>, %B: memref<?x?xf32, #strided2D>, %C: memref<?x?xf32, #strided2D>, %D: memref<?x?xf32, #strided2D>, %E: memref<?x?xf32, #strided2D>) -> memref<?x?xf32, #strided2D> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  %0 = dim %C, 1 : memref<?x?xf32, #strided2D>
  linalg.matmul(%A, %B, %C) : memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>
  linalg.matmul(%A, %C, %E) : memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>
  %1 = dim %C, 0 : memref<?x?xf32, #strided2D>
  %2 = dim %D, 1 : memref<?x?xf32, #strided2D>
  loop.for %arg5 = %c0 to %1 step %c2 {
    loop.for %arg6 = %c0 to %2 step %c3 {
      loop.for %arg7 = %c0 to %0 step %c4 {
        %3 = affine.apply #map0(%arg5)
        %4 = affine.apply #map1(%arg7)
        %5 = linalg.subview %C[%arg5, %3, %c1, %arg7, %4, %c1] : memref<?x?xf32, #strided2D>
        %6 = affine.apply #map2(%arg6)
        %7 = linalg.subview %D[%arg7, %4, %c1, %arg6, %6, %c1] : memref<?x?xf32, #strided2D>
        %8 = linalg.subview %E[%arg5, %3, %c1, %arg6, %6, %c1] : memref<?x?xf32, #strided2D>
        linalg.matmul(%5, %7, %8) : memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>
      }
    }
  }
  return %E : memref<?x?xf32, #strided2D>
}
// CHECK-LABEL: func @f6
//       CHECK:   (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
// Cannot fuse C due to interleaved read of C that would be bypassed.
// Cannot fuse E (WAW).
//   CHECK:   linalg.matmul
//   CHECK:   linalg.matmul
//   CHECK:   loop.for
//   CHECK:     loop.for
//   CHECK:       loop.for
//   CHECK:         linalg.matmul
// CHECK-NOT:       linalg.matmul

func @f7(%A: memref<?x?xf32, #strided2D>, %B: memref<?x?xf32, #strided2D>, %C: memref<?x?xf32, #strided2D>, %D: memref<?x?xf32, #strided2D>, %E: memref<?x?xf32, #strided2D>) -> memref<?x?xf32, #strided2D> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  %0 = dim %A, 0 : memref<?x?xf32, #strided2D>
  %1 = dim %A, 1 : memref<?x?xf32, #strided2D>
  %2 = dim %C, 1 : memref<?x?xf32, #strided2D>
  %3 = dim %C, 0 : memref<?x?xf32, #strided2D>
  %4 = dim %D, 1 : memref<?x?xf32, #strided2D>
  linalg.matmul(%A, %C, %E) : memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>
  linalg.matmul(%A, %B, %C) : memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>
  loop.for %arg5 = %c0 to %0 step %c2 {
    loop.for %arg6 = %c0 to %2 step %c3 {
      loop.for %arg7 = %c0 to %1 step %c4 {
        %5 = affine.apply #map0(%arg5)
        %6 = affine.apply #map1(%arg7)
        %7 = linalg.subview %A[%arg5, %5, %c1, %arg7, %6, %c1] : memref<?x?xf32, #strided2D>
        %8 = affine.apply #map2(%arg6)
        %9 = linalg.subview %C[%arg7, %6, %c1, %arg6, %8, %c1] : memref<?x?xf32, #strided2D>
        %10 = linalg.subview %E[%arg5, %5, %c1, %arg6, %8, %c1] : memref<?x?xf32, #strided2D>
        linalg.matmul(%7, %9, %10) : memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>
      }
    }
  }
  loop.for %arg5 = %c0 to %3 step %c2 {
    loop.for %arg6 = %c0 to %4 step %c3 {
      loop.for %arg7 = %c0 to %2 step %c4 {
        %5 = affine.apply #map0(%arg5)
        %6 = affine.apply #map1(%arg7)
        %7 = linalg.subview %C[%arg5, %5, %c1, %arg7, %6, %c1] : memref<?x?xf32, #strided2D>
        %8 = affine.apply #map2(%arg6)
        %9 = linalg.subview %D[%arg7, %6, %c1, %arg6, %8, %c1] : memref<?x?xf32, #strided2D>
        %10 = linalg.subview %E[%arg5, %5, %c1, %arg6, %8, %c1] : memref<?x?xf32, #strided2D>
        linalg.matmul(%7, %9, %10) : memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>
      }
    }
  }
  return %E : memref<?x?xf32, #strided2D>
}
// CHECK-LABEL: func @f7
//       CHECK:   (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
//       CHECK:   %[[A_0:.*]] = dim %[[A]], 0 : memref<?x?xf32, #[[strided2D]]>
//       CHECK:   %[[A_1:.*]] = dim %[[A]], 1 : memref<?x?xf32, #[[strided2D]]>
//       CHECK:   %[[C_1:.*]] = dim %[[C]], 1 : memref<?x?xf32, #[[strided2D]]>
//       CHECK:   %[[C_0:.*]] = dim %[[C]], 0 : memref<?x?xf32, #[[strided2D]]>
//       CHECK:   %[[D_1:.*]] = dim %[[D]], 1 : memref<?x?xf32, #[[strided2D]]>
//       CHECK:   linalg.matmul(%[[A]], %[[C]], %[[E]])
//       CHECK:   loop.for %{{.*}} = %{{.*}} to %[[A_0]] step %{{.*}} {
//       CHECK:     loop.for %{{.*}} = %{{.*}} to %[[C_1]] step %{{.*}} {
//       CHECK:       loop.for %{{.*}} = %{{.*}} to %[[A_1]] step %{{.*}} {
//       CHECK:         linalg.matmul
//       CHECK:         linalg.matmul
//       CHECK:   loop.for %{{.*}} = %{{.*}} to %[[C_0]] step %{{.*}} {
//       CHECK:     loop.for %{{.*}} = %{{.*}} to %[[D_1]] step %{{.*}} {
//       CHECK:       loop.for %{{.*}} = %{{.*}} to %[[C_1]] step %{{.*}} {
//       CHECK:         linalg.matmul
//   CHECK-NOT:         linalg.matmul

func @f8(%A: memref<?x?xf32, #strided2D>, %B: memref<?x?xf32, #strided2D>, %C: memref<?x?xf32, #strided2D>, %D: memref<?x?xf32, #strided2D>, %E: memref<?x?xf32, #strided2D>) -> memref<?x?xf32, #strided2D> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  %0 = dim %A, 0 : memref<?x?xf32, #strided2D>
  %1 = dim %A, 1 : memref<?x?xf32, #strided2D>
  linalg.matmul(%A, %C, %D) : memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>
  linalg.matmul(%A, %B, %C) : memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>
  %2 = dim %D, 1 : memref<?x?xf32, #strided2D>
  loop.for %arg5 = %c0 to %0 step %c2 {
    loop.for %arg6 = %c0 to %2 step %c3 {
      loop.for %arg7 = %c0 to %1 step %c4 {
        %3 = affine.apply #map0(%arg5)
        %4 = affine.apply #map1(%arg7)
        %5 = linalg.subview %A[%arg5, %3, %c1, %arg7, %4, %c1] : memref<?x?xf32, #strided2D>
        %6 = affine.apply #map2(%arg6)
        %7 = linalg.subview %D[%arg7, %4, %c1, %arg6, %6, %c1] : memref<?x?xf32, #strided2D>
        %8 = linalg.subview %E[%arg5, %3, %c1, %arg6, %6, %c1] : memref<?x?xf32, #strided2D>
        linalg.matmul(%5, %7, %8) : memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>
      }
    }
  }
  return %E : memref<?x?xf32, #strided2D>
}
// CHECK-LABEL: func @f8
//       CHECK:   (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
//   CHECK:   linalg.matmul
//   CHECK:   linalg.matmul
//   CHECK:   loop.for
//   CHECK:     loop.for
//   CHECK:       loop.for
//   CHECK:         linalg.matmul
// CHECK-NOT:       linalg.matmul

#id_2d = (i, j) -> (i, j)
#pointwise_2d_trait = {
  indexing_maps = [#id_2d, #id_2d, #id_2d],
  n_loop_types = [2, 0, 0],
  n_views = [2, 1]
}
func @pointwise(%A: memref<?x?xf32, #strided2D>, %B: memref<?x?xf32, #strided2D>, %C: memref<?x?xf32, #strided2D>, %D: memref<?x?xf32, #strided2D>) {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  linalg.generic #pointwise_2d_trait %A, %A, %B {
  ^bb0(%E: f32, %arg5: f32, %arg6: f32):   // no predecessors
    %2 = addf %E, %arg5 : f32
    linalg.yield %2 : f32
  }: memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>
  %0 = dim %B, 0 : memref<?x?xf32, #strided2D>
  %1 = dim %B, 1 : memref<?x?xf32, #strided2D>
  loop.for %E = %c0 to %0 step %c2 {
    loop.for %arg5 = %c0 to %1 step %c3 {
      %2 = affine.apply #map0(%E)
      %3 = affine.apply #map1(%arg5)
      %4 = linalg.subview %B[%E, %2, %c1, %arg5, %3, %c1] : memref<?x?xf32, #strided2D>
      %5 = linalg.subview %C[%E, %2, %c1, %arg5, %3, %c1] : memref<?x?xf32, #strided2D>
      %6 = linalg.subview %D[%E, %2, %c1, %arg5, %3, %c1] : memref<?x?xf32, #strided2D>
      linalg.generic #pointwise_2d_trait %4, %5, %6 {
      ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):       // no predecessors
        %7 = mulf %arg6, %arg7 : f32
        linalg.yield %7 : f32
      }: memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>
    }
  }
  return
}
// CHECK-LABEL: func @pointwise
//       CHECK:   loop.for
//       CHECK:     loop.for
//   CHECK-NOT:   loop.for
//       CHECK:       linalg.generic
//       CHECK:         addf
//       CHECK:       linalg.generic
//       CHECK:         mulf
