// RUN: mlir-opt %s -loop-fusion -split-input-file -verify | FileCheck %s
// RUN: mlir-opt %s -loop-fusion -fusion-src-loop-depth=1 -fusion-dst-loop-depth=1 -split-input-file -verify | FileCheck %s  --check-prefix DEPTH1

// TODO(andydavis) Add more tests:
// *) Add nested fusion test cases when non-constant loop bound support is
//    added to iteration domain in dependence check.
// *) Add a test w/ floordiv/ceildiv/mod when supported in dependence check.
// *) Add tests which check fused computation slice indexing and loop bounds.
// TODO(andydavis) Test clean up: move memref allocs to func args.

// -----

// CHECK: [[MAP0:#map[0-9]+]] = (d0) -> (d0)

// CHECK-LABEL: func @should_fuse_raw_dep_for_locality() {
func @should_fuse_raw_dep_for_locality() {
  %m = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32

  for %i0 = 0 to 10 {
    store %cf7, %m[%i0] : memref<10xf32>
  }
  for %i1 = 0 to 10 {
    %v0 = load %m[%i1] : memref<10xf32>
  }
  // CHECK:      for %i0 = 0 to 10 {
  // CHECK-NEXT:   %1 = affine_apply [[MAP0]](%i0)
  // CHECK-NEXT:   store %cst, %0[%1] : memref<10xf32>
  // CHECK-NEXT:   %2 = load %0[%i0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = (d0) -> (d0)

// CHECK-LABEL: func @should_fuse_reduction_to_pointwise() {
func @should_fuse_reduction_to_pointwise() {
  %a = alloc() : memref<10x10xf32>
  %b = alloc() : memref<10xf32>
  %c = alloc() : memref<10xf32>

  %cf7 = constant 7.0 : f32

  for %i0 = 0 to 10 {
    for %i1 = 0 to 10 {
      %v0 = load %b[%i0] : memref<10xf32>
      %v1 = load %a[%i0, %i1] : memref<10x10xf32>
      %v3 = addf %v0, %v1 : f32
      store %v3, %b[%i0] : memref<10xf32>
    }
  }
  for %i2 = 0 to 10 {
    %v4 = load %b[%i2] : memref<10xf32>
    store %v4, %c[%i2] : memref<10xf32>
  }

  // Should fuse in entire inner loop on %i1 from source loop nest, as %i1
  // is not used in the access function of the store/load on %b.
  // CHECK:       for %i0 = 0 to 10 {
  // CHECK-NEXT:    %3 = affine_apply [[MAP0]](%i0)
  // CHECK-NEXT:    for %i1 = 0 to 10 {
  // CHECK-NEXT:      %4 = load %1[%3] : memref<10xf32>
  // CHECK-NEXT:      %5 = load %0[%3, %i1] : memref<10x10xf32>
  // CHECK-NEXT:      %6 = addf %4, %5 : f32
  // CHECK-NEXT:      store %6, %1[%3] : memref<10xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:    %7 = load %1[%i0] : memref<10xf32>
  // CHECK-NEXT:    store %7, %2[%i0] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK: [[MAP_SHIFT_MINUS_ONE_D0:#map[0-9]+]] = (d0, d1) -> (d0 - 1)
// CHECK: [[MAP_SHIFT_MINUS_ONE_D1:#map[0-9]+]] = (d0, d1) -> (d1 - 1)
// CHECK: [[MAP_SHIFT_BY_ONE:#map[0-9]+]] = (d0, d1) -> (d0 + 1, d1 + 1)

// CHECK-LABEL: func @should_fuse_loop_nests_with_shifts() {
func @should_fuse_loop_nests_with_shifts() {
  %a = alloc() : memref<10x10xf32>
  %cf7 = constant 7.0 : f32

  for %i0 = 0 to 10 {
    for %i1 = 0 to 10 {
      %a0 = affine_apply (d0, d1) -> (d0 + 1, d1 + 1) (%i0, %i1)
      store %cf7, %a[%a0#0, %a0#1] : memref<10x10xf32>
    }
  }
  for %i2 = 0 to 10 {
    for %i3 = 0 to 10 {
      %v0 = load %a[%i2, %i3] : memref<10x10xf32>
    }
  }

  // CHECK:      for %i0 = 0 to 10 {
  // CHECK-NEXT:   for %i1 = 0 to 10 {
  // CHECK-NEXT:     %1 = affine_apply [[MAP_SHIFT_MINUS_ONE_D0]](%i0, %i1)
  // CHECK-NEXT:     %2 = affine_apply [[MAP_SHIFT_MINUS_ONE_D1]](%i0, %i1)
  // CHECK-NEXT:     %3 = affine_apply [[MAP_SHIFT_BY_ONE]](%1, %2)
  // CHECK-NEXT:     store %cst, %0[%3#0, %3#1] : memref<10x10xf32>
  // CHECK-NEXT:     %4 = load %0[%i0, %i1] : memref<10x10xf32>
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-DAG: [[MAP_DIM0:#map[0-9]+]] = (d0, d1) -> (d0)
// CHECK-DAG: [[MAP_DIM1:#map[0-9]+]] = (d0, d1) -> (d1)

// CHECK-LABEL: func @should_fuse_loop_nest() {
func @should_fuse_loop_nest() {
  %a = alloc() : memref<10x10xf32>
  %b = alloc() : memref<10x10xf32>
  %cf7 = constant 7.0 : f32

  for %i0 = 0 to 10 {
    for %i1 = 0 to 10 {
      store %cf7, %a[%i0, %i1] : memref<10x10xf32>
    }
  }
  for %i2 = 0 to 10 {
    for %i3 = 0 to 10 {
      %v0 = load %a[%i3, %i2] : memref<10x10xf32>
      store %v0, %b[%i2, %i3] : memref<10x10xf32>
    }
  }
  for %i4 = 0 to 10 {
    for %i5 = 0 to 10 {
      %v1 = load %b[%i4, %i5] : memref<10x10xf32>
    }
  }

  // CHECK:      for %i0 = 0 to 10 {
  // CHECK-NEXT:   for %i1 = 0 to 10 {
  // CHECK-NEXT:     %2 = affine_apply [[MAP_DIM1]](%i0, %i1)
  // CHECK-NEXT:     %3 = affine_apply [[MAP_DIM0]](%i0, %i1)
  // CHECK-NEXT:     store %cst, %0[%2, %3] : memref<10x10xf32>
  // CHECK-NEXT:     %4 = affine_apply [[MAP_DIM0]](%i0, %i1)
  // CHECK-NEXT:     %5 = affine_apply [[MAP_DIM1]](%i0, %i1)
  // CHECK-NEXT:     %6 = load %0[%5, %4] : memref<10x10xf32>
  // CHECK-NEXT:     store %6, %1[%4, %5] : memref<10x10xf32>
  // CHECK-NEXT:     %7 = load %1[%i0, %i1] : memref<10x10xf32>
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = (d0) -> (d0)

// CHECK-LABEL: func @should_fuse_across_intermediate_loop_with_no_deps() {
func @should_fuse_across_intermediate_loop_with_no_deps() {
  %a = alloc() : memref<10xf32>
  %b = alloc() : memref<10xf32>
  %c = alloc() : memref<10xf32>

  %cf7 = constant 7.0 : f32

  for %i0 = 0 to 10 {
    %v0 = load %a[%i0] : memref<10xf32>
    store %v0, %b[%i0] : memref<10xf32>
  }
  for %i1 = 0 to 10 {
    store %cf7, %c[%i1] : memref<10xf32>
  }
  for %i2 = 0 to 10 {
    %v1 = load %b[%i2] : memref<10xf32>
  }

  // Should fuse first loop (past second loop with no dependences) into third.
  // CHECK:      for %i0 = 0 to 10 {
  // CHECK-NEXT:   store %cst, %2[%i0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      for %i1 = 0 to 10 {
  // CHECK-NEXT:   %3 = affine_apply [[MAP0]](%i1)
  // CHECK-NEXT:   %4 = load %0[%3] : memref<10xf32>
  // CHECK-NEXT:   store %4, %1[%3] : memref<10xf32>
  // CHECK-NEXT:   %5 = load %1[%i1] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = (d0) -> (d0)

// CHECK-LABEL: func @should_fuse_all_loops() {
func @should_fuse_all_loops() {
  %a = alloc() : memref<10xf32>
  %b = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32

  // Set up flow dependences from first and second loops to third.
  for %i0 = 0 to 10 {
    store %cf7, %a[%i0] : memref<10xf32>
  }
  for %i1 = 0 to 10 {
    store %cf7, %b[%i1] : memref<10xf32>
  }
  for %i2 = 0 to 10 {
    %v0 = load %a[%i2] : memref<10xf32>
    %v1 = load %b[%i2] : memref<10xf32>
  }

  // Should fuse first and second loops into third.
  // CHECK:      for %i0 = 0 to 10 {
  // CHECK-NEXT:   %2 = affine_apply [[MAP0]](%i0)
  // CHECK-NEXT:   store %cst, %0[%2] : memref<10xf32>
  // CHECK-NEXT:   %3 = affine_apply [[MAP0]](%i0)
  // CHECK-NEXT:   store %cst, %1[%3] : memref<10xf32>
  // CHECK-NEXT:   %4 = load %0[%i0] : memref<10xf32>
  // CHECK-NEXT:   %5 = load %1[%i0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = (d0) -> (d0)

// CHECK-LABEL: func @should_fuse_first_and_second_loops() {
func @should_fuse_first_and_second_loops() {
  %a = alloc() : memref<10xf32>
  %b = alloc() : memref<10xf32>
  %c = alloc() : memref<10xf32>

  %cf7 = constant 7.0 : f32

  for %i0 = 0 to 10 {
    store %cf7, %a[%i0] : memref<10xf32>
  }
  for %i1 = 0 to 10 {
    %v0 = load %a[%i1] : memref<10xf32>
    store %cf7, %b[%i1] : memref<10xf32>
  }
  for %i2 = 0 to 10 {
    %v1 = load %c[%i2] : memref<10xf32>
  }

  // Should fuse first loop into the second (last loop should not be fused).
  // CHECK:      for %i0 = 0 to 10 {
  // CHECK-NEXT:   %3 = affine_apply [[MAP0]](%i0)
  // CHECK-NEXT:   store %cst, %0[%3] : memref<10xf32>
  // CHECK-NEXT:   %4 = load %0[%i0] : memref<10xf32>
  // CHECK-NEXT:   store %cst, %1[%i0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      for %i1 = 0 to 10 {
  // CHECK-NEXT:   %5 = load %2[%i1] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return

  return
}

// -----

// CHECK-LABEL: func @should_not_fuse_would_create_cycle() {
func @should_not_fuse_would_create_cycle() {
  %a = alloc() : memref<10xf32>
  %b = alloc() : memref<10xf32>
  %c = alloc() : memref<10xf32>

  %cf7 = constant 7.0 : f32

  // Set up the following dependences:
  // 1) loop0 -> loop1 on memref '%a'
  // 2) loop0 -> loop2 on memref '%b'
  // 3) loop1 -> loop2 on memref '%c'
  for %i0 = 0 to 10 {
    %v0 = load %a[%i0] : memref<10xf32>
    store %cf7, %b[%i0] : memref<10xf32>
  }
  for %i1 = 0 to 10 {
    store %cf7, %a[%i1] : memref<10xf32>
    %v1 = load %c[%i1] : memref<10xf32>
  }
  for %i2 = 0 to 10 {
    %v2 = load %b[%i2] : memref<10xf32>
    store %cf7, %c[%i2] : memref<10xf32>
  }
  // Should not fuse: fusing loop first loop into last would create a cycle.
  // CHECK:      for %i0 = 0 to 10 {
  // CHECK-NEXT:   %3 = load %0[%i0] : memref<10xf32>
  // CHECK-NEXT:   store %cst, %1[%i0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      for %i1 = 0 to 10 {
  // CHECK-NEXT:   store %cst, %0[%i1] : memref<10xf32>
  // CHECK-NEXT:   %4 = load %2[%i1] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      for %i2 = 0 to 10 {
  // CHECK-NEXT:   %5 = load %1[%i2] : memref<10xf32>
  // CHECK-NEXT:   store %cst, %2[%i2] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func @should_not_fuse_raw_dep_would_be_violated() {
func @should_not_fuse_raw_dep_would_be_violated() {
  %m = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32

  for %i0 = 0 to 10 {
    store %cf7, %m[%i0] : memref<10xf32>
  }
  for %i1 = 0 to 10 {
    %v0 = load %m[%i1] : memref<10xf32>
  }
  for %i2 = 0 to 10 {
    %v1 = load %m[%i2] : memref<10xf32>
  }
  // Fusing loop %i0 to %i2 would violate the RAW dependence between %i0 and %i1
  // CHECK:      for %i0 = 0 to 10 {
  // CHECK-NEXT:   store %cst, %0[%i0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      for %i1 = 0 to 10 {
  // CHECK-NEXT:   %1 = load %0[%i1] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      for %i2 = 0 to 10 {
  // CHECK-NEXT:   %2 = load %0[%i2] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func @should_not_fuse_waw_dep_would_be_violated() {
func @should_not_fuse_waw_dep_would_be_violated() {
  %m = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32

  for %i0 = 0 to 10 {
    store %cf7, %m[%i0] : memref<10xf32>
  }
  for %i1 = 0 to 10 {
    store %cf7, %m[%i1] : memref<10xf32>
  }
  for %i2 = 0 to 10 {
    %v1 = load %m[%i2] : memref<10xf32>
  }
  // Fusing loop %i0 to %i2 would violate the WAW dependence between %i0 and %i1
  // CHECK:      for %i0 = 0 to 10 {
  // CHECK-NEXT:   store %cst, %0[%i0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      for %i1 = 0 to 10 {
  // CHECK-NEXT:   store %cst, %0[%i1] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      for %i2 = 0 to 10 {
  // CHECK-NEXT:   %1 = load %0[%i2] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func @should_not_fuse_war_dep_would_be_violated() {
func @should_not_fuse_war_dep_would_be_violated() {
  %a = alloc() : memref<10xf32>
  %b = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32

  for %i0 = 0 to 10 {
    %v0 = load %a[%i0] : memref<10xf32>
    store %v0, %b[%i0] : memref<10xf32>
  }
  for %i1 = 0 to 10 {
    store %cf7, %a[%i1] : memref<10xf32>
  }
  for %i2 = 0 to 10 {
    %v1 = load %b[%i2] : memref<10xf32>
  }
  // Fusing loop %i0 to %i2 would violate the WAR dependence between %i0 and %i1
  // CHECK:      for %i0 = 0 to 10 {
  // CHECK-NEXT:   %2 = load %0[%i0] : memref<10xf32>
  // CHECK-NEXT:   store %2, %1[%i0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      for %i1 = 0 to 10 {
  // CHECK-NEXT:   store %cst, %0[%i1] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      for %i2 = 0 to 10 {
  // CHECK-NEXT:   %3 = load %1[%i2] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func @should_not_fuse_if_top_level_access() {
func @should_not_fuse_if_top_level_access() {
  %m = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32

  for %i0 = 0 to 10 {
    store %cf7, %m[%i0] : memref<10xf32>
  }
  for %i1 = 0 to 10 {
    %v0 = load %m[%i1] : memref<10xf32>
  }

  %c0 = constant 4 : index
  %v1 = load %m[%c0] : memref<10xf32>
  // Top-level load to '%m' should prevent fusion.
  // CHECK:      for %i0 = 0 to 10 {
  // CHECK-NEXT:   store %cst, %0[%i0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      for %i1 = 0 to 10 {
  // CHECK-NEXT:   %1 = load %0[%i1] : memref<10xf32>
  // CHECK-NEXT: }
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = (d0) -> (d0)

// CHECK-LABEL: func @should_fuse_no_top_level_access() {
func @should_fuse_no_top_level_access() {
  %m = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32

  for %i0 = 0 to 10 {
    store %cf7, %m[%i0] : memref<10xf32>
  }
  for %i1 = 0 to 10 {
    %v0 = load %m[%i1] : memref<10xf32>
  }
  // CHECK:      for %i0 = 0 to 10 {
  // CHECK-NEXT:   %1 = affine_apply #map0(%i0)
  // CHECK-NEXT:   store %cst, %0[%1] : memref<10xf32>
  // CHECK-NEXT:   %2 = load %0[%i0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

#set0 = (d0) : (1 == 0)

// CHECK-LABEL: func @should_not_fuse_if_inst_at_top_level() {
func @should_not_fuse_if_inst_at_top_level() {
  %m = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32

  for %i0 = 0 to 10 {
    store %cf7, %m[%i0] : memref<10xf32>
  }
  for %i1 = 0 to 10 {
    %v0 = load %m[%i1] : memref<10xf32>
  }
  %c0 = constant 4 : index
  if #set0(%c0) {
  }
  // Top-level IfInst should prevent fusion.
  // CHECK:      for %i0 = 0 to 10 {
  // CHECK-NEXT:   store %cst, %0[%i0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      for %i1 = 0 to 10 {
  // CHECK-NEXT:   %1 = load %0[%i1] : memref<10xf32>
  // CHECK-NEXT: }
  return
}

// -----

#set0 = (d0) : (1 == 0)

// CHECK-LABEL: func @should_not_fuse_if_inst_in_loop_nest() {
func @should_not_fuse_if_inst_in_loop_nest() {
  %m = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32
  %c4 = constant 4 : index

  for %i0 = 0 to 10 {
    store %cf7, %m[%i0] : memref<10xf32>
  }
  for %i1 = 0 to 10 {
    if #set0(%c4) {
    }
    %v0 = load %m[%i1] : memref<10xf32>
  }

  // IfInst in ForInst should prevent fusion.
  // CHECK:      for %i0 = 0 to 10 {
  // CHECK-NEXT:   store %cst, %0[%i0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      for %i1 = 0 to 10 {
  // CHECK-NEXT:   if #set0(%c4) {
  // CHECK-NEXT:   }  
  // CHECK-NEXT:   %1 = load %0[%i1] : memref<10xf32>
  // CHECK-NEXT: }
  return
}

// -----

// CHECK-DAG: [[MAP_D0:#map[0-9]+]] = (d0, d1, d2) -> (d0)
// CHECK-DAG: [[MAP_D1:#map[0-9]+]] = (d0, d1, d2) -> (d1)
// CHECK-DAG: [[MAP_D2:#map[0-9]+]] = (d0, d1, d2) -> (d2)
// CHECK: [[MAP_IDENTITY:#map[0-9]+]] = (d0, d1, d2) -> (d0, d1, d2)
// CHECK: [[MAP_PERMUTE:#map[0-9]+]] = (d0, d1, d2) -> (d1, d2, d0)

// CHECK-LABEL: func @remap_ivs() {
func @remap_ivs() {
  %m = alloc() : memref<10x20x30xf32>

  %cf7 = constant 7.0 : f32
  for %i0 = 0 to 10 {
    for %i1 = 0 to 20 {
      for %i2 = 0 to 30 {
        %a0 = affine_apply (d0, d1, d2) -> (d0, d1, d2) (%i0, %i1, %i2)
        store %cf7, %m[%a0#0, %a0#1, %a0#2] : memref<10x20x30xf32>
      }
    }
  }
  for %i3 = 0 to 30 {
    for %i4 = 0 to 10 {
      for %i5 = 0 to 20 {
        %a1 = affine_apply (d0, d1, d2) -> (d1, d2, d0) (%i3, %i4, %i5)
        %v0 = load %m[%a1#0, %a1#1, %a1#2] : memref<10x20x30xf32>
      }
    }
  }
// CHECK:       for %i0 = 0 to 30 {
// CHECK-NEXT:    for %i1 = 0 to 10 {
// CHECK-NEXT:      for %i2 = 0 to 20 {
// CHECK-NEXT:        %1 = affine_apply [[MAP_D1]](%i0, %i1, %i2)
// CHECK-NEXT:        %2 = affine_apply [[MAP_D2]](%i0, %i1, %i2)
// CHECK-NEXT:        %3 = affine_apply [[MAP_D0]](%i0, %i1, %i2)
// CHECK-NEXT:        %4 = affine_apply [[MAP_IDENTITY]](%1, %2, %3)
// CHECK-NEXT:        store %cst, %0[%4#0, %4#1, %4#2] : memref<10x20x30xf32>
// CHECK-NEXT:        %5 = affine_apply [[MAP_PERMUTE]](%i0, %i1, %i2)
// CHECK-NEXT:        %6 = load %0[%5#0, %5#1, %5#2] : memref<10x20x30xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return

  return
}

// -----

// DEPTH1: #map0 = (d0) -> (d0)
// DEPTH1: #map1 = (d0, d1, d2) -> (d0, d1, d2)

// DEPTH1-LABEL: func @fuse_slice_at_depth1() {
func @fuse_slice_at_depth1() {
  %m = alloc() : memref<100x16x100xf32>

  %cf7 = constant 7.0 : f32
  for %i0 = 0 to 100 {
    for %i1 = 0 to 16 {
      for %i2 = 0 to 100 {
        %a0 = affine_apply (d0, d1, d2) -> (d0, d1, d2) (%i0, %i1, %i2)
        store %cf7, %m[%a0#0, %a0#1, %a0#2] : memref<100x16x100xf32>
      }
    }
  }
  for %i3 = 0 to 100 {
    for %i4 = 0 to 16 {
      for %i5 = 0 to 100 {
        %a1 = affine_apply (d0, d1, d2) -> (d0, d1, d2) (%i3, %i4, %i5)
        %v0 = load %m[%a1#0, %a1#1, %a1#2] : memref<100x16x100xf32>
      }
    }
  }
// DEPTH1:       for %i0 = 0 to 100 {
// DEPTH1-NEXT:    %1 = affine_apply #map0(%i0)
// DEPTH1-NEXT:    for %i1 = 0 to 16 {
// DEPTH1-NEXT:      for %i2 = 0 to 100 {
// DEPTH1-NEXT:        %2 = affine_apply #map1(%1, %i1, %i2)
// DEPTH1-NEXT:        store %cst, %0[%2#0, %2#1, %2#2] : memref<100x16x100xf32>
// DEPTH1-NEXT:      }
// DEPTH1-NEXT:    }
// DEPTH1-NEXT:    for %i3 = 0 to 16 {
// DEPTH1-NEXT:      for %i4 = 0 to 100 {
// DEPTH1-NEXT:        %3 = affine_apply #map1(%i0, %i3, %i4)
// DEPTH1-NEXT:        %4 = load %0[%3#0, %3#1, %3#2] : memref<100x16x100xf32>
// DEPTH1-NEXT:      }
// DEPTH1-NEXT:    }
// DEPTH1-NEXT:  }
// DEPTH1-NEXT:  return
  return
}

// -----

// CHECK-DAG: #map0 = (d0, d1) -> (d0 * 4 + d1)
// CHECK-DAG: #map1 = (d0) -> (d0 floordiv 4, d0 mod 4)

// Reshape from a 64 x f32 to 16 x 4 x f32.
// CHECK-LABEL: func @fuse_reshape_64_16_4
func @fuse_reshape_64_16_4(%in : memref<64xf32>) {
  %out = alloc() : memref<16x4xf32>

  for %i0 = 0 to 64 {
    %v = load %in[%i0] : memref<64xf32>
    %idx = affine_apply (d0) -> (d0 floordiv 4, d0 mod 4) (%i0)
    store %v, %out[%idx#0, %idx#1] : memref<16x4xf32>
  }

  for %i1 = 0 to 16 {
    for %i2 = 0 to 4 {
      %w = load %out[%i1, %i2] : memref<16x4xf32>
      "foo"(%w) : (f32) -> ()
    }
  }
  return
  // CHECK:      for %i0 =
  // CHECK-NEXT:   for %i1 =
  // CHECK-NOT:    for
  // CHECK:        }
  // CHECK-NEXT: }
  // CHECK-NEXT: return
}

// -----
// CHECK: #map0 = (d0) -> (d0 floordiv 4)
// CHECK: #map1 = (d0) -> (d0 mod 4)

// Reshape a 16x4xf32 to 64xf32.
// CHECK-LABEL: func @fuse_reshape_16_4_64
func @fuse_reshape_16_4_64() {
  %in = alloc() : memref<16x4xf32>
  %out = alloc() : memref<64xf32>

  for %i0 = 0 to 16 {
    for %i1 = 0 to 4 {
      %v = load %in[%i0, %i1] : memref<16x4xf32>
      %idx = affine_apply (d0, d1) -> (4*d0 + d1) (%i0, %i1)
      store %v, %out[%idx] : memref<64xf32>
    }
  }

  for %i2 = 0 to 64 {
    %w = load %out[%i2] : memref<64xf32>
    "foo"(%w) : (f32) -> ()
  }
// CHECK:       for %i0 = 0 to 64 {
// CHECK-NEXT:    %2 = affine_apply #map0(%i0)
// CHECK-NEXT:    %3 = affine_apply #map1(%i0)
// CHECK-NEXT:    %4 = load %0[%2, %3] : memref<16x4xf32>
// CHECK-NEXT:    %5 = affine_apply #map2(%2, %3)
// CHECK-NEXT:    store %4, %1[%5] : memref<64xf32>
// CHECK-NEXT:    %6 = load %1[%i0] : memref<64xf32>
// CHECK-NEXT:    "foo"(%6) : (f32) -> ()
// CHECK-NEXT:  }
// CHECK-NEXT:  return
  return
}


// -----

// All three loop nests below (6-d one, 2-d one, 2-d one is fused into a single
// 2-d loop nest).
// CHECK-LABEL: func @R6_to_R2_reshape
func @R6_to_R2_reshape_square() -> memref<64x9xi32> {
  %in = alloc() : memref<2x2x3x3x16x1xi32>
  %out = alloc() : memref<64x9xi32>

  // Initialize input with a different value for each 8x128 chunk.
  for %i0 = 0 to 2 {
    for %i1 = 0 to 2 {
      for %i2 = 0 to 3 {
        for %i3 = 0 to 3 {
          for %i4 = 0 to 16 {
            for %i5 = 0 to 1 {
              %val = "foo"(%i0, %i1, %i2, %i3, %i4, %i5) : (index, index, index, index, index, index) -> i32
              store %val, %in[%i0, %i1, %i2, %i3, %i4, %i5] : memref<2x2x3x3x16x1xi32>
            }
          }
        }
      }
    }
  }

  for %ii = 0 to 64 {
    for %jj = 0 to 9 {
      // Convert output coordinates to linear index.
      %a0 = affine_apply (d0, d1) -> (d0 * 9 + d1) (%ii, %jj)
      %a1 = affine_apply (d0) -> (
          d0 floordiv (2 * 3 * 3 * 16 * 1),
          (d0 mod 288) floordiv (3 * 3 * 16 * 1),
          ((d0 mod 288) mod 144) floordiv 48,
          (((d0 mod 288) mod 144) mod 48) floordiv 16,
          ((((d0 mod 288) mod 144) mod 48) mod 16),
          (((d0 mod 144) mod 144) mod 48) mod 16
        ) (%a0)
      %v = load %in[%a1#0, %a1#1, %a1#3, %a1#4, %a1#2, %a1#5]
        : memref<2x2x3x3x16x1xi32>
      store %v, %out[%ii, %jj] : memref<64x9xi32>
    }
  }

  for %i = 0 to 64 {
    for %j = 0 to 9 {
      %a = load %out[%i, %j] : memref<64x9xi32>
      %b = muli %a, %a : i32
      store %b, %out[%i, %j] : memref<64x9xi32>
    }
  }
  return %out : memref<64x9xi32>
}
// Everything above is fused to a single 2-d loop nest, and the 6-d tensor %in
// is eliminated if -memref-dataflow-opt is also supplied.
//
// CHECK:       for %i0 = 0 to 64 {
// CHECK-NEXT:    for %i1 = 0 to 9 {
// CHECK-NEXT:      %2 = affine_apply #map0(%i0, %i1)
// CHECK-NEXT:      %3 = affine_apply #map1(%i0, %i1)
// CHECK-NEXT:      %4 = affine_apply #map2(%i0, %i1)
// CHECK-NEXT:      %5 = affine_apply #map3(%i0, %i1)
// CHECK-NEXT:      %6 = affine_apply #map4(%i0, %i1)
// CHECK-NEXT:      %7 = "foo"(%2, %3, %4, %5, %6, %c0) : (index, index, index, index, index, index) -> i32
// CHECK-NEXT:      store %7, %0[%2, %3, %4, %5, %6, %c0] : memref<2x2x3x3x16x1xi32>
// CHECK-NEXT:      %8 = affine_apply #map5(%i0, %i1)
// CHECK-NEXT:      %9 = affine_apply #map6(%i0, %i1)
// CHECK-NEXT:      %10 = affine_apply #map7(%8, %9)
// CHECK-NEXT:      %11 = affine_apply #map8(%10)
// CHECK-NEXT:      %12 = load %0[%11#0, %11#1, %11#3, %11#4, %11#2, %11#5] : memref<2x2x3x3x16x1xi32>
// CHECK-NEXT:      store %12, %1[%8, %9] : memref<64x9xi32>
// CHECK-NEXT:      %13 = load %1[%i0, %i1] : memref<64x9xi32>
// CHECK-NEXT:      %14 = muli %13, %13 : i32
// CHECK-NEXT:      store %14, %1[%i0, %i1] : memref<64x9xi32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return %1 : memref<64x9xi32>

// -----

// CHECK-LABEL: func @fuse_symbolic_bounds
func @fuse_symbolic_bounds(%M : index, %N : index) {
  %m = alloc() : memref<800x800xf32>

  %c0 = constant 0.0 : f32
  %s = constant 5 : index

  for %i0 = 0 to %M {
    for %i1 = 0 to (d0) -> (d0 + 5) (%N) {
      store %c0, %m[%i0, %i1] : memref<800 x 800 x f32>
    }
  }

  for %i2 = 0 to %M {
    for %i3 = 0 to %N {
      %idx = affine_apply (d0, d1)[s0] -> (d0, d1 + s0) (%i2, %i3)[%s]
      %v = load %m[%idx#0, %idx#1] : memref<800 x 800 x f32>
    }
  }

  return
}
