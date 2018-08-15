// RUN: %S/../../mlir-opt %s -o - | FileCheck %s

// CHECK: #map0 = (d0, d1) -> (d0, d1)
// CHECK: #map1 = (d0, d1)[s0] -> (d0 + s0, d1)

// CHECK-LABEL: cfgfunc @alloc() {
cfgfunc @alloc() {
bb0:
  // Test simple alloc.
  // CHECK: %0 = alloc() : memref<1024x64xf32, #map0, 1>
  %0 = alloc() : memref<1024x64xf32, (d0, d1) -> (d0, d1), 1>

  %1 = "constant"() {value: 0} : () -> affineint
  %2 = "constant"() {value: 1} : () -> affineint

  // Test alloc with dynamic dimensions.
  // CHECK: %1 = alloc(%c0, %c1) : memref<?x?xf32, #map0, 1>
  %3 = alloc(%1, %2) : memref<?x?xf32, (d0, d1) -> (d0, d1), 1>

  // Test alloc with no dynamic dimensions and one symbol.
  // CHECK: %2 = alloc()[%c0] : memref<2x4xf32, #map1, 1>
  %4 = alloc()[%1] : memref<2x4xf32, (d0, d1)[s0] -> ((d0 + s0), d1), 1>

  // Test alloc with dynamic dimensions and one symbol.
  // CHECK: %3 = alloc(%c1)[%c0] : memref<2x?xf32, #map1, 1>
  %5 = alloc(%2)[%1] : memref<2x?xf32, (d0, d1)[s0] -> (d0 + s0, d1), 1>

  // CHECK:   return
  return
}

// CHECK-LABEL: cfgfunc @dealloc() {
cfgfunc @dealloc() {
bb0:
  // CHECK: %0 = alloc() : memref<1024x64xf32, #map0>
  %0 = alloc() : memref<1024x64xf32, (d0, d1) -> (d0, d1), 0>

  // CHECK: dealloc %0 : memref<1024x64xf32, #map0>
  dealloc %0 : memref<1024x64xf32, (d0, d1) -> (d0, d1), 0>
  return
}

// CHECK-LABEL: cfgfunc @load_store
cfgfunc @load_store() {
bb0:
  // CHECK: %0 = alloc() : memref<1024x64xf32, #map0, 1>
  %0 = alloc() : memref<1024x64xf32, (d0, d1) -> (d0, d1), 1>

  %1 = constant 0 : affineint
  %2 = constant 1 : affineint

  // CHECK: %1 = load %0[%c0, %c1] : memref<1024x64xf32, #map0, 1>
  %3 = load %0[%1, %2] : memref<1024x64xf32, (d0, d1) -> (d0, d1), 1>

  // CHECK: store %1, %0[%c0, %c1] : memref<1024x64xf32, #map0, 1>
  store %3, %0[%1, %2] : memref<1024x64xf32, (d0, d1) -> (d0, d1), 1>

  return
}