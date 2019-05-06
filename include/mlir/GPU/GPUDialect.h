//===- GPUDialect.h - MLIR Dialect for GPU Kernels --------------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file defines the GPU kernel-related operations and puts them in the
// corresponding dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_GPU_GPUDIALECT_H
#define MLIR_GPU_GPUDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {

/// The dialect containing GPU kernel launching operations and related
/// facilities.
class GPUDialect : public Dialect {
public:
  /// Create the dialect in the given `context`.
  GPUDialect(MLIRContext *context);

  /// Get the canonical string name of the dialect.
  static StringRef getDialectName();
};

/// Utility class for the GPU dialect to represent triples of `Value`s
/// accessible through `.x`, `.y`, and `.z` similarly to CUDA notation.
struct KernelDim3 {
  Value *x;
  Value *y;
  Value *z;
};

/// GPU kernel launch operation.  Takes a 3D grid of thread blocks as leading
/// operands, followed by kernel data operands.  Has one region representing
/// the kernel to be executed.  This region is not allowed to use values defined
/// outside it.
class LaunchOp : public Op<LaunchOp, OpTrait::AtLeastNOperands<6>::Impl,
                           OpTrait::ZeroResult,
                           OpTrait::NthRegionIsIsolatedAbove<0>::Impl> {
public:
  friend Operation;
  using Op::Op;

  static void build(Builder *builder, OperationState *result, Value *gridSizeX,
                    Value *gridSizeY, Value *gridSizeZ, Value *blockSizeX,
                    Value *blockSizeY, Value *blockSizeZ,
                    ArrayRef<Value *> operands);

  /// Get the kernel region.
  Region &getBody();

  /// Get the SSA values corresponding to kernel block identifiers.
  KernelDim3 getBlockIds();
  /// Get the SSA values corresponding to kernel thread identifiers.
  KernelDim3 getThreadIds();
  /// Get the SSA values corresponding to kernel grid size.
  KernelDim3 getGridSize();
  /// Get the SSA values corresponding to kernel block size.
  KernelDim3 getBlockSize();

  LogicalResult verify();

  /// Custom syntax support.
  void print(OpAsmPrinter *p);
  static bool parse(OpAsmParser *parser, OperationState *result);

  static StringRef getOperationName() { return "gpu.launch"; }

private:
  static StringRef getBlocksKeyword() { return "blocks"; }
  static StringRef getThreadsKeyword() { return "threads"; }
  static StringRef getArgsKeyword() { return "args"; }

  /// The number of launch configuration operands, placed at the leading
  /// positions of the operand list.
  static constexpr unsigned kNumConfigOperands = 6;

  /// The number of region attributes containing the launch configuration,
  /// placed in the leading positions of the argument list.
  static constexpr unsigned kNumConfigRegionAttributes = 12;
};

} // end namespace mlir

#endif // MLIR_GPUKERNEL_GPUDIALECT_H
