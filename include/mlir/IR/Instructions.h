//===- Instructions.h - MLIR CFG Instruction Classes ------------*- C++ -*-===//
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
// This file defines the classes for CFGFunction instructions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_INSTRUCTIONS_H
#define MLIR_IR_INSTRUCTIONS_H

#include "mlir/Support/LLVM.h"

namespace mlir {
  class BasicBlock;
  class CFGFunction;


/// Terminator instructions are the last part of a basic block, used to
/// represent control flow and returns.
class TerminatorInst {
public:
  enum class Kind {
    Branch,
    Return
  };

  Kind getKind() const { return kind; }

  /// Return the BasicBlock that contains this terminator instruction.
  BasicBlock *getBlock() const {
    return block;
  }
  CFGFunction *getFunction() const;

  void print(raw_ostream &os) const;
  void dump() const;

protected:
  TerminatorInst(Kind kind, BasicBlock *block) : kind(kind), block(block) {}

private:
  Kind kind;
  BasicBlock *block;
};

/// The 'br' instruction is an unconditional from one basic block to another,
/// and may pass basic block arguments to the successor.
class BranchInst : public TerminatorInst {
public:
  explicit BranchInst(BasicBlock *dest, BasicBlock *parent);

  /// Return the block this branch jumps to.
  BasicBlock *getDest() const {
    return dest;
  }

  // TODO: need to take BB arguments.

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const TerminatorInst *inst) {
    return inst->getKind() == Kind::Branch;
  }
private:
  BasicBlock *dest;
};


/// The 'return' instruction represents the end of control flow within the
/// current function, and can return zero or more results.  The result list is
/// required to align with the result list of the containing function's type.
class ReturnInst : public TerminatorInst {
public:
  explicit ReturnInst(BasicBlock *parent);

  // TODO: Needs to take an operand list.

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const TerminatorInst *inst) {
    return inst->getKind() == Kind::Return;
  }
};

} // end namespace mlir

#endif  // MLIR_IR_INSTRUCTIONS_H
