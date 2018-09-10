//===- Builders.h - Helpers for constructing MLIR Classes -------*- C++ -*-===//
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

#ifndef MLIR_IR_BUILDERS_H
#define MLIR_IR_BUILDERS_H

#include "mlir/IR/CFGFunction.h"
#include "mlir/IR/MLFunction.h"
#include "mlir/IR/Statements.h"

namespace mlir {
class MLIRContext;
class Module;
class UnknownLoc;
class UniquedFilename;
class FileLineColLoc;
class Type;
class PrimitiveType;
class IntegerType;
class FunctionType;
class VectorType;
class RankedTensorType;
class UnrankedTensorType;
class BoolAttr;
class IntegerAttr;
class FloatAttr;
class StringAttr;
class TypeAttr;
class ArrayAttr;
class FunctionAttr;
class AffineMapAttr;
class AffineMap;
class AffineExpr;
class AffineConstantExpr;
class AffineDimExpr;
class AffineSymbolExpr;

/// This class is a general helper class for creating context-global objects
/// like types, attributes, and affine expressions.
class Builder {
public:
  explicit Builder(MLIRContext *context) : context(context) {}
  explicit Builder(Module *module);

  MLIRContext *getContext() const { return context; }

  Identifier getIdentifier(StringRef str);
  Module *createModule();

  // Locations.
  UnknownLoc *getUnknownLoc();
  UniquedFilename getUniquedFilename(StringRef filename);
  FileLineColLoc *getFileLineColLoc(UniquedFilename filename, unsigned line,
                                    unsigned column);

  // Types.
  FloatType *getBF16Type();
  FloatType *getF16Type();
  FloatType *getF32Type();
  FloatType *getF64Type();

  OtherType *getAffineIntType();
  OtherType *getTFControlType();
  OtherType *getTFStringType();
  IntegerType *getIntegerType(unsigned width);
  FunctionType *getFunctionType(ArrayRef<Type *> inputs,
                                ArrayRef<Type *> results);
  MemRefType *getMemRefType(ArrayRef<int> shape, Type *elementType,
                            ArrayRef<AffineMap *> affineMapComposition = {},
                            unsigned memorySpace = 0);
  VectorType *getVectorType(ArrayRef<unsigned> shape, Type *elementType);
  RankedTensorType *getTensorType(ArrayRef<int> shape, Type *elementType);
  UnrankedTensorType *getTensorType(Type *elementType);

  // Attributes.
  BoolAttr *getBoolAttr(bool value);
  IntegerAttr *getIntegerAttr(int64_t value);
  FloatAttr *getFloatAttr(double value);
  StringAttr *getStringAttr(StringRef bytes);
  ArrayAttr *getArrayAttr(ArrayRef<Attribute *> value);
  AffineMapAttr *getAffineMapAttr(AffineMap *value);
  TypeAttr *getTypeAttr(Type *type);
  FunctionAttr *getFunctionAttr(const Function *value);

  // Affine Expressions and Affine Map.
  AffineMap *getAffineMap(unsigned dimCount, unsigned symbolCount,
                          ArrayRef<AffineExpr *> results,
                          ArrayRef<AffineExpr *> rangeSizes);
  AffineDimExpr *getDimExpr(unsigned position);
  AffineSymbolExpr *getSymbolExpr(unsigned position);
  AffineConstantExpr *getConstantExpr(int64_t constant);
  AffineExpr *getAddExpr(AffineExpr *lhs, AffineExpr *rhs);
  AffineExpr *getSubExpr(AffineExpr *lhs, AffineExpr *rhs);
  AffineExpr *getMulExpr(AffineExpr *lhs, AffineExpr *rhs);
  AffineExpr *getModExpr(AffineExpr *lhs, AffineExpr *rhs);
  AffineExpr *getFloorDivExpr(AffineExpr *lhs, AffineExpr *rhs);
  AffineExpr *getCeilDivExpr(AffineExpr *lhs, AffineExpr *rhs);

  // Integer set.
  IntegerSet *getIntegerSet(unsigned dimCount, unsigned symbolCount,
                            ArrayRef<AffineExpr *> constraints,
                            ArrayRef<bool> isEq);

  // Special cases of affine maps and integer sets
  // One constant result: () -> (val).
  AffineMap *getConstantMap(int64_t val);
  // One dimension id identity map: (i) -> (i).
  AffineMap *getDimIdentityMap();
  // One symbol identity map: ()[s] -> (s).
  AffineMap *getSymbolIdentityMap();

  // TODO: Helpers for affine map/exprs, etc.
protected:
  MLIRContext *context;
};

/// This class helps build a CFGFunction.  Instructions that are created are
/// automatically inserted at an insertion point or added to the current basic
/// block.
class CFGFuncBuilder : public Builder {
public:
  CFGFuncBuilder(BasicBlock *block, BasicBlock::iterator insertPoint)
      : Builder(block->getFunction()->getContext()),
        function(block->getFunction()) {
    setInsertionPoint(block, insertPoint);
  }

  CFGFuncBuilder(OperationInst *insertBefore)
      : CFGFuncBuilder(insertBefore->getBlock(),
                       BasicBlock::iterator(insertBefore)) {}

  CFGFuncBuilder(BasicBlock *block)
      : Builder(block->getFunction()->getContext()),
        function(block->getFunction()) {
    setInsertionPoint(block);
  }

  CFGFuncBuilder(CFGFunction *function)
      : Builder(function->getContext()), function(function) {}

  /// Reset the insertion point to no location.  Creating an operation without a
  /// set insertion point is an error, but this can still be useful when the
  /// current insertion point a builder refers to is being removed.
  void clearInsertionPoint() {
    this->block = nullptr;
    insertPoint = BasicBlock::iterator();
  }

  /// Set the insertion point to the specified location.
  void setInsertionPoint(BasicBlock *block, BasicBlock::iterator insertPoint) {
    assert(block->getFunction() == function &&
           "can't move to a different function");
    this->block = block;
    this->insertPoint = insertPoint;
  }

  /// Set the insertion point to the specified operation.
  void setInsertionPoint(OperationInst *inst) {
    setInsertionPoint(inst->getBlock(), BasicBlock::iterator(inst));
  }

  /// Set the insertion point to the end of the specified block.
  void setInsertionPoint(BasicBlock *block) {
    setInsertionPoint(block, block->end());
  }

  void insert(OperationInst *opInst) {
    block->getOperations().insert(insertPoint, opInst);
  }

  /// Add new basic block and set the insertion point to the end of it.  If an
  /// 'insertBefore' basic block is passed, the block will be placed before the
  /// specified block.  If not, the block will be appended to the end of the
  /// current function.
  BasicBlock *createBlock(BasicBlock *insertBefore = nullptr);

  /// Create an operation given the fields represented as an OperationState.
  OperationInst *createOperation(const OperationState &state);

  /// Create operation of specific op type at the current insertion point
  /// without verifying to see if it is valid.
  template <typename OpTy, typename... Args>
  OpPointer<OpTy> create(Location *location, Args... args) {
    OperationState state(getContext(), location, OpTy::getOperationName());
    OpTy::build(this, &state, args...);
    auto *inst = createOperation(state);
    auto result = inst->template getAs<OpTy>();
    assert(result && "Builder didn't return the right type");
    return result;
  }

  /// Create operation of specific op type at the current insertion point.  If
  /// the result is an invalid op (the verifier hook fails), emit a the
  /// specified error message and return null.
  template <typename OpTy, typename... Args>
  OpPointer<OpTy> createChecked(Location *location, Args... args) {
    OperationState state(getContext(), location, OpTy::getOperationName());
    OpTy::build(this, &state, args...);
    auto *inst = createOperation(state);

    // If the OperationInst we produce is valid, return it.
    if (!OpTy::verifyInvariants(inst)) {
      auto result = inst->template getAs<OpTy>();
      assert(result && "Builder didn't return the right type");
      return result;
    }

    // Otherwise, the error message got emitted.  Just remove the instruction
    // we made.
    inst->eraseFromBlock();
    return OpPointer<OpTy>();
  }

  OperationInst *cloneOperation(const OperationInst &srcOpInst) {
    auto *op = srcOpInst.clone();
    insert(op);
    return op;
  }

  // Terminators.

  ReturnInst *createReturn(Location *location, ArrayRef<CFGValue *> operands) {
    return insertTerminator(ReturnInst::create(location, operands));
  }

  BranchInst *createBranch(Location *location, BasicBlock *dest,
                           ArrayRef<CFGValue *> operands = {}) {
    return insertTerminator(BranchInst::create(location, dest, operands));
  }

  CondBranchInst *createCondBranch(Location *location, CFGValue *condition,
                                   BasicBlock *trueDest,
                                   BasicBlock *falseDest) {
    return insertTerminator(
        CondBranchInst::create(location, condition, trueDest, falseDest));
  }

private:
  template <typename T>
  T *insertTerminator(T *term) {
    block->setTerminator(term);
    return term;
  }

  CFGFunction *function;
  BasicBlock *block = nullptr;
  BasicBlock::iterator insertPoint;
};

/// This class helps build an MLFunction.  Statements that are created are
/// automatically inserted at an insertion point or added to the current
/// statement block.
class MLFuncBuilder : public Builder {
public:
  /// Create ML function builder and set insertion point to the given statement,
  /// which will cause subsequent insertions to go right before it.
  MLFuncBuilder(Statement *stmt)
      // TODO: Eliminate findFunction from this.
      : Builder(stmt->findFunction()->getContext()) {
    setInsertionPoint(stmt);
  }

  MLFuncBuilder(StmtBlock *block, StmtBlock::iterator insertPoint)
      // TODO: Eliminate findFunction from this.
      : Builder(block->findFunction()->getContext()) {
    setInsertionPoint(block, insertPoint);
  }

  /// Reset the insertion point to no location.  Creating an operation without a
  /// set insertion point is an error, but this can still be useful when the
  /// current insertion point a builder refers to is being removed.
  void clearInsertionPoint() {
    this->block = nullptr;
    insertPoint = StmtBlock::iterator();
  }

  /// Set the insertion point to the specified location.
  /// Unlike CFGFuncBuilder, MLFuncBuilder allows to set insertion
  /// point to a different function.
  void setInsertionPoint(StmtBlock *block, StmtBlock::iterator insertPoint) {
    // TODO: check that insertPoint is in this rather than some other block.
    this->block = block;
    this->insertPoint = insertPoint;
  }

  /// Set the insertion point to the specified operation, which will cause
  /// subsequent insertions to go right before it.
  void setInsertionPoint(Statement *stmt) {
    setInsertionPoint(stmt->getBlock(), StmtBlock::iterator(stmt));
  }

  /// Set the insertion point to the start of the specified block.
  void setInsertionPointToStart(StmtBlock *block) {
    setInsertionPoint(block, block->begin());
  }

  /// Set the insertion point to the end of the specified block.
  void setInsertionPointToEnd(StmtBlock *block) {
    setInsertionPoint(block, block->end());
  }

  /// Get the current insertion point of the builder.
  StmtBlock::iterator getInsertionPoint() const { return insertPoint; }

  /// Create an operation given the fields represented as an OperationState.
  OperationStmt *createOperation(const OperationState &state);

  /// Create operation of specific op type at the current insertion point.
  template <typename OpTy, typename... Args>
  OpPointer<OpTy> create(Location *location, Args... args) {
    OperationState state(getContext(), location, OpTy::getOperationName());
    OpTy::build(this, &state, args...);
    auto *stmt = createOperation(state);
    auto result = stmt->template getAs<OpTy>();
    assert(result && "Builder didn't return the right type");
    return result;
  }

  /// Create operation of specific op type at the current insertion point.  If
  /// the result is an invalid op (the verifier hook fails), emit an error and
  /// return null.
  template <typename OpTy, typename... Args>
  OpPointer<OpTy> createChecked(Location *location, Args... args) {
    OperationState state(getContext(), location, OpTy::getOperationName());
    OpTy::build(this, &state, args...);
    auto *stmt = createOperation(state);

    // If the OperationStmt we produce is valid, return it.
    if (!OpTy::verifyInvariants(stmt)) {
      auto result = stmt->template getAs<OpTy>();
      assert(result && "Builder didn't return the right type");
      return result;
    }

    // Otherwise, the error message got emitted.  Just remove the statement
    // we made.
    stmt->eraseFromBlock();
    return OpPointer<OpTy>();
  }

  /// Create a deep copy of the specified statement, remapping any operands that
  /// use values outside of the statement using the map that is provided (
  /// leaving them alone if no entry is present).  Replaces references to cloned
  /// sub-statements to the corresponding statement that is copied, and adds
  /// those mappings to the map.
  Statement *clone(const Statement &stmt,
                   OperationStmt::OperandMapTy &operandMapping) {
    Statement *cloneStmt = stmt.clone(operandMapping, getContext());
    block->getStatements().insert(insertPoint, cloneStmt);
    return cloneStmt;
  }

  // Create for statement. When step is not specified, it is set to 1.
  ForStmt *createFor(Location *location, ArrayRef<MLValue *> lbOperands,
                     AffineMap *lbMap, ArrayRef<MLValue *> ubOperands,
                     AffineMap *ubMap, int64_t step = 1);

  /// Create if statement.
  IfStmt *createIf(Location *location, ArrayRef<MLValue *> operands,
                   IntegerSet *set);

private:
  StmtBlock *block = nullptr;
  StmtBlock::iterator insertPoint;
};

} // namespace mlir

#endif
