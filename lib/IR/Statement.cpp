//===- Statement.cpp - MLIR Statement Classes ----------------------------===//
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

#include "mlir/IR/MLFunction.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Statements.h"
#include "mlir/IR/StmtVisitor.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/DenseMap.h"
using namespace mlir;

//===----------------------------------------------------------------------===//
// StmtResult
//===------------------------------------------------------------------===//

/// Return the result number of this result.
unsigned StmtResult::getResultNumber() const {
  // Results are always stored consecutively, so use pointer subtraction to
  // figure out what number this is.
  return this - &getOwner()->getStmtResults()[0];
}

//===----------------------------------------------------------------------===//
// Statement
//===------------------------------------------------------------------===//

// Statements are deleted through the destroy() member because we don't have
// a virtual destructor.
Statement::~Statement() {
  assert(block == nullptr && "statement destroyed but still in a block");
}

/// Destroy this statement or one of its subclasses.
void Statement::destroy() {
  switch (this->getKind()) {
  case Kind::Operation:
    cast<OperationStmt>(this)->destroy();
    break;
  case Kind::For:
    delete cast<ForStmt>(this);
    break;
  case Kind::If:
    delete cast<IfStmt>(this);
    break;
  }
}

/// Return the context this operation is associated with.
MLIRContext *Statement::getContext() const {
  // Work a bit to avoid calling findFunction() and getting its context.
  switch (getKind()) {
  case Kind::Operation:
    return cast<OperationStmt>(this)->getContext();
  case Kind::For:
    return cast<ForStmt>(this)->getType()->getContext();
  case Kind::If:
    // TODO(shpeisman): When if statement has value operands, we can get a
    // context from their type.
    return findFunction()->getContext();
  }
}

Statement *Statement::getParentStmt() const {
  return block ? block->getParentStmt() : nullptr;
}

MLFunction *Statement::findFunction() const {
  return block ? block->findFunction() : nullptr;
}

bool Statement::isInnermost() const {
  struct NestedLoopCounter : public StmtWalker<NestedLoopCounter> {
    unsigned numNestedLoops;
    NestedLoopCounter() : numNestedLoops(0) {}
    void walkForStmt(const ForStmt *fs) { numNestedLoops++; }
  };

  NestedLoopCounter nlc;
  nlc.walk(const_cast<Statement *>(this));
  return nlc.numNestedLoops == 1;
}

/// Emit a note about this statement, reporting up to any diagnostic
/// handlers that may be listening.
void Statement::emitNote(const Twine &message) const {
  getContext()->emitDiagnostic(getLoc(), message,
                               MLIRContext::DiagnosticKind::Note);
}

/// Emit a warning about this statement, reporting up to any diagnostic
/// handlers that may be listening.
void Statement::emitWarning(const Twine &message) const {
  getContext()->emitDiagnostic(getLoc(), message,
                               MLIRContext::DiagnosticKind::Warning);
}

/// Emit an error about fatal conditions with this statement, reporting up to
/// any diagnostic handlers that may be listening.  NOTE: This may terminate
/// the containing application, only use when the IR is in an inconsistent
/// state.
void Statement::emitError(const Twine &message) const {
  getContext()->emitDiagnostic(getLoc(), message,
                               MLIRContext::DiagnosticKind::Error);
}
//===----------------------------------------------------------------------===//
// ilist_traits for Statement
//===----------------------------------------------------------------------===//

StmtBlock *llvm::ilist_traits<::mlir::Statement>::getContainingBlock() {
  size_t Offset(
      size_t(&((StmtBlock *)nullptr->*StmtBlock::getSublistAccess(nullptr))));
  iplist<Statement> *Anchor(static_cast<iplist<Statement> *>(this));
  return reinterpret_cast<StmtBlock *>(reinterpret_cast<char *>(Anchor) -
                                       Offset);
}

/// This is a trait method invoked when a statement is added to a block.  We
/// keep the block pointer up to date.
void llvm::ilist_traits<::mlir::Statement>::addNodeToList(Statement *stmt) {
  assert(!stmt->getBlock() && "already in a statement block!");
  stmt->block = getContainingBlock();
}

/// This is a trait method invoked when a statement is removed from a block.
/// We keep the block pointer up to date.
void llvm::ilist_traits<::mlir::Statement>::removeNodeFromList(
    Statement *stmt) {
  assert(stmt->block && "not already in a statement block!");
  stmt->block = nullptr;
}

/// This is a trait method invoked when a statement is moved from one block
/// to another.  We keep the block pointer up to date.
void llvm::ilist_traits<::mlir::Statement>::transferNodesFromList(
    ilist_traits<Statement> &otherList, stmt_iterator first,
    stmt_iterator last) {
  // If we are transferring statements within the same block, the block
  // pointer doesn't need to be updated.
  StmtBlock *curParent = getContainingBlock();
  if (curParent == otherList.getContainingBlock())
    return;

  // Update the 'block' member of each statement.
  for (; first != last; ++first)
    first->block = curParent;
}

/// Remove this statement (and its descendants) from its StmtBlock and delete
/// all of them.
void Statement::eraseFromBlock() {
  assert(getBlock() && "Statement has no block");
  getBlock()->getStatements().erase(this);
}

//===----------------------------------------------------------------------===//
// OperationStmt
//===----------------------------------------------------------------------===//

/// Create a new OperationStmt with the specific fields.
OperationStmt *OperationStmt::create(Attribute *location, Identifier name,
                                     ArrayRef<MLValue *> operands,
                                     ArrayRef<Type *> resultTypes,
                                     ArrayRef<NamedAttribute> attributes,
                                     MLIRContext *context) {
  auto byteSize = totalSizeToAlloc<StmtOperand, StmtResult>(operands.size(),
                                                            resultTypes.size());
  void *rawMem = malloc(byteSize);

  // Initialize the OperationStmt part of the statement.
  auto stmt = ::new (rawMem) OperationStmt(
      location, name, operands.size(), resultTypes.size(), attributes, context);

  // Initialize the operands and results.
  auto stmtOperands = stmt->getStmtOperands();
  for (unsigned i = 0, e = operands.size(); i != e; ++i)
    new (&stmtOperands[i]) StmtOperand(stmt, operands[i]);

  auto stmtResults = stmt->getStmtResults();
  for (unsigned i = 0, e = resultTypes.size(); i != e; ++i)
    new (&stmtResults[i]) StmtResult(resultTypes[i], stmt);
  return stmt;
}

OperationStmt::OperationStmt(Attribute *location, Identifier name,
                             unsigned numOperands, unsigned numResults,
                             ArrayRef<NamedAttribute> attributes,
                             MLIRContext *context)
    : Operation(/*isInstruction=*/false, name, attributes, context),
      Statement(Kind::Operation, location), numOperands(numOperands),
      numResults(numResults) {}

OperationStmt::~OperationStmt() {
  // Explicitly run the destructors for the operands and results.
  for (auto &operand : getStmtOperands())
    operand.~StmtOperand();

  for (auto &result : getStmtResults())
    result.~StmtResult();
}

void OperationStmt::destroy() {
  this->~OperationStmt();
  free(this);
}

/// Return the context this operation is associated with.
MLIRContext *OperationStmt::getContext() const {
  // If we have a result or operand type, that is a constant time way to get
  // to the context.
  if (getNumResults())
    return getResult(0)->getType()->getContext();
  if (getNumOperands())
    return getOperand(0)->getType()->getContext();

  // In the very odd case where we have no operands or results, fall back to
  // doing a find.
  return findFunction()->getContext();
}

//===----------------------------------------------------------------------===//
// ForStmt
//===----------------------------------------------------------------------===//

ForStmt::ForStmt(Attribute *location, AffineConstantExpr *lowerBound,
                 AffineConstantExpr *upperBound, int64_t step,
                 MLIRContext *context)
    : Statement(Kind::For, location),
      MLValue(MLValueKind::ForStmt, Type::getAffineInt(context)),
      StmtBlock(StmtBlockKind::For), lowerBound(lowerBound),
      upperBound(upperBound), step(step) {}

//===----------------------------------------------------------------------===//
// IfStmt
//===----------------------------------------------------------------------===//

IfStmt::IfStmt(Attribute *location, IntegerSet *condition)
    : Statement(Kind::If, location), thenClause(new IfClause(this)),
      elseClause(nullptr), condition(condition) {}

IfStmt::~IfStmt() {
  delete thenClause;
  if (elseClause)
    delete elseClause;
  // An IfStmt's IntegerSet 'condition' should not be deleted since it is
  // allocated through MLIRContext's bump pointer allocator.
}

//===----------------------------------------------------------------------===//
// Statement Cloning
//===----------------------------------------------------------------------===//

/// Create a deep copy of this statement, remapping any operands that use
/// values outside of the statement using the map that is provided (leaving
/// them alone if no entry is present).  Replaces references to cloned
/// sub-statements to the corresponding statement that is copied, and adds
/// those mappings to the map.
Statement *Statement::clone(DenseMap<const MLValue *, MLValue *> &operandMap,
                            MLIRContext *context) const {
  // If the specified value is in operandMap, return the remapped value.
  // Otherwise return the value itself.
  auto remapOperand = [&](const MLValue *value) -> MLValue * {
    auto it = operandMap.find(value);
    return it != operandMap.end() ? it->second : const_cast<MLValue *>(value);
  };

  if (auto *opStmt = dyn_cast<OperationStmt>(this)) {
    SmallVector<MLValue *, 8> operands;
    operands.reserve(opStmt->getNumOperands());
    for (auto *opValue : opStmt->getOperands())
      operands.push_back(remapOperand(opValue));

    SmallVector<Type *, 8> resultTypes;
    resultTypes.reserve(opStmt->getNumResults());
    for (auto *result : opStmt->getResults())
      resultTypes.push_back(result->getType());
    auto *newOp =
        OperationStmt::create(getLoc(), opStmt->getName(), operands,
                              resultTypes, opStmt->getAttrs(), context);
    // Remember the mapping of any results.
    for (unsigned i = 0, e = opStmt->getNumResults(); i != e; ++i)
      operandMap[opStmt->getResult(i)] = newOp->getResult(i);
    return newOp;
  }

  if (auto *forStmt = dyn_cast<ForStmt>(this)) {
    auto *newFor =
        new ForStmt(getLoc(), forStmt->getLowerBound(),
                    forStmt->getUpperBound(), forStmt->getStep(), context);
    // Remember the induction variable mapping.
    operandMap[forStmt] = newFor;

    // TODO: remap operands in loop bounds when they are added.
    // Recursively clone the body of the for loop.
    for (auto &subStmt : *forStmt)
      newFor->push_back(subStmt.clone(operandMap, context));

    return newFor;
  }

  // Otherwise, we must have an If statement.
  auto *ifStmt = cast<IfStmt>(this);
  auto *newIf = new IfStmt(getLoc(), ifStmt->getCondition());

  // TODO: remap operands with remapOperand when if statements have them.

  auto *resultThen = newIf->getThen();
  for (auto &childStmt : *ifStmt->getThen())
    resultThen->push_back(childStmt.clone(operandMap, context));

  if (ifStmt->hasElse()) {
    auto *resultElse = newIf->createElse();
    for (auto &childStmt : *ifStmt->getElse())
      resultElse->push_back(childStmt.clone(operandMap, context));
  }

  return newIf;
}
