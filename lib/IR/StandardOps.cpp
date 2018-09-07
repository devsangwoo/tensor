//===- StandardOps.cpp - Standard MLIR Operations -------------------------===//
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

#include "mlir/IR/StandardOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSet.h"
#include "mlir/IR/SSAValue.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

static void printDimAndSymbolList(Operation::const_operand_iterator begin,
                                  Operation::const_operand_iterator end,
                                  unsigned numDims, OpAsmPrinter *p) {
  *p << '(';
  p->printOperands(begin, begin + numDims);
  *p << ')';

  if (begin + numDims != end) {
    *p << '[';
    p->printOperands(begin + numDims, end);
    *p << ']';
  }
}

// Parses dimension and symbol list, and sets 'numDims' to the number of
// dimension operands parsed.
// Returns 'false' on success and 'true' on error.
static bool
parseDimAndSymbolList(OpAsmParser *parser,
                      SmallVector<SSAValue *, 4> &operands, unsigned &numDims) {
  SmallVector<OpAsmParser::OperandType, 8> opInfos;
  if (parser->parseOperandList(opInfos, -1, OpAsmParser::Delimiter::Paren))
    return true;
  // Store number of dimensions for validation by caller.
  numDims = opInfos.size();

  // Parse the optional symbol operands.
  auto *affineIntTy = parser->getBuilder().getAffineIntType();
  if (parser->parseOperandList(opInfos, -1,
                               OpAsmParser::Delimiter::OptionalSquare) ||
      parser->resolveOperands(opInfos, affineIntTy, operands))
    return true;
  return false;
}

//===----------------------------------------------------------------------===//
// AddFOp
//===----------------------------------------------------------------------===//

void AddFOp::build(Builder *builder, OperationState *result, SSAValue *lhs,
                   SSAValue *rhs) {
  assert(lhs->getType() == rhs->getType());
  result->addOperands({lhs, rhs});
  result->types.push_back(lhs->getType());
}

bool AddFOp::parse(OpAsmParser *parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 2> ops;
  Type *type;
  return parser->parseOperandList(ops, 2) ||
         parser->parseOptionalAttributeDict(result->attributes) ||
         parser->parseColonType(type) ||
         parser->resolveOperands(ops, type, result->operands) ||
         parser->addTypeToList(type, result->types);
}

void AddFOp::print(OpAsmPrinter *p) const {
  *p << "addf " << *getOperand(0) << ", " << *getOperand(1);
  p->printOptionalAttrDict(getAttrs());
  *p << " : " << *getType();
}

// TODO: Have verify functions return std::string to enable more descriptive
// error messages.
// Return an error message on failure.
const char *AddFOp::verify() const {
  // TODO: Check that the types of the LHS and RHS match.
  // TODO: This should be a refinement of TwoOperands.
  // TODO: There should also be a OneResultWhoseTypeMatchesFirstOperand.
  return nullptr;
}

//===----------------------------------------------------------------------===//
// AffineApplyOp
//===----------------------------------------------------------------------===//

bool AffineApplyOp::parse(OpAsmParser *parser, OperationState *result) {
  auto &builder = parser->getBuilder();
  auto *affineIntTy = builder.getAffineIntType();

  AffineMapAttr *mapAttr;
  unsigned numDims;
  if (parser->parseAttribute(mapAttr, "map", result->attributes) ||
      parseDimAndSymbolList(parser, result->operands, numDims) ||
      parser->parseOptionalAttributeDict(result->attributes))
    return true;
  auto *map = mapAttr->getValue();

  if (map->getNumDims() != numDims ||
      numDims + map->getNumSymbols() != result->operands.size()) {
    return parser->emitError(parser->getNameLoc(),
                             "dimension or symbol index mismatch");
  }

  result->types.append(map->getNumResults(), affineIntTy);
  return false;
}

void AffineApplyOp::print(OpAsmPrinter *p) const {
  auto *map = getAffineMap();
  *p << "affine_apply " << *map;
  printDimAndSymbolList(operand_begin(), operand_end(), map->getNumDims(), p);
  p->printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/"map");
}

const char *AffineApplyOp::verify() const {
  // Check that affine map attribute was specified.
  auto *affineMapAttr = getAttrOfType<AffineMapAttr>("map");
  if (!affineMapAttr)
    return "requires an affine map";

  // Check input and output dimensions match.
  auto *map = affineMapAttr->getValue();

  // Verify that operand count matches affine map dimension and symbol count.
  if (getNumOperands() != map->getNumDims() + map->getNumSymbols())
    return "operand count and affine map dimension and symbol count must match";

  // Verify that result count matches affine map result count.
  if (getNumResults() != map->getNumResults())
    return "result count and affine map result count must match";

  return nullptr;
}

// The result of the affine apply operation can be used as a dimension id if it
// is a CFG value or if it is an MLValue, and all the operands are valid
// dimension ids.
bool AffineApplyOp::isValidDim() const {
  for (auto *op : getOperands()) {
    if (auto *v = dyn_cast<MLValue>(op))
      if (!v->isValidDim())
        return false;
  }
  return true;
}

// The result of the affine apply operation can be used as a symbol if it is
// a CFG value or if it is an MLValue, and all the operands are symbols.
bool AffineApplyOp::isValidSymbol() const {
  for (auto *op : getOperands()) {
    if (auto *v = dyn_cast<MLValue>(op))
      if (!v->isValidSymbol())
        return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

void AllocOp::build(Builder *builder, OperationState *result,
                    MemRefType *memrefType, ArrayRef<SSAValue *> operands) {
  result->addOperands(operands);
  result->types.push_back(memrefType);
}

void AllocOp::print(OpAsmPrinter *p) const {
  MemRefType *type = cast<MemRefType>(getMemRef()->getType());
  *p << "alloc";
  // Print dynamic dimension operands.
  printDimAndSymbolList(operand_begin(), operand_end(),
                        type->getNumDynamicDims(), p);
  p->printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/"map");
  *p << " : " << *type;
}

bool AllocOp::parse(OpAsmParser *parser, OperationState *result) {
  MemRefType *type;

  // Parse the dimension operands and optional symbol operands, followed by a
  // memref type.
  unsigned numDimOperands;
  if (parseDimAndSymbolList(parser, result->operands, numDimOperands) ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonType(type))
    return true;

  // Check numDynamicDims against number of question marks in memref type.
  if (numDimOperands != type->getNumDynamicDims()) {
    return parser->emitError(parser->getNameLoc(),
                             "dimension operand count does not equal memref "
                             "dynamic dimension count");
  }

  // Check that the number of symbol operands matches the number of symbols in
  // the first affinemap of the memref's affine map composition.
  // Note that a memref must specify at least one affine map in the composition.
  if (result->operands.size() - numDimOperands !=
      type->getAffineMaps()[0]->getNumSymbols()) {
    return parser->emitError(
        parser->getNameLoc(),
        "affine map symbol operand count does not equal memref affine map "
        "symbol count");
  }

  result->types.push_back(type);
  return false;
}

const char *AllocOp::verify() const {
  // TODO(andydavis): Verify alloc.
  return nullptr;
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

void CallOp::build(Builder *builder, OperationState *result, Function *callee,
                   ArrayRef<SSAValue *> operands) {
  result->addOperands(operands);
  result->addAttribute("callee", builder->getFunctionAttr(callee));
  result->addTypes(callee->getType()->getResults());
}

bool CallOp::parse(OpAsmParser *parser, OperationState *result) {
  StringRef calleeName;
  llvm::SMLoc calleeLoc;
  FunctionType *calleeType = nullptr;
  SmallVector<OpAsmParser::OperandType, 4> operands;
  Function *callee = nullptr;
  if (parser->parseFunctionName(calleeName, calleeLoc) ||
      parser->parseOperandList(operands, /*requiredOperandCount=*/-1,
                               OpAsmParser::Delimiter::Paren) ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonType(calleeType) ||
      parser->resolveFunctionName(calleeName, calleeType, calleeLoc, callee) ||
      parser->addTypesToList(calleeType->getResults(), result->types) ||
      parser->resolveOperands(operands, calleeType->getInputs(), calleeLoc,
                              result->operands))
    return true;

  result->addAttribute("callee", parser->getBuilder().getFunctionAttr(callee));
  return false;
}

void CallOp::print(OpAsmPrinter *p) const {
  *p << "call ";
  p->printFunctionReference(getCallee());
  *p << '(';
  p->printOperands(getOperands());
  *p << ')';
  p->printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/"callee");
  *p << " : " << *getCallee()->getType();
}

const char *CallOp::verify() const {
  // Check that the callee attribute was specified.
  auto *fnAttr = getAttrOfType<FunctionAttr>("callee");
  if (!fnAttr)
    return "requires a 'callee' function attribute";

  // Verify that the operand and result types match the callee.
  auto *fnType = fnAttr->getValue()->getType();
  if (fnType->getNumInputs() != getNumOperands())
    return "incorrect number of operands for callee";

  for (unsigned i = 0, e = fnType->getNumInputs(); i != e; ++i) {
    if (getOperand(i)->getType() != fnType->getInput(i))
      return "operand type mismatch";
  }

  if (fnType->getNumResults() != getNumResults())
    return "incorrect number of results for callee";

  for (unsigned i = 0, e = fnType->getNumResults(); i != e; ++i) {
    if (getResult(i)->getType() != fnType->getResult(i))
      return "result type mismatch";
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// CallIndirectOp
//===----------------------------------------------------------------------===//

void CallIndirectOp::build(Builder *builder, OperationState *result,
                           SSAValue *callee, ArrayRef<SSAValue *> operands) {
  auto *fnType = cast<FunctionType>(callee->getType());
  result->operands.push_back(callee);
  result->addOperands(operands);
  result->addTypes(fnType->getResults());
}

bool CallIndirectOp::parse(OpAsmParser *parser, OperationState *result) {
  FunctionType *calleeType = nullptr;
  OpAsmParser::OperandType callee;
  llvm::SMLoc operandsLoc;
  SmallVector<OpAsmParser::OperandType, 4> operands;
  return parser->parseOperand(callee) ||
         parser->getCurrentLocation(&operandsLoc) ||
         parser->parseOperandList(operands, /*requiredOperandCount=*/-1,
                                  OpAsmParser::Delimiter::Paren) ||
         parser->parseOptionalAttributeDict(result->attributes) ||
         parser->parseColonType(calleeType) ||
         parser->resolveOperand(callee, calleeType, result->operands) ||
         parser->resolveOperands(operands, calleeType->getInputs(), operandsLoc,
                                 result->operands) ||
         parser->addTypesToList(calleeType->getResults(), result->types);
}

void CallIndirectOp::print(OpAsmPrinter *p) const {
  *p << "call_indirect ";
  p->printOperand(getCallee());
  *p << '(';
  auto operandRange = getOperands();
  p->printOperands(++operandRange.begin(), operandRange.end());
  *p << ')';
  p->printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/"callee");
  *p << " : " << *getCallee()->getType();
}

const char *CallIndirectOp::verify() const {
  // The callee must be a function.
  auto *fnType = dyn_cast<FunctionType>(getCallee()->getType());
  if (!fnType)
    return "callee must have function type";

  // Verify that the operand and result types match the callee.
  if (fnType->getNumInputs() != getNumOperands() - 1)
    return "incorrect number of operands for callee";

  for (unsigned i = 0, e = fnType->getNumInputs(); i != e; ++i) {
    if (getOperand(i + 1)->getType() != fnType->getInput(i))
      return "operand type mismatch";
  }

  if (fnType->getNumResults() != getNumResults())
    return "incorrect number of results for callee";

  for (unsigned i = 0, e = fnType->getNumResults(); i != e; ++i) {
    if (getResult(i)->getType() != fnType->getResult(i))
      return "result type mismatch";
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// Constant*Op
//===----------------------------------------------------------------------===//

void ConstantOp::print(OpAsmPrinter *p) const {
  *p << "constant " << *getValue();
  p->printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/"value");

  if (!isa<FunctionAttr>(getValue()))
    *p << " : " << *getType();
}

bool ConstantOp::parse(OpAsmParser *parser, OperationState *result) {
  Attribute *valueAttr;
  Type *type;

  if (parser->parseAttribute(valueAttr, "value", result->attributes) ||
      parser->parseOptionalAttributeDict(result->attributes))
    return true;

  // 'constant' taking a function reference doesn't get a redundant type
  // specifier.  The attribute itself carries it.
  if (auto *fnAttr = dyn_cast<FunctionAttr>(valueAttr))
    return parser->addTypeToList(fnAttr->getValue()->getType(), result->types);

  return parser->parseColonType(type) ||
         parser->addTypeToList(type, result->types);
}

/// The constant op requires an attribute, and furthermore requires that it
/// matches the return type.
const char *ConstantOp::verify() const {
  auto *value = getValue();
  if (!value)
    return "requires a 'value' attribute";

  auto *type = this->getType();
  if (isa<IntegerType>(type) || type->isAffineInt()) {
    if (!isa<IntegerAttr>(value))
      return "requires 'value' to be an integer for an integer result type";
    return nullptr;
  }

  if (isa<FloatType>(type)) {
    if (!isa<FloatAttr>(value))
      return "requires 'value' to be a floating point constant";
    return nullptr;
  }

  if (type->isTFString()) {
    if (!isa<StringAttr>(value))
      return "requires 'value' to be a string constant";
    return nullptr;
  }

  if (isa<FunctionType>(type)) {
    if (!isa<FunctionAttr>(value))
      return "requires 'value' to be a function reference";
    return nullptr;
  }

  return "requires a result type that aligns with the 'value' attribute";
}

void ConstantFloatOp::build(Builder *builder, OperationState *result,
                            double value, FloatType *type) {
  result->addAttribute("value", builder->getFloatAttr(value));
  result->types.push_back(type);
}

bool ConstantFloatOp::isClassFor(const Operation *op) {
  return ConstantOp::isClassFor(op) &&
         isa<FloatType>(op->getResult(0)->getType());
}

/// ConstantIntOp only matches values whose result type is an IntegerType.
bool ConstantIntOp::isClassFor(const Operation *op) {
  return ConstantOp::isClassFor(op) &&
         isa<IntegerType>(op->getResult(0)->getType());
}

void ConstantIntOp::build(Builder *builder, OperationState *result,
                          int64_t value, unsigned width) {
  result->addAttribute("value", builder->getIntegerAttr(value));
  result->types.push_back(builder->getIntegerType(width));
}

/// ConstantAffineIntOp only matches values whose result type is AffineInt.
bool ConstantAffineIntOp::isClassFor(const Operation *op) {
  return ConstantOp::isClassFor(op) &&
         op->getResult(0)->getType()->isAffineInt();
}

void ConstantAffineIntOp::build(Builder *builder, OperationState *result,
                                int64_t value) {
  result->addAttribute("value", builder->getIntegerAttr(value));
  result->types.push_back(builder->getAffineIntType());
}

//===----------------------------------------------------------------------===//
// AffineApplyOp
//===----------------------------------------------------------------------===//

void AffineApplyOp::build(Builder *builder, OperationState *result,
                          AffineMap *map, ArrayRef<SSAValue *> operands) {
  result->addOperands(operands);
  result->types.append(map->getNumResults(), builder->getAffineIntType());
  result->addAttribute("map", builder->getAffineMapAttr(map));
}

//===----------------------------------------------------------------------===//
// DeallocOp
//===----------------------------------------------------------------------===//

void DeallocOp::build(Builder *builder, OperationState *result,
                      SSAValue *memref) {
  result->addOperands(memref);
}

void DeallocOp::print(OpAsmPrinter *p) const {
  *p << "dealloc " << *getMemRef() << " : " << *getMemRef()->getType();
}

bool DeallocOp::parse(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType memrefInfo;
  MemRefType *type;

  return parser->parseOperand(memrefInfo) || parser->parseColonType(type) ||
         parser->resolveOperand(memrefInfo, type, result->operands);
}

const char *DeallocOp::verify() const {
  if (!isa<MemRefType>(getMemRef()->getType()))
    return "operand must be a memref";
  return nullptr;
}

//===----------------------------------------------------------------------===//
// DimOp
//===----------------------------------------------------------------------===//

void DimOp::print(OpAsmPrinter *p) const {
  *p << "dim " << *getOperand() << ", " << getIndex();
  p->printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/"index");
  *p << " : " << *getOperand()->getType();
}

bool DimOp::parse(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType operandInfo;
  IntegerAttr *indexAttr;
  Type *type;

  return parser->parseOperand(operandInfo) || parser->parseComma() ||
         parser->parseAttribute(indexAttr, "index", result->attributes) ||
         parser->parseOptionalAttributeDict(result->attributes) ||
         parser->parseColonType(type) ||
         parser->resolveOperand(operandInfo, type, result->operands) ||
         parser->addTypeToList(parser->getBuilder().getAffineIntType(),
                               result->types);
}

const char *DimOp::verify() const {
  // Check that we have an integer index operand.
  auto indexAttr = getAttrOfType<IntegerAttr>("index");
  if (!indexAttr)
    return "requires an integer attribute named 'index'";
  uint64_t index = (uint64_t)indexAttr->getValue();

  auto *type = getOperand()->getType();
  if (auto *tensorType = dyn_cast<RankedTensorType>(type)) {
    if (index >= tensorType->getRank())
      return "index is out of range";
  } else if (auto *memrefType = dyn_cast<MemRefType>(type)) {
    if (index >= memrefType->getRank())
      return "index is out of range";

  } else if (isa<UnrankedTensorType>(type)) {
    // ok, assumed to be in-range.
  } else {
    return "requires an operand with tensor or memref type";
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// ExtractElementOp
//===----------------------------------------------------------------------===//

void ExtractElementOp::build(Builder *builder, OperationState *result,
                             SSAValue *aggregate,
                             ArrayRef<SSAValue *> indices) {
  auto *aggregateType = cast<VectorOrTensorType>(aggregate->getType());
  result->addOperands(aggregate);
  result->addOperands(indices);
  result->types.push_back(aggregateType->getElementType());
}

void ExtractElementOp::print(OpAsmPrinter *p) const {
  *p << "extract_element " << *getAggregate() << '[';
  p->printOperands(getIndices());
  *p << ']';
  p->printOptionalAttrDict(getAttrs());
  *p << " : " << *getAggregate()->getType();
}

bool ExtractElementOp::parse(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType aggregateInfo;
  SmallVector<OpAsmParser::OperandType, 4> indexInfo;
  VectorOrTensorType *type;

  auto affineIntTy = parser->getBuilder().getAffineIntType();
  return parser->parseOperand(aggregateInfo) ||
         parser->parseOperandList(indexInfo, -1,
                                  OpAsmParser::Delimiter::Square) ||
         parser->parseOptionalAttributeDict(result->attributes) ||
         parser->parseColonType(type) ||
         parser->resolveOperand(aggregateInfo, type, result->operands) ||
         parser->resolveOperands(indexInfo, affineIntTy, result->operands) ||
         parser->addTypeToList(type->getElementType(), result->types);
}

const char *ExtractElementOp::verify() const {
  if (getNumOperands() == 0)
    return "expected an aggregate to index into";

  auto *aggregateType = dyn_cast<VectorOrTensorType>(getAggregate()->getType());
  if (!aggregateType)
    return "first operand must be a vector or tensor";

  if (getResult()->getType() != aggregateType->getElementType())
    return "result type must match element type of aggregate";

  for (auto *idx : getIndices())
    if (!idx->getType()->isAffineInt())
      return "index to extract_element must have 'affineint' type";

  // Verify the # indices match if we have a ranked type.
  auto aggregateRank = aggregateType->getRankIfPresent();
  if (aggregateRank != -1 && aggregateRank != getNumOperands() - 1)
    return "incorrect number of indices for extract_element";

  return nullptr;
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

void LoadOp::build(Builder *builder, OperationState *result, SSAValue *memref,
                   ArrayRef<SSAValue *> indices) {
  auto *memrefType = cast<MemRefType>(memref->getType());
  result->addOperands(memref);
  result->addOperands(indices);
  result->types.push_back(memrefType->getElementType());
}

void LoadOp::print(OpAsmPrinter *p) const {
  *p << "load " << *getMemRef() << '[';
  p->printOperands(getIndices());
  *p << ']';
  p->printOptionalAttrDict(getAttrs());
  *p << " : " << *getMemRef()->getType();
}

bool LoadOp::parse(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType memrefInfo;
  SmallVector<OpAsmParser::OperandType, 4> indexInfo;
  MemRefType *type;

  auto affineIntTy = parser->getBuilder().getAffineIntType();
  return parser->parseOperand(memrefInfo) ||
         parser->parseOperandList(indexInfo, -1,
                                  OpAsmParser::Delimiter::Square) ||
         parser->parseOptionalAttributeDict(result->attributes) ||
         parser->parseColonType(type) ||
         parser->resolveOperand(memrefInfo, type, result->operands) ||
         parser->resolveOperands(indexInfo, affineIntTy, result->operands) ||
         parser->addTypeToList(type->getElementType(), result->types);
}

const char *LoadOp::verify() const {
  if (getNumOperands() == 0)
    return "expected a memref to load from";

  auto *memRefType = dyn_cast<MemRefType>(getMemRef()->getType());
  if (!memRefType)
    return "first operand must be a memref";

  if (getResult()->getType() != memRefType->getElementType())
    return "result type must match element type of memref";

  if (memRefType->getRank() != getNumOperands() - 1)
    return "incorrect number of indices for load";

  for (auto *idx : getIndices())
    if (!idx->getType()->isAffineInt())
      return "index to load must have 'affineint' type";

  // TODO: Verify we have the right number of indices.

  // TODO: in MLFunction verify that the indices are parameters, IV's, or the
  // result of an affine_apply.
  return nullptr;
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

bool ReturnOp::parse(OpAsmParser *parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 2> opInfo;
  SmallVector<Type *, 2> types;
  llvm::SMLoc loc;
  return parser->getCurrentLocation(&loc) || parser->parseOperandList(opInfo) ||
         (!opInfo.empty() && parser->parseColonTypeList(types)) ||
         parser->resolveOperands(opInfo, types, loc, result->operands);
}

void ReturnOp::print(OpAsmPrinter *p) const {
  *p << "return";
  if (getNumOperands() > 0) {
    *p << " ";
    p->printOperands(operand_begin(), operand_end());
    *p << " : ";
    interleave(operand_begin(), operand_end(),
               [&](const SSAValue *e) { p->printType(e->getType()); },
               [&]() { *p << ", "; });
  }
}

const char *ReturnOp::verify() const {
  // ReturnOp must be part of an ML function.
  if (auto *stmt = dyn_cast<OperationStmt>(getOperation())) {
    StmtBlock *block = stmt->getBlock();
    if (!block || !isa<MLFunction>(block) || &block->back() != stmt)
      return "must be the last statement in the ML function";

    // Return success. Checking that operand types match those in the function
    // signature is performed in the ML function verifier.
    return nullptr;
  }
  return "cannot occur in a CFG function";
}

//===----------------------------------------------------------------------===//
// ShapeCastOp
//===----------------------------------------------------------------------===//

void ShapeCastOp::build(Builder *builder, OperationState *result,
                        SSAValue *input, Type *resultType) {
  result->addOperands(input);
  result->addTypes(resultType);
}

const char *ShapeCastOp::verify() const {
  auto *opType = dyn_cast<TensorType>(getOperand()->getType());
  auto *resType = dyn_cast<TensorType>(getResult()->getType());
  if (!opType || !resType)
    return "requires input and result types to be tensors";

  if (opType == resType)
    return "requires the input and result type to be different";

  if (opType->getElementType() != resType->getElementType())
    return "requires input and result element types to be the same";

  // If the source or destination are unranked, then the cast is valid.
  auto *opRType = dyn_cast<RankedTensorType>(opType);
  auto *resRType = dyn_cast<RankedTensorType>(resType);
  if (!opRType || !resRType)
    return nullptr;

  // If they are both ranked, they have to have the same rank, and any specified
  // dimensions must match.
  if (opRType->getRank() != resRType->getRank())
    return "requires input and result ranks to match";

  for (unsigned i = 0, e = opRType->getRank(); i != e; ++i) {
    int opDim = opRType->getDimSize(i), resultDim = resRType->getDimSize(i);
    if (opDim != -1 && resultDim != -1 && opDim != resultDim)
      return "requires static dimensions to match";
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

void StoreOp::build(Builder *builder, OperationState *result,
                    SSAValue *valueToStore, SSAValue *memref,
                    ArrayRef<SSAValue *> indices) {
  result->addOperands(valueToStore);
  result->addOperands(memref);
  result->addOperands(indices);
}

void StoreOp::print(OpAsmPrinter *p) const {
  *p << "store " << *getValueToStore();
  *p << ", " << *getMemRef() << '[';
  p->printOperands(getIndices());
  *p << ']';
  p->printOptionalAttrDict(getAttrs());
  *p << " : " << *getMemRef()->getType();
}

bool StoreOp::parse(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType storeValueInfo;
  OpAsmParser::OperandType memrefInfo;
  SmallVector<OpAsmParser::OperandType, 4> indexInfo;
  MemRefType *memrefType;

  auto affineIntTy = parser->getBuilder().getAffineIntType();
  return parser->parseOperand(storeValueInfo) || parser->parseComma() ||
         parser->parseOperand(memrefInfo) ||
         parser->parseOperandList(indexInfo, -1,
                                  OpAsmParser::Delimiter::Square) ||
         parser->parseOptionalAttributeDict(result->attributes) ||
         parser->parseColonType(memrefType) ||
         parser->resolveOperand(storeValueInfo, memrefType->getElementType(),
                                result->operands) ||
         parser->resolveOperand(memrefInfo, memrefType, result->operands) ||
         parser->resolveOperands(indexInfo, affineIntTy, result->operands);
}

const char *StoreOp::verify() const {
  if (getNumOperands() < 2)
    return "expected a value to store and a memref";

  // Second operand is a memref type.
  auto *memRefType = dyn_cast<MemRefType>(getMemRef()->getType());
  if (!memRefType)
    return "second operand must be a memref";

  // First operand must have same type as memref element type.
  if (getValueToStore()->getType() != memRefType->getElementType())
    return "first operand must have same type memref element type ";

  if (getNumOperands() != 2 + memRefType->getRank())
    return "store index operand count not equal to memref rank";

  for (auto *idx : getIndices())
    if (!idx->getType()->isAffineInt())
      return "index to load must have 'affineint' type";

  // TODO: Verify we have the right number of indices.

  // TODO: in MLFunction verify that the indices are parameters, IV's, or the
  // result of an affine_apply.
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Register operations.
//===----------------------------------------------------------------------===//

/// Install the standard operations in the specified operation set.
void mlir::registerStandardOperations(OperationSet &opSet) {
  opSet.addOperations<AddFOp, AffineApplyOp, AllocOp, CallOp, CallIndirectOp,
                      ConstantOp, DeallocOp, DimOp, ExtractElementOp, LoadOp,
                      ReturnOp, ShapeCastOp, StoreOp>(
      /*prefix=*/"");
}
