//===- Types.cpp - MLIR Type Classes --------------------------------------===//
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

#include "mlir/IR/Types.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/STLExtras.h"
using namespace mlir;

IntegerType::IntegerType(unsigned width, MLIRContext *context)
  : Type(Kind::Integer, context), width(width) {
}

FloatType::FloatType(Kind kind, MLIRContext *context) : Type(kind, context) {}

OtherType::OtherType(Kind kind, MLIRContext *context) : Type(kind, context) {}

FunctionType::FunctionType(Type *const *inputsAndResults, unsigned numInputs,
                           unsigned numResults, MLIRContext *context)
  : Type(Kind::Function, context, numInputs),
    numResults(numResults), inputsAndResults(inputsAndResults) {
}

VectorType::VectorType(ArrayRef<unsigned> shape, Type *elementType,
                       MLIRContext *context)
    : Type(Kind::Vector, context, shape.size()), shapeElements(shape.data()),
      elementType(elementType) {}

RankedTensorType::RankedTensorType(ArrayRef<int> shape, Type *elementType,
                                   MLIRContext *context)
  : TensorType(Kind::RankedTensor, elementType, context),
    shapeElements(shape.data()) {
  setSubclassData(shape.size());
}

UnrankedTensorType::UnrankedTensorType(Type *elementType, MLIRContext *context)
  : TensorType(Kind::UnrankedTensor, elementType, context) {
}

MemRefType::MemRefType(ArrayRef<int> shape, Type *elementType,
                       ArrayRef<AffineMap *> affineMapList,
                       unsigned memorySpace, MLIRContext *context)
    : Type(Kind::MemRef, context, shape.size()), elementType(elementType),
      shapeElements(shape.data()), numAffineMaps(affineMapList.size()),
      affineMapList(affineMapList.data()), memorySpace(memorySpace) {}

ArrayRef<AffineMap*> MemRefType::getAffineMaps() const {
  return ArrayRef<AffineMap*>(affineMapList, numAffineMaps);
}

unsigned MemRefType::getNumDynamicDims() {
  unsigned numDynamicDims = 0;
  for (int dimSize : getShape()) {
    if (dimSize < 0)
      ++numDynamicDims;
  }
  return numDynamicDims;
}
