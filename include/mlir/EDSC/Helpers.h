//===- Helpers.h - MLIR Declarative Helper Functionality --------*- C++ -*-===//
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
// Provides helper classes and syntactic sugar for declarative builders.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EDSC_HELPERS_H_
#define MLIR_EDSC_HELPERS_H_

#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Intrinsics.h"

namespace mlir {
namespace edsc {

class IndexedValue;

/// An IndexHandle is a simple wrapper around a ValueHandle.
/// IndexHandles are ubiquitous enough to justify a new type to allow simple
/// declarations without boilerplate such as:
///
/// ```c++
///    IndexHandle i, j, k;
/// ```
struct IndexHandle : public ValueHandle {
  explicit IndexHandle()
      : ValueHandle(ScopedContext::getBuilder()->getIndexType()) {}
  explicit IndexHandle(index_t v) : ValueHandle(v) {}
  explicit IndexHandle(Value *v) : ValueHandle(v) {
    assert(v->getType() == ScopedContext::getBuilder()->getIndexType() &&
           "Expected index type");
  }
  explicit IndexHandle(ValueHandle v) : ValueHandle(v) {
    assert(v.getType() == ScopedContext::getBuilder()->getIndexType() &&
           "Expected index type");
  }
  IndexHandle &operator=(const ValueHandle &v) {
    assert(v.getType() == ScopedContext::getBuilder()->getIndexType() &&
           "Expected index type");
    /// Creating a new IndexHandle(v) and then std::swap rightly complains the
    /// binding has already occurred and that we should use another name.
    this->t = v.getType();
    this->v = v.getValue();
    return *this;
  }
};

/// Helper structure to be used with EDSCValueBuilder / EDSCInstructionBuilder.
/// It serves the purpose of removing boilerplate specialization for the sole
/// purpose of implicitly converting ArrayRef<ValueHandle> -> ArrayRef<Value*>.
class ValueHandleArray {
public:
  ValueHandleArray(ArrayRef<ValueHandle> vals) {
    values.append(vals.begin(), vals.end());
  }
  ValueHandleArray(ArrayRef<IndexHandle> vals) {
    values.append(vals.begin(), vals.end());
  }
  ValueHandleArray(ArrayRef<index_t> vals) {
    llvm::SmallVector<IndexHandle, 8> tmp(vals.begin(), vals.end());
    values.append(tmp.begin(), tmp.end());
  }
  operator ArrayRef<Value *>() { return values; }

private:
  llvm::SmallVector<Value *, 8> values;
};

// Base class for MemRefView and VectorView.
class View {
public:
  unsigned rank() const { return lbs.size(); }
  ValueHandle lb(unsigned idx) { return lbs[idx]; }
  ValueHandle ub(unsigned idx) { return ubs[idx]; }
  int64_t step(unsigned idx) { return steps[idx]; }
  std::tuple<ValueHandle, ValueHandle, int64_t> range(unsigned idx) {
    return std::make_tuple(lbs[idx], ubs[idx], steps[idx]);
  }
  void swapRanges(unsigned i, unsigned j) {
    if (i == j)
      return;
    lbs[i].swap(lbs[j]);
    ubs[i].swap(ubs[j]);
    std::swap(steps[i], steps[j]);
  }

  ArrayRef<ValueHandle> getLbs() { return lbs; }
  ArrayRef<ValueHandle> getUbs() { return ubs; }
  ArrayRef<int64_t> getSteps() { return steps; }

protected:
  SmallVector<ValueHandle, 8> lbs;
  SmallVector<ValueHandle, 8> ubs;
  SmallVector<int64_t, 8> steps;
};

/// A MemRefView represents the information required to step through a
/// MemRef. It has placeholders for non-contiguous tensors that fit within the
/// Fortran subarray model.
/// At the moment it can only capture a MemRef with an identity layout map.
// TODO(ntv): Support MemRefs with layoutMaps.
class MemRefView : public View {
public:
  explicit MemRefView(Value *v);
  MemRefView(const MemRefView &) = default;
  MemRefView &operator=(const MemRefView &) = default;

  unsigned fastestVarying() const { return rank() - 1; }

private:
  friend IndexedValue;
  ValueHandle base;
};

/// A VectorView represents the information required to step through a
/// Vector accessing each scalar element at a time. It is the counterpart of
/// a MemRefView but for vectors. This exists purely for boilerplate avoidance.
class VectorView : public View {
public:
  explicit VectorView(Value *v);
  VectorView(const VectorView &) = default;
  VectorView &operator=(const VectorView &) = default;

private:
  friend IndexedValue;
  ValueHandle base;
};

/// This helper class is an abstraction over memref, that purely for sugaring
/// purposes and allows writing compact expressions such as:
///
/// ```mlir
///    IndexedValue A(...), B(...), C(...);
///    For(ivs, zeros, shapeA, ones, {
///      C(ivs) = A(ivs) + B(ivs)
///    });
/// ```
///
/// Assigning to an IndexedValue emits an actual store operation, while using
/// converting an IndexedValue to a ValueHandle emits an actual load operation.
struct IndexedValue {
  explicit IndexedValue(Type t) : base(t) {}
  explicit IndexedValue(Value *v) : IndexedValue(ValueHandle(v)) {}
  explicit IndexedValue(ValueHandle v) : base(v) {}

  IndexedValue(const IndexedValue &rhs) = default;

  ValueHandle operator()() { return ValueHandle(*this); }
  /// Returns a new `IndexedValue`.
  IndexedValue operator()(ValueHandle index) {
    IndexedValue res(base);
    res.indices.push_back(index);
    return res;
  }
  template <typename... Args>
  IndexedValue operator()(ValueHandle index, Args... indices) {
    return IndexedValue(base, index).append(indices...);
  }
  IndexedValue operator()(llvm::ArrayRef<ValueHandle> indices) {
    return IndexedValue(base, indices);
  }
  IndexedValue operator()(llvm::ArrayRef<IndexHandle> indices) {
    return IndexedValue(
        base, llvm::ArrayRef<ValueHandle>(indices.begin(), indices.end()));
  }

  /// Emits a `store`.
  // NOLINTNEXTLINE: unconventional-assign-operator
  InstructionHandle operator=(const IndexedValue &rhs) {
    ValueHandle rrhs(rhs);
    assert(getBase().getType().cast<MemRefType>().getRank() == indices.size() &&
           "Unexpected number of indices to store in MemRef");
    return intrinsics::store(rrhs, getBase(), ValueHandleArray(indices));
  }
  // NOLINTNEXTLINE: unconventional-assign-operator
  InstructionHandle operator=(ValueHandle rhs) {
    assert(getBase().getType().cast<MemRefType>().getRank() == indices.size() &&
           "Unexpected number of indices to store in MemRef");
    return intrinsics::store(rhs, getBase(), ValueHandleArray(indices));
  }

  /// Emits a `load` when converting to a ValueHandle.
  operator ValueHandle() const {
    assert(getBase().getType().cast<MemRefType>().getRank() == indices.size() &&
           "Unexpected number of indices to store in MemRef");
    return intrinsics::load(getBase(), ValueHandleArray(indices));
  }

  ValueHandle getBase() const { return base; }

  /// Operator overloadings.
  ValueHandle operator+(ValueHandle e);
  ValueHandle operator-(ValueHandle e);
  ValueHandle operator*(ValueHandle e);
  ValueHandle operator/(ValueHandle e);
  InstructionHandle operator+=(ValueHandle e);
  InstructionHandle operator-=(ValueHandle e);
  InstructionHandle operator*=(ValueHandle e);
  InstructionHandle operator/=(ValueHandle e);
  ValueHandle operator+(IndexedValue e) {
    return *this + static_cast<ValueHandle>(e);
  }
  ValueHandle operator-(IndexedValue e) {
    return *this - static_cast<ValueHandle>(e);
  }
  ValueHandle operator*(IndexedValue e) {
    return *this * static_cast<ValueHandle>(e);
  }
  ValueHandle operator/(IndexedValue e) {
    return *this / static_cast<ValueHandle>(e);
  }
  InstructionHandle operator+=(IndexedValue e) {
    return this->operator+=(static_cast<ValueHandle>(e));
  }
  InstructionHandle operator-=(IndexedValue e) {
    return this->operator-=(static_cast<ValueHandle>(e));
  }
  InstructionHandle operator*=(IndexedValue e) {
    return this->operator*=(static_cast<ValueHandle>(e));
  }
  InstructionHandle operator/=(IndexedValue e) {
    return this->operator/=(static_cast<ValueHandle>(e));
  }

private:
  IndexedValue(ValueHandle base, ArrayRef<ValueHandle> indices)
      : base(base), indices(indices.begin(), indices.end()) {}

  IndexedValue &append() { return *this; }

  template <typename T, typename... Args>
  IndexedValue &append(T index, Args... indices) {
    this->indices.push_back(static_cast<ValueHandle>(index));
    append(indices...);
    return *this;
  }
  ValueHandle base;
  llvm::SmallVector<ValueHandle, 8> indices;
};

} // namespace edsc
} // namespace mlir

#endif // MLIR_EDSC_HELPERS_H_
