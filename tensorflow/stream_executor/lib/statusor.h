<<<<<<< HEAD
/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// StatusOr<T> is the union of a Status object and a T object. StatusOr models
// the concept of an object that is either a value, or an error Status
// explaining why such a value is not present. To this end, StatusOr<T> does not
// allow its Status value to be Status::OK.
=======
// Copyright 2008 Google Inc. All Rights Reserved.
// Author: acm@google.com (Andrew Morrow)
// Author: zhengxq@google.com (Xiaoqiang Zheng)
//
// StatusOr<T> is the union of a Status object and a T
// object. StatusOr models the concept of an object that is either a
// usable value, or an error Status explaining why such a value is
// not present. To this end, StatusOr<T> does not allow its Status
// value to be Status::OK. Further, StatusOr<T*> does not allow the
// contained pointer to be NULL.
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
//
// The primary use-case for StatusOr<T> is as the return value of a
// function which may fail.
//
// Example client usage for a StatusOr<T>, where T is not a pointer:
//
//  StatusOr<float> result = DoBigCalculationThatCouldFail();
//  if (result.ok()) {
//    float answer = result.ValueOrDie();
//    printf("Big calculation yielded: %f", answer);
//  } else {
//    LOG(ERROR) << result.status();
//  }
//
// Example client usage for a StatusOr<T*>:
//
//  StatusOr<Foo*> result = FooFactory::MakeNewFoo(arg);
//  if (result.ok()) {
//    std::unique_ptr<Foo> foo(result.ValueOrDie());
//    foo->DoSomethingCool();
//  } else {
//    LOG(ERROR) << result.status();
//  }
//
// Example client usage for a StatusOr<std::unique_ptr<T>>:
//
//  StatusOr<std::unique_ptr<Foo>> result = FooFactory::MakeNewFoo(arg);
//  if (result.ok()) {
<<<<<<< HEAD
//    std::unique_ptr<Foo> foo = std::move(result.ValueOrDie());
=======
//    std::unique_ptr<Foo> foo = result.ConsumeValueOrDie();
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
//    foo->DoSomethingCool();
//  } else {
//    LOG(ERROR) << result.status();
//  }
//
// Example factory implementation returning StatusOr<T*>:
//
//  StatusOr<Foo*> FooFactory::MakeNewFoo(int arg) {
//    if (arg <= 0) {
<<<<<<< HEAD
//      return tensorflow::InvalidArgument("Arg must be positive");
=======
//      return Status(port::error::INVALID_ARGUMENT,
//                            "Arg must be positive");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
//    } else {
//      return new Foo(arg);
//    }
//  }
//
<<<<<<< HEAD
// Note that the assignment operators require that destroying the currently
// stored value cannot invalidate the argument; in other words, the argument
// cannot be an alias for the current value, or anything owned by the current
// value.
#ifndef TENSORFLOW_STREAM_EXECUTOR_LIB_STATUSOR_H_
#define TENSORFLOW_STREAM_EXECUTOR_LIB_STATUSOR_H_

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/statusor_internals.h"

namespace stream_executor {
namespace port {

#if defined(__clang__)
// Only clang supports warn_unused_result as a type annotation.
template <typename T>
class TF_MUST_USE_RESULT StatusOr;
#endif

template <typename T>
class StatusOr : private internal_statusor::StatusOrData<T>,
                 private internal_statusor::TraitsBase<
                     std::is_copy_constructible<T>::value,
                     std::is_move_constructible<T>::value> {
  template <typename U>
  friend class StatusOr;

  typedef internal_statusor::StatusOrData<T> Base;

 public:
  typedef T element_type;

  // Constructs a new StatusOr with Status::UNKNOWN status.  This is marked
  // 'explicit' to try to catch cases like 'return {};', where people think
  // StatusOr<std::vector<int>> will be initialized with an empty vector,
  // instead of a Status::UNKNOWN status.
  explicit StatusOr();

  // StatusOr<T> will be copy constructible/assignable if T is copy
  // constructible.
  StatusOr(const StatusOr&) = default;
  StatusOr& operator=(const StatusOr&) = default;

  // StatusOr<T> will be move constructible/assignable if T is move
  // constructible.
  StatusOr(StatusOr&&) = default;
  StatusOr& operator=(StatusOr&&) = default;

  // Conversion copy/move constructor, T must be convertible from U.
  template <typename U, typename std::enable_if<
                            std::is_convertible<U, T>::value>::type* = nullptr>
  StatusOr(const StatusOr<U>& other);
  template <typename U, typename std::enable_if<
                            std::is_convertible<U, T>::value>::type* = nullptr>
  StatusOr(StatusOr<U>&& other);

  // Conversion copy/move assignment operator, T must be convertible from U.
  template <typename U, typename std::enable_if<
                            std::is_convertible<U, T>::value>::type* = nullptr>
  StatusOr& operator=(const StatusOr<U>& other);
  template <typename U, typename std::enable_if<
                            std::is_convertible<U, T>::value>::type* = nullptr>
  StatusOr& operator=(StatusOr<U>&& other);

  // Constructs a new StatusOr with the given value. After calling this
  // constructor, calls to ValueOrDie() will succeed, and calls to status() will
  // return OK.
  //
  // NOTE: Not explicit - we want to use StatusOr<T> as a return type
  // so it is convenient and sensible to be able to do 'return T()'
  // when the return type is StatusOr<T>.
  //
  // REQUIRES: T is copy constructible.
  StatusOr(const T& value);

  // Constructs a new StatusOr with the given non-ok status. After calling
  // this constructor, calls to ValueOrDie() will CHECK-fail.
=======

#ifndef TENSORFLOW_STREAM_EXECUTOR_LIB_STATUSOR_H_
#define TENSORFLOW_STREAM_EXECUTOR_LIB_STATUSOR_H_

#include <new>
#include "tensorflow/stream_executor/platform/port.h"
#include <type_traits>
#include <utility>

#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"

namespace perftools {
namespace gputools {
namespace port {

template<typename T>
class StatusOr {
  template<typename U> friend class StatusOr;

 public:
  // Construct a new StatusOr with Status::UNKNOWN status
  StatusOr() : status_(error::UNKNOWN, "") {}

  // Construct a new StatusOr with the given non-ok status. After calling
  // this constructor, calls to ValueOrDie() is invalid.
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  //
  // NOTE: Not explicit - we want to use StatusOr<T> as a return
  // value, so it is convenient and sensible to be able to do 'return
  // Status()' when the return type is StatusOr<T>.
  //
<<<<<<< HEAD
  // REQUIRES: !status.ok(). This requirement is DCHECKed.
  // In optimized builds, passing Status::OK() here will have the effect
  // of passing tensorflow::error::INTERNAL as a fallback.
  StatusOr(const Status& status);
  StatusOr& operator=(const Status& status);

  // TODO(b/62186997): Add operator=(T) overloads.

  // Similar to the `const T&` overload.
  //
  // REQUIRES: T is move constructible.
  StatusOr(T&& value);

  // RValue versions of the operations declared above.
  StatusOr(Status&& status);
  StatusOr& operator=(Status&& status);

  // Returns this->status().ok()
  bool ok() const { return this->status_.ok(); }

  // Returns a reference to our status. If this contains a T, then
  // returns Status::OK().
  const Status& status() const &;
  Status status() &&;

  // Returns a reference to our current value, or CHECK-fails if !this->ok().
  //
  // Note: for value types that are cheap to copy, prefer simple code:
  //
  //   T value = statusor.ValueOrDie();
  //
  // Otherwise, if the value type is expensive to copy, but can be left
  // in the StatusOr, simply assign to a reference:
  //
  //   T& value = statusor.ValueOrDie();  // or `const T&`
  //
  // Otherwise, if the value type supports an efficient move, it can be
  // used as follows:
  //
  //   T value = std::move(statusor).ValueOrDie();
  //
  // The std::move on statusor instead of on the whole expression enables
  // warnings about possible uses of the statusor object after the move.
  // C++ style guide waiver for ref-qualified overloads granted in cl/143176389
  // See go/ref-qualifiers for more details on such overloads.
  const T& ValueOrDie() const &;
  T& ValueOrDie() &;
  const T&& ValueOrDie() const &&;
  T&& ValueOrDie() &&;

  T ConsumeValueOrDie() { return std::move(ValueOrDie()); }

  // Ignores any errors. This method does nothing except potentially suppress
  // complaints from any tools that are checking that errors are not dropped on
  // the floor.
  void IgnoreError() const;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation details for StatusOr<T>

template <typename T>
StatusOr<T>::StatusOr() : Base(Status(tensorflow::error::UNKNOWN, "")) {}

template <typename T>
StatusOr<T>::StatusOr(const T& value) : Base(value) {}

template <typename T>
StatusOr<T>::StatusOr(const Status& status) : Base(status) {}

template <typename T>
StatusOr<T>& StatusOr<T>::operator=(const Status& status) {
  this->Assign(status);
  return *this;
}

template <typename T>
StatusOr<T>::StatusOr(T&& value) : Base(std::move(value)) {}

template <typename T>
StatusOr<T>::StatusOr(Status&& status) : Base(std::move(status)) {}

template <typename T>
StatusOr<T>& StatusOr<T>::operator=(Status&& status) {
  this->Assign(std::move(status));
  return *this;
}

template <typename T>
template <typename U,
          typename std::enable_if<std::is_convertible<U, T>::value>::type*>
inline StatusOr<T>::StatusOr(const StatusOr<U>& other)
    : Base(static_cast<const typename StatusOr<U>::Base&>(other)) {}

template <typename T>
template <typename U,
          typename std::enable_if<std::is_convertible<U, T>::value>::type*>
inline StatusOr<T>& StatusOr<T>::operator=(const StatusOr<U>& other) {
  if (other.ok())
    this->Assign(other.ValueOrDie());
  else
    this->Assign(other.status());
  return *this;
}

template <typename T>
template <typename U,
          typename std::enable_if<std::is_convertible<U, T>::value>::type*>
inline StatusOr<T>::StatusOr(StatusOr<U>&& other)
    : Base(static_cast<typename StatusOr<U>::Base&&>(other)) {}

template <typename T>
template <typename U,
          typename std::enable_if<std::is_convertible<U, T>::value>::type*>
inline StatusOr<T>& StatusOr<T>::operator=(StatusOr<U>&& other) {
  if (other.ok()) {
    this->Assign(std::move(other).ValueOrDie());
  } else {
    this->Assign(std::move(other).status());
  }
  return *this;
}

template <typename T>
const Status& StatusOr<T>::status() const & {
  return this->status_;
}
template <typename T>
Status StatusOr<T>::status() && {
  // Note that we copy instead of moving the status here so that
  // ~StatusOrData() can call ok() without invoking UB.
  return ok() ? Status::OK() : this->status_;
}

template <typename T>
const T& StatusOr<T>::ValueOrDie() const & {
  this->EnsureOk();
  return this->data_;
}

template <typename T>
T& StatusOr<T>::ValueOrDie() & {
  this->EnsureOk();
  return this->data_;
}

template <typename T>
const T&& StatusOr<T>::ValueOrDie() const && {
  this->EnsureOk();
  return std::move(this->data_);
}

template <typename T>
T&& StatusOr<T>::ValueOrDie() && {
  this->EnsureOk();
  return std::move(this->data_);
}

template <typename T>
void StatusOr<T>::IgnoreError() const {
  // no-op
}

}  // namespace port

#define TF_ASSERT_OK_AND_ASSIGN(lhs, rexpr)                             \
  TF_ASSERT_OK_AND_ASSIGN_IMPL(                                         \
      TF_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, \
      rexpr);

#define TF_ASSERT_OK_AND_ASSIGN_IMPL(statusor, lhs, rexpr)  \
  auto statusor = (rexpr);                                  \
  ASSERT_TRUE(statusor.status().ok()) << statusor.status(); \
  lhs = std::move(statusor.ValueOrDie())

#define TF_STATUS_MACROS_CONCAT_NAME(x, y) TF_STATUS_MACROS_CONCAT_IMPL(x, y)
#define TF_STATUS_MACROS_CONCAT_IMPL(x, y) x##y

#define TF_ASSIGN_OR_RETURN(lhs, rexpr) \
  TF_ASSIGN_OR_RETURN_IMPL(             \
      TF_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, rexpr)

#define TF_ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr) \
  auto statusor = (rexpr);                             \
  if (TF_PREDICT_FALSE(!statusor.ok())) {              \
    return statusor.status();                          \
  }                                                    \
  lhs = std::move(statusor.ValueOrDie())

}  // namespace stream_executor
=======
  // REQUIRES: status != Status::OK.
  // In optimized builds, passing Status::OK here will have the effect
  // of passing PosixErrorSpace::EINVAL as a fallback.
  StatusOr(const Status& status);  // NOLINT

  // Construct a new StatusOr with the given value. If T is a plain pointer,
  // value must not be NULL. After calling this constructor, calls to
  // ValueOrDie() will succeed, and calls to status() will return OK.
  //
  // NOTE: Not explicit - we want to use StatusOr<T> as a return type
  // so it is convenient and sensible to be able to do 'return T()'
  // when when the return type is StatusOr<T>.
  //
  // REQUIRES: if T is a plain pointer, value != NULL.
  // In optimized builds, passing a NULL pointer here will have
  // the effect of passing PosixErrorSpace::EINVAL as a fallback.
  StatusOr(const T& value);  // NOLINT

  // Conversion copy constructor, T must be copy constructible from U
  template <typename U>
  StatusOr(const StatusOr<U>& other)  // NOLINT
      : status_(other.status_),
        value_(other.value_) {}

  // Conversion assignment operator, T must be assignable from U
  template <typename U>
  StatusOr& operator=(const StatusOr<U>& other) {
    status_ = other.status_;
    value_ = other.value_;
    return *this;
  }

  // Rvalue-reference overloads of the other constructors and assignment
  // operators, to support move-only types and avoid unnecessary copying.
  StatusOr(T&& value);  // NOLINT

  // Move conversion operator to avoid unecessary copy.
  // T must be assignable from U.
  // Not marked with explicit so the implicit conversion can happen.
  template <typename U>
  StatusOr(StatusOr<U>&& other)  // NOLINT
      : status_(std::move(other.status_)),
        value_(std::move(other.value_)) {}

  // Move assignment opeartor to avoid unnecessary copy.
  // T must be assignable from U
  template <typename U>
  StatusOr& operator=(StatusOr<U>&& other) {
    status_ = std::move(other.status_);
    value_ = std::move(other.value_);
    return *this;
  }

  // Returns a reference to our status. If this contains a T, then
  // returns Status::OK.
  const Status& status() const { return status_; }

  // Returns this->status().ok()
  bool ok() const { return status_.ok(); }

  // Returns a reference to our current value, requires that this->ok().
  // If you need to initialize a T object from the stored value,
  // ConsumeValueOrDie() may be more efficient.
  const T& ValueOrDie() const;

  // Returns our current value, requires this->ok(). Use this if
  // you would otherwise want to say std::move(s.ValueOrDie()), for example
  // if you need to initialize a T object from the stored value and you don't
  // need subsequent access to the stored value. It uses T's move constructor,
  // if it has one, so it will work with move-only types, and will often be
  // more efficient than ValueOrDie, but may leave the stored value
  // in an arbitrary valid state.
  T ConsumeValueOrDie();

 private:
  Status status_;
  T value_;

  void CheckValueNotNull(const T& value);

  template <typename U>
  struct IsNull {
    // For non-pointer U, a reference can never be NULL.
    static inline bool IsValueNull(const U& t) { return false; }
  };

  template <typename U>
  struct IsNull<U*> {
    static inline bool IsValueNull(const U* t) { return t == NULL; }
  };
};

////////////////////////////////////////////////////////////////////////////////
// Implementation details for StatusOr<T>

template <typename T>
StatusOr<T>::StatusOr(const T& value)
    : status_(), value_(value) {
  CheckValueNotNull(value);
}

template <typename T>
const T& StatusOr<T>::ValueOrDie() const {
  assert(status_.ok());
  return value_;
}

template <typename T>
T StatusOr<T>::ConsumeValueOrDie() {
  assert(status_.ok());
  return std::move(value_);
}

template <typename T>
StatusOr<T>::StatusOr(const Status& status)
    : status_(status) {
  assert(!status.ok());
  if (status.ok()) {
    status_ =
        Status(error::INTERNAL,
               "Status::OK is not a valid constructor argument to StatusOr<T>");
  }
}

template <typename T>
StatusOr<T>::StatusOr(T&& value)
    : status_() {
  CheckValueNotNull(value);
  value_ = std::move(value);
}

template <typename T>
void StatusOr<T>::CheckValueNotNull(const T& value) {
  assert(!IsNull<T>::IsValueNull(value));
  if (IsNull<T>::IsValueNull(value)) {
    status_ =
        Status(error::INTERNAL,
               "NULL is not a valid constructor argument to StatusOr<T*>");
  }
}

}  // namespace port
}  // namespace gputools
}  // namespace perftools
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#endif  // TENSORFLOW_STREAM_EXECUTOR_LIB_STATUSOR_H_
