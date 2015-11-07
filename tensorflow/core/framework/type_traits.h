<<<<<<< HEAD
/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_TYPE_TRAITS_H_
#define TENSORFLOW_CORE_FRAMEWORK_TYPE_TRAITS_H_
=======
#ifndef TENSORFLOW_FRAMEWORK_TYPE_TRAITS_H_
#define TENSORFLOW_FRAMEWORK_TYPE_TRAITS_H_
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#include <limits>
#include <utility>

<<<<<<< HEAD
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/platform/types.h"
=======
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/port.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

namespace tensorflow {

// Functions to define quantization attribute of types.
struct true_type {
  static const bool value = true;
};
struct false_type {
  static const bool value = false;
};

// Default is_quantized is false.
template <typename T>
struct is_quantized : false_type {};

// Specialize the quantized types.
template <>
struct is_quantized<qint8> : true_type {};
template <>
struct is_quantized<quint8> : true_type {};
template <>
struct is_quantized<qint32> : true_type {};
<<<<<<< HEAD
template <>
struct is_quantized<qint16> : true_type {};
template <>
struct is_quantized<quint16> : true_type {};

// Default is_complex is false.
template <typename T>
struct is_complex : false_type {};

// Specialize std::complex<float> and std::complex<double> types.
template <>
struct is_complex<std::complex<float>> : true_type {};
template <>
struct is_complex<std::complex<double>> : true_type {};

// is_simple_type<T>::value if T[] can be safely constructed and destructed
// without running T() and ~T().  We do not use std::is_trivial<T>
// directly because std::complex<float> and std::complex<double> are
// not trivial, but their arrays can be constructed and destructed
// without running their default ctors and dtors.
template <typename T>
struct is_simple_type {
  static constexpr bool value =
      std::is_trivial<T>::value || std::is_same<T, Eigen::half>::value ||
      std::is_same<T, complex64>::value || std::is_same<T, complex128>::value ||
      is_quantized<T>::value || std::is_same<T, bfloat16>::value;
=======

// All types not specialized are marked invalid.
template <class T>
struct IsValidDataType {
  static constexpr bool value = false;
};

// Extra validity checking; not part of public API.
struct TestIsValidDataType {
  static_assert(IsValidDataType<int64>::value, "Incorrect impl for int64");
  static_assert(IsValidDataType<int32>::value, "Incorrect impl for int32");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
};

}  // namespace tensorflow

// Define numeric limits for our quantized as subclasses of the
// standard types.
namespace std {
template <>
class numeric_limits<tensorflow::qint8>
    : public numeric_limits<tensorflow::int8> {};
template <>
class numeric_limits<tensorflow::quint8>
    : public numeric_limits<tensorflow::uint8> {};
template <>
<<<<<<< HEAD
class numeric_limits<tensorflow::qint16>
    : public numeric_limits<tensorflow::int16> {};
template <>
class numeric_limits<tensorflow::quint16>
    : public numeric_limits<tensorflow::uint16> {};
template <>
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
class numeric_limits<tensorflow::qint32>
    : public numeric_limits<tensorflow::int32> {};

// Specialize is_signed for quantized types.
template <>
struct is_signed<tensorflow::qint8> : public is_signed<tensorflow::int8> {};
template <>
struct is_signed<tensorflow::quint8> : public is_signed<tensorflow::uint8> {};
template <>
<<<<<<< HEAD
struct is_signed<tensorflow::qint16> : public is_signed<tensorflow::int16> {};
template <>
struct is_signed<tensorflow::quint16> : public is_signed<tensorflow::uint16> {};
template <>
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
struct is_signed<tensorflow::qint32> : public is_signed<tensorflow::int32> {};

}  // namespace std

<<<<<<< HEAD
#endif  // TENSORFLOW_CORE_FRAMEWORK_TYPE_TRAITS_H_
=======
#endif  // TENSORFLOW_FRAMEWORK_TYPE_TRAITS_H_
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
