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

=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
// See docs in ../ops/parse_ops.cc.

#include <algorithm>
#include "tensorflow/core/framework/op_kernel.h"
<<<<<<< HEAD
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/byte_order.h"
=======
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

namespace tensorflow {

template <typename T>
class DecodeRawOp : public OpKernel {
 public:
  explicit DecodeRawOp(OpKernelConstruction* context) : OpKernel(context) {
<<<<<<< HEAD
    OP_REQUIRES_OK(context, context->GetAttr("out_type", &out_type_));

    const bool host_is_little_endian = port::kLittleEndian;
    bool data_is_little_endian;
    OP_REQUIRES_OK(context,
                   context->GetAttr("little_endian", &data_is_little_endian));
    convert_data_endianness_ = host_is_little_endian != data_is_little_endian;
=======
    OP_REQUIRES_OK(context, context->GetAttr("little_endian", &little_endian_));
    OP_REQUIRES_OK(context, context->GetAttr("out_type", &out_type_));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }

  void Compute(OpKernelContext* context) override {
    const auto& input = context->input(0);
<<<<<<< HEAD
    int64 str_size = -1;
    auto flat_in = input.flat<tstring>();
    for (int64 i = 0; i < flat_in.size(); ++i) {
=======
    int str_size = -1;
    auto flat_in = input.flat<string>();
    for (int i = 0; i < flat_in.size(); ++i) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      const string& in_str = flat_in(i);
      if (str_size == -1) {
        str_size = in_str.size();
      } else {
        OP_REQUIRES(context, str_size == in_str.size(),
                    errors::InvalidArgument(
                        "DecodeRaw requires input strings to all be the same "
                        "size, but element ",
                        i, " has size ", str_size, " != ", in_str.size()));
      }
    }
    TensorShape out_shape = input.shape();
<<<<<<< HEAD
    if (str_size == -1 || str_size == 0) {  // Empty input
      out_shape.AddDim(0);
=======
    if (str_size == -1) {  // Empty input
      out_shape.AddDim(1);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output("output", out_shape,
                                                       &output_tensor));
      return;
    }
    OP_REQUIRES(
        context, str_size % sizeof(T) == 0,
        errors::InvalidArgument("Input to DecodeRaw has length ", str_size,
                                " that is not a multiple of ", sizeof(T),
                                ", the size of ", DataTypeString(out_type_)));
<<<<<<< HEAD
    const int64 added_dim = str_size / sizeof(T);
=======
    const int added_dim = str_size / sizeof(T);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    out_shape.AddDim(added_dim);
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output("output", out_shape, &output_tensor));
    auto out = output_tensor->flat_inner_dims<T>();
    DCHECK_EQ(flat_in.size(), out.dimensions()[0]);
<<<<<<< HEAD
    T* out_data = out.data();

    // If the data is already in the host's byte order, or if the width of the
    // output type is a single byte, we can copy the memory directly.
    if (!convert_data_endianness_ || sizeof(T) == 1) {
      for (int64 i = 0; i < flat_in.size(); ++i) {
        const T* in_data = reinterpret_cast<const T*>(flat_in(i).data());
        memcpy(out_data, in_data, str_size);
        out_data += added_dim;
      }
    } else {
      // Otherwise, the data is not in the host's byte order, and rather than a
      // direct copy, we need to reverse the byte ordering of each element.
      int64 element_size;
      if (out_type_ == DT_COMPLEX64 || out_type_ == DT_COMPLEX128) {
        // For Complex data type, real and imaginary parts need to be swapped
        // separately
        element_size = sizeof(T) / 2;
      } else {
        element_size = sizeof(T);
      }
      for (int64 i = 0; i < flat_in.size(); ++i) {
        const char* in_data_bytes =
            reinterpret_cast<const char*>(flat_in(i).data());
        char* out_data_bytes = reinterpret_cast<char*>(out_data);
        const char* p = in_data_bytes;
        char* q = out_data_bytes;
        for (; p < in_data_bytes + str_size;
             p += element_size, q += element_size) {
          std::reverse_copy(p, p + element_size, q);
        }
        out_data += added_dim;
      }
=======
    OP_REQUIRES(
        context,
        little_endian_ == ::tensorflow::port::kLittleEndian || sizeof(T) == 1,
        errors::Unimplemented("Unimplemented support for little_endian=",
                              little_endian_ ? "true" : "false"));
    // Endianness matches, so just copy each string byte-for-byte.
    T* out_data = out.data();
    for (int i = 0; i < flat_in.size(); ++i) {
      const T* in_data = reinterpret_cast<const T*>(flat_in(i).data());
      memcpy(out_data, in_data, str_size);
      out_data += added_dim;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    }
  }

 private:
<<<<<<< HEAD
  // True if the endianness of the data and the endianness of the host are
  // different, and the data needs conversion.
  bool convert_data_endianness_;

  // True if the input data is in little endian format.
  bool data_is_little_endian_;
=======
  bool little_endian_;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  DataType out_type_;
};

#define REGISTER(type)                                                       \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("DecodeRaw").Device(DEVICE_CPU).TypeConstraint<type>("out_type"), \
      DecodeRawOp<type>)

<<<<<<< HEAD
REGISTER(Eigen::half);
REGISTER(float);
REGISTER(double);
REGISTER(int32);
REGISTER(uint16);
=======
REGISTER(float);
REGISTER(double);
REGISTER(int32);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
REGISTER(uint8);
REGISTER(int16);
REGISTER(int8);
REGISTER(int64);
<<<<<<< HEAD
REGISTER(bool);
REGISTER(complex64);
REGISTER(complex128);
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#undef REGISTER

}  // namespace tensorflow
