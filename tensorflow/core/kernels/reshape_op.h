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

#ifndef TENSORFLOW_CORE_KERNELS_RESHAPE_OP_H_
#define TENSORFLOW_CORE_KERNELS_RESHAPE_OP_H_
=======
#ifndef TENSORFLOW_KERNELS_RESHAPE_OP_H_
#define TENSORFLOW_KERNELS_RESHAPE_OP_H_
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#include <memory>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
<<<<<<< HEAD
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

// Note that this op is subclassed for QuantizedReshapeOp.
=======
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

namespace tensorflow {

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
class ReshapeOp : public OpKernel {
 public:
  explicit ReshapeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& sizes = context->input(1);
    // Preliminary validation of sizes.
<<<<<<< HEAD
    OP_REQUIRES(context, IsLegacyVector(sizes.shape()),
                errors::InvalidArgument("sizes input must be 1-D, not ",
                                        sizes.shape().DebugString()));
=======
    OP_REQUIRES(context, TensorShapeUtils::IsLegacyVector(sizes.shape()),
                errors::InvalidArgument("sizes input must be 1-D, not shape ",
                                        sizes.shape().ShortDebugString()));
    const int64 num_dims = sizes.NumElements();
    OP_REQUIRES(
        context, num_dims <= 8,
        errors::InvalidArgument(num_dims, " > max 8 output dims supported"));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

    // Compute the output shape.  Determine product of specified
    // dimensions, and find the index of the unspecified one.
    TensorShape shape;
<<<<<<< HEAD
    int64 product = 1;
    int unknown_index = -1;
    bool sizes_has_zero_dim;
    switch (sizes.dtype()) {
      case DT_INT32:
        OP_REQUIRES_OK(context,
                       ValidateSizes<int32>(sizes, &product, &unknown_index,
                                            &shape, &sizes_has_zero_dim));
        break;
      case DT_INT64:
        OP_REQUIRES_OK(context,
                       ValidateSizes<int64>(sizes, &product, &unknown_index,
                                            &shape, &sizes_has_zero_dim));
        break;
      default:
        context->CtxFailure(errors::InvalidArgument(
            "desired shape must be a DT_INT32 or DT_INT64 vector, not a ",
            DataTypeString(sizes.dtype())));
        return;
    }
    if (unknown_index != -1) {
      int64 input_num_elements = 1;
      bool input_has_zero_dim = false;
      for (int dim = 0; dim < input.dims(); dim++) {
        // For zero dimension, we don't count it into `input_num_elements`
        // unless `sizes` has no zero dimension, so we are still able to
        // infer shapes for other dimensions.
        if (input.dim_size(dim) > 0 || !sizes_has_zero_dim) {
          input_num_elements *= input.dim_size(dim);
        } else {
          input_has_zero_dim = true;
        }
      }

      const int64 missing = input_num_elements / product;
      if (!input_has_zero_dim) {
        OP_REQUIRES(
            context, product * missing == input_num_elements,
            errors::InvalidArgument(
                "Input to reshape is a tensor with ", input_num_elements,
                " values, but the requested shape requires a multiple of ",
                product));
      }
      shape.set_dim(unknown_index, missing);
    }
    OP_REQUIRES(context, shape.num_elements() == input.NumElements(),
                errors::InvalidArgument("Input to reshape is a tensor with ",
                                        input.NumElements(),
                                        " values, but the requested shape has ",
=======
    int32 product = 1;
    int unknown_index = -1;
    auto Svec = sizes.flat<int32>();
    for (int d = 0; d < num_dims; ++d) {
      const int32 size = Svec(d);
      if (size == -1) {
        OP_REQUIRES(
            context, unknown_index == -1,
            errors::InvalidArgument("only one input size may be -1, not both ",
                                    unknown_index, " and ", d));
        unknown_index = d;
        shape.AddDim(1);
      } else {
        OP_REQUIRES(context, size >= 0,
                    errors::InvalidArgument(
                        "size ", d, " must be non-negative, not ", size));
        shape.AddDim(size);
        product *= size;
      }
    }
    if (unknown_index != -1) {
      OP_REQUIRES(
          context, product > 0,
          errors::InvalidArgument("cannot infer the missing input size for "
                                  "an empty tensor unless all specified "
                                  "input sizes are non-zero"));
      const int32 missing = input.NumElements() / product;
      OP_REQUIRES(context, product * missing == input.NumElements(),
                  errors::InvalidArgument("Input has ", input.NumElements(),
                                          " values, which isn't divisible by ",
                                          product));
      shape.set_dim(unknown_index, missing);
    }
    OP_REQUIRES(context, shape.num_elements() == input.NumElements(),
                errors::InvalidArgument("Input has ", input.NumElements(),
                                        " values, which isn't the same as ",
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                                        shape.num_elements()));

    // Actually produce the reshaped output.
    Tensor output(input.dtype());
    CHECK(output.CopyFrom(input, shape));
    context->set_output(0, output);
  }

  bool IsExpensive() override { return false; }
<<<<<<< HEAD

 private:
  template <typename Tshape>
  Status ValidateSizes(const Tensor& sizes, int64* product, int* unknown_index,
                       TensorShape* shape, bool* has_zero_dim) {
    *product = 1;
    *unknown_index = -1;
    *has_zero_dim = false;
    const int64 num_dims = sizes.NumElements();
    auto Svec = sizes.flat<Tshape>();
    for (int d = 0; d < num_dims; ++d) {
      const Tshape size = Svec(d);
      if (size == -1) {
        if (*unknown_index != -1) {
          return errors::InvalidArgument(
              "Only one input size may be -1, not both ", *unknown_index,
              " and ", d);
        }
        *unknown_index = d;
        shape->AddDim(1);
      } else if (size < 0) {
        return errors::InvalidArgument("Size ", d,
                                       " must be non-negative, not ", size);
      } else if (size == 0) {
        // We don't include zero-sized dimension in product, so that we can
        // still calculate number of elements for non-zero-sized dimensions and
        // therefore infer their shapes.
        shape->AddDim(size);
        *has_zero_dim = true;
      } else {
        shape->AddDim(size);
        (*product) *= size;
      }
    }
    return Status::OK();
  }
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
};

}  // namespace tensorflow

<<<<<<< HEAD
#endif  // TENSORFLOW_CORE_KERNELS_RESHAPE_OP_H_
=======
#endif  // TENSORFLOW_KERNELS_RESHAPE_OP_H_
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
