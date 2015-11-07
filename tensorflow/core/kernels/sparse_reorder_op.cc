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

#define EIGEN_USE_THREADS

#include <algorithm>
#include <numeric>
=======
#define EIGEN_USE_THREADS

#include <algorithm>
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#include <unordered_map>
#include <utility>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
<<<<<<< HEAD
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
=======
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/public/tensor.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

template <typename T>
class SparseReorderOp : public OpKernel {
 public:
  explicit SparseReorderOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_ind = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_ind.shape()),
                errors::InvalidArgument(
<<<<<<< HEAD
                    "Input indices should be a matrix but received shape ",
=======
                    "Input indices should be a matrix but received shape",
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                    input_ind.shape().DebugString()));

    const Tensor& input_val = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_val.shape()),
                errors::InvalidArgument(
<<<<<<< HEAD
                    "Input values should be a vector but received shape ",
=======
                    "Input values should be a vector but received shape",
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                    input_val.shape().DebugString()));

    const Tensor& input_shape_in = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_shape_in.shape()),
                errors::InvalidArgument(
<<<<<<< HEAD
                    "Input shape should be a vector but received shape ",
=======
                    "Input shape should be a vector but received shape",
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                    input_shape_in.shape().DebugString()));

    const TensorShape input_shape(input_shape_in.vec<int64>());

    gtl::InlinedVector<int64, 8> std_order(input_shape.dims());
    std::iota(std_order.begin(), std_order.end(), 0);

    // Check if the sparse tensor is already ordered correctly
<<<<<<< HEAD
    sparse::SparseTensor input_sp;
    OP_REQUIRES_OK(
        context, sparse::SparseTensor::Create(input_ind, input_val, input_shape,
                                              std_order, &input_sp));

    if (input_sp.IndicesValid().ok()) {
=======
    sparse::SparseTensor input_sp(input_ind, input_val, input_shape, std_order);

    if (input_sp.IndicesValid()) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      context->set_output(0, input_sp.indices());
      context->set_output(1, input_sp.values());
    } else {
      // Deep-copy the input Tensors, then reorder in-place
<<<<<<< HEAD
      sparse::SparseTensor reordered_sp;
      OP_REQUIRES_OK(context,
                     sparse::SparseTensor::Create(tensor::DeepCopy(input_ind),
                                                  tensor::DeepCopy(input_val),
                                                  input_shape, &reordered_sp));
=======
      sparse::SparseTensor reordered_sp(tensor::DeepCopy(input_ind),
                                        tensor::DeepCopy(input_val),
                                        input_shape);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      reordered_sp.Reorder<T>(std_order);
      context->set_output(0, reordered_sp.indices());
      context->set_output(1, reordered_sp.values());
    }
  }
};

#define REGISTER_KERNELS(type)                                            \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("SparseReorder").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SparseReorderOp<type>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS
}  // namespace tensorflow
