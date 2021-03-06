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

// An example Op.

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

REGISTER_OP("Fact")
    .Output("fact: string")
    .SetShapeFn(tensorflow::shape_inference::UnknownShape);

class FactOp : public tensorflow::OpKernel {
 public:
  explicit FactOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    // Output a scalar string.
    tensorflow::Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, tensorflow::TensorShape(), &output_tensor));
    using tensorflow::string;
    auto output = output_tensor->template scalar<tensorflow::tstring>();
=======
// An example Op.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("Fact")
    .Output("fact: string")
    .Doc(R"doc(
Output a fact about factorials.
)doc");

class FactOp : public OpKernel {
 public:
  explicit FactOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Output a scalar string.
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape(), &output_tensor));
    auto output = output_tensor->template scalar<string>();
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

    output() = "0! == 1";
  }
};

<<<<<<< HEAD
REGISTER_KERNEL_BUILDER(Name("Fact").Device(tensorflow::DEVICE_CPU), FactOp);
=======
REGISTER_KERNEL_BUILDER(Name("Fact").Device(DEVICE_CPU), FactOp);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
