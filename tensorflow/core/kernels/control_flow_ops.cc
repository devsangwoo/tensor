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
#include "tensorflow/core/kernels/control_flow_ops.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
<<<<<<< HEAD
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

void SwitchOp::Compute(OpKernelContext* context) {
  const Tensor& outputPorts = context->input(1);
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(outputPorts.shape()),
              errors::InvalidArgument("The second input must be a scalar, "
                                      "but it has shape ",
                                      outputPorts.shape().DebugString()));

  bool pred = outputPorts.scalar<bool>()();
  int port = (pred) ? 1 : 0;
  if (context->input_is_ref(0)) {
    context->forward_ref_input_to_ref_output(0, port);
  } else {
    context->set_output(port, context->input(0));
  }
}

void SwitchNOp::Compute(OpKernelContext* context) {
  const Tensor& output_index_t = context->input(1);
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(output_index_t.shape()),
              errors::InvalidArgument("The second input must be a scalar, "
                                      "but it has shape ",
                                      output_index_t.shape().DebugString()));
  int output_index = output_index_t.scalar<int>()();
  if (output_index < 0 || output_index >= num_outputs()) {
    output_index = num_outputs() - 1;
  }
  context->set_output(output_index, context->input(0));
}

REGISTER_KERNEL_BUILDER(
    Name("Switch").Device(DEVICE_DEFAULT).HostMemory("pred"), SwitchOp);

REGISTER_KERNEL_BUILDER(
    Name("_SwitchN").Device(DEVICE_DEFAULT).HostMemory("output_index"),
    SwitchNOp);
=======
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {

// A switch op has two inputs and two outputs. It forwards the value of
// Input:0 to the output specified by input:1. Input:1 is a boolean tensor.
// Input:0 is forwarded to output:0 if input:1 is false, otherwise to
// output:1.
class SwitchOp : public OpKernel {
 public:
  explicit SwitchOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& outputPorts = context->input(1);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(outputPorts.shape()),
        errors::InvalidArgument("The second input must be a scalar, "
                                "but it has shape ",
                                outputPorts.shape().ShortDebugString()));

    bool pred = outputPorts.scalar<bool>()();
    int port = (pred) ? 1 : 0;
    if (IsRefType(context->input_dtype(0))) {
      context->forward_ref_input_to_ref_output(0, port);
    } else {
      context->set_output(port, context->input(0));
    }
  }

  bool IsExpensive() override { return false; }

  ~SwitchOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(SwitchOp);
};
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#define REGISTER_CPU_SWITCH(type)                         \
  REGISTER_KERNEL_BUILDER(Name("Switch")                  \
                              .Device(DEVICE_CPU)         \
                              .HostMemory("pred")         \
                              .TypeConstraint<type>("T"), \
<<<<<<< HEAD
                          SwitchOp)                       \
  REGISTER_KERNEL_BUILDER(Name("_SwitchN")                \
                              .Device(DEVICE_CPU)         \
                              .HostMemory("output_index") \
                              .TypeConstraint<type>("T"), \
                          SwitchNOp)
=======
                          SwitchOp)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#define REGISTER_CPU_REF_SWITCH(type)                     \
  REGISTER_KERNEL_BUILDER(Name("RefSwitch")               \
                              .Device(DEVICE_CPU)         \
                              .HostMemory("pred")         \
                              .TypeConstraint<type>("T"), \
                          SwitchOp)

#define REGISTER_GPU_SWITCH(type)                         \
  REGISTER_KERNEL_BUILDER(Name("Switch")                  \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("pred")         \
                              .TypeConstraint<type>("T"), \
<<<<<<< HEAD
                          SwitchOp)                       \
  REGISTER_KERNEL_BUILDER(Name("_SwitchN")                \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("output_index") \
                              .TypeConstraint<type>("T"), \
                          SwitchNOp)
=======
                          SwitchOp)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#define REGISTER_GPU_REF_SWITCH(type)                     \
  REGISTER_KERNEL_BUILDER(Name("RefSwitch")               \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("pred")         \
                              .TypeConstraint<type>("T"), \
                          SwitchOp)

TF_CALL_ALL_TYPES(REGISTER_CPU_SWITCH);
TF_CALL_ALL_TYPES(REGISTER_CPU_REF_SWITCH);
<<<<<<< HEAD
TF_CALL_QUANTIZED_TYPES(REGISTER_CPU_SWITCH);
TF_CALL_QUANTIZED_TYPES(REGISTER_CPU_REF_SWITCH);
REGISTER_CPU_SWITCH(uint64);

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_SWITCH);
TF_CALL_QUANTIZED_TYPES(REGISTER_GPU_SWITCH);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_REF_SWITCH);
TF_CALL_QUANTIZED_TYPES(REGISTER_GPU_REF_SWITCH);
REGISTER_GPU_SWITCH(uint64);
TF_CALL_variant(REGISTER_GPU_SWITCH);
=======

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_SWITCH);
REGISTER_GPU_SWITCH(bool);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_REF_SWITCH);
REGISTER_GPU_REF_SWITCH(int32);
REGISTER_GPU_REF_SWITCH(bool);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#undef REGISTER_CPU_SWITCH
#undef REGISTER_CPU_REF_SWITCH
#undef REGISTER_GPU_SWITCH
#undef REGISTER_GPU_REF_SWITCH

<<<<<<< HEAD
// Special GPU kernels for int32 and string.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
#define REGISTER_GPU_HOST_KERNEL(type)                    \
  REGISTER_KERNEL_BUILDER(Name("Switch")                  \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("pred")         \
                              .HostMemory("output_false") \
                              .HostMemory("output_true")  \
                              .TypeConstraint<type>("T"), \
                          SwitchOp)                       \
  REGISTER_KERNEL_BUILDER(Name("_SwitchN")                \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("output_index") \
                              .HostMemory("outputs")      \
                              .TypeConstraint<type>("T"), \
                          SwitchNOp)

#define REGISTER_GPU_HOST_REF_KERNEL(type)                \
  REGISTER_KERNEL_BUILDER(Name("RefSwitch")               \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("pred")         \
                              .HostMemory("output_false") \
                              .HostMemory("output_true")  \
                              .TypeConstraint<type>("T"), \
                          SwitchOp)

REGISTER_GPU_HOST_KERNEL(int32);
REGISTER_GPU_HOST_REF_KERNEL(int32);
REGISTER_GPU_HOST_KERNEL(bool);
REGISTER_GPU_HOST_REF_KERNEL(bool);
REGISTER_GPU_HOST_KERNEL(tstring);
REGISTER_GPU_HOST_REF_KERNEL(tstring);
REGISTER_GPU_HOST_KERNEL(ResourceHandle);

#undef REGISTER_GPU_HOST_KERNEL
#undef REGISTER_GPU_HOST_REF_KERNEL

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_SWITCH(type)                        \
  REGISTER_KERNEL_BUILDER(Name("Switch")                  \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("pred")         \
                              .TypeConstraint<type>("T"), \
                          SwitchOp)
TF_CALL_REAL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_SWITCH);

#define REGISTER_SYCL_REF_SWITCH(type)                    \
  REGISTER_KERNEL_BUILDER(Name("RefSwitch")               \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("pred")         \
                              .TypeConstraint<type>("T"), \
                          SwitchOp)
TF_CALL_REAL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_REF_SWITCH);

#undef REGISTER_SYCL_SWITCH
#undef REGISTER_SYCL_REF_SWITCH

#define REGISTER_SYCL_HOST_KERNEL(type)                   \
  REGISTER_KERNEL_BUILDER(Name("Switch")                  \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("data")         \
                              .HostMemory("pred")         \
                              .HostMemory("output_false") \
                              .HostMemory("output_true")  \
                              .TypeConstraint<type>("T"), \
                          SwitchOp)

REGISTER_SYCL_HOST_KERNEL(bool);
REGISTER_SYCL_HOST_KERNEL(tstring);
REGISTER_SYCL_HOST_KERNEL(int32);

#define REGISTER_SYCL_HOST_REF_KERNEL(type)               \
  REGISTER_KERNEL_BUILDER(Name("RefSwitch")               \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("data")         \
                              .HostMemory("pred")         \
                              .HostMemory("output_false") \
                              .HostMemory("output_true")  \
                              .TypeConstraint<type>("T"), \
                          SwitchOp)

REGISTER_SYCL_HOST_REF_KERNEL(int32);
REGISTER_SYCL_HOST_REF_KERNEL(bool);
REGISTER_SYCL_HOST_REF_KERNEL(tstring);

#undef REGISTER_SYCL_HOST_KERNEL
#undef REGISTER_SYCL_HOST_REF_KERNEL
#endif  // TENSORFLOW_USE_SYCL
=======
// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Switch")
                            .Device(DEVICE_GPU)
                            .HostMemory("data")
                            .HostMemory("pred")
                            .HostMemory("output_false")
                            .HostMemory("output_true")
                            .TypeConstraint<int32>("T"),
                        SwitchOp);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

class RefSelectOp : public OpKernel {
 public:
  explicit RefSelectOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("N", &num_ref_inputs_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& index_tensor = context->input(0);
<<<<<<< HEAD
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(index_tensor.shape()),
                errors::InvalidArgument("Index must be a scalar, "
                                        "but it has shape ",
                                        index_tensor.shape().DebugString()));
=======
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(index_tensor.shape()),
        errors::InvalidArgument("Index must be a scalar, "
                                "but it has shape ",
                                index_tensor.shape().ShortDebugString()));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

    int32 index = index_tensor.scalar<int32>()();

    OP_REQUIRES(context, index >= 0 && index < num_ref_inputs_,
                errors::InvalidArgument("Index must be in the range [0, ",
                                        num_ref_inputs_, ") but got ", index));
    context->forward_ref_input_to_ref_output(index + 1, 0);
  }

  bool IsExpensive() override { return false; }

  ~RefSelectOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(RefSelectOp);

 private:
  int num_ref_inputs_;
};

#define REGISTER_CPU_REF_SELECT(type)                     \
  REGISTER_KERNEL_BUILDER(Name("RefSelect")               \
                              .Device(DEVICE_CPU)         \
                              .HostMemory("index")        \
                              .TypeConstraint<type>("T"), \
                          RefSelectOp)
TF_CALL_ALL_TYPES(REGISTER_CPU_REF_SELECT);

#undef REGISTER_CPU_REF_SWITCH

<<<<<<< HEAD
MergeOp::MergeOp(OpKernelConstruction* context) : OpKernel(context) {
  const DataType dt = context->input_type(0);
  const int num_in = context->num_inputs();
  OP_REQUIRES_OK(context, context->MatchSignature(DataTypeVector(num_in, dt),
                                                  {dt, DT_INT32}));
}

void MergeOp::Compute(OpKernelContext* context) {
  bool input_seen = false;
  for (int i = 0; i < context->num_inputs(); ++i) {
    if (context->has_input(i)) {
      if (input_seen) {
        context->SetStatus(
            errors::Internal("Merge can not have more than one valid input."));
        return;
      }
      input_seen = true;

      if (IsRefType(context->input_dtype(i))) {
        context->forward_ref_input_to_ref_output(i, 0);
      } else {
        context->set_output(0, context->input(i));
      }
      Tensor* value_index = nullptr;
      OP_REQUIRES_OK(
          context, context->allocate_output(1, TensorShape({}), &value_index));
      value_index->scalar<int32>()() = i;
    }
  }
}

REGISTER_KERNEL_BUILDER(Name("Merge").Device(DEVICE_CPU), MergeOp);
REGISTER_KERNEL_BUILDER(
    Name("Merge").Device(DEVICE_DEFAULT).HostMemory("value_index"), MergeOp);
REGISTER_KERNEL_BUILDER(Name("RefMerge").Device(DEVICE_CPU), MergeOp);
=======
// A merge op has n inputs and two outputs. It forwards the value of the
// first input that becomes available to its first output, and the
// index of the first input to its second output.
class MergeOp : public OpKernel {
 public:
  explicit MergeOp(OpKernelConstruction* context) : OpKernel(context) {
    const DataType dt = context->input_type(0);
    const int num_in = context->num_inputs();
    OP_REQUIRES_OK(context, context->MatchSignature(DataTypeVector(num_in, dt),
                                                    {dt, DT_INT32}));
  }

  void Compute(OpKernelContext* context) override {
    bool input_seen = false;
    for (int i = 0; i < context->num_inputs(); ++i) {
      if (context->has_input(i)) {
        if (input_seen) {
          context->SetStatus(errors::Internal(
              "Merge can not have more than one valid input."));
          return;
        }
        input_seen = true;

        context->set_output(0, context->input(i));
        Tensor* value_index = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}),
                                                         &value_index));
        value_index->scalar<int32>()() = i;
      }
    }
  }

  bool IsExpensive() override { return false; }

  ~MergeOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(MergeOp);
};

REGISTER_KERNEL_BUILDER(Name("Merge").Device(DEVICE_CPU), MergeOp);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#define REGISTER_GPU_KERNEL(type)                         \
  REGISTER_KERNEL_BUILDER(Name("Merge")                   \
                              .Device(DEVICE_GPU)         \
                              .TypeConstraint<type>("T")  \
                              .HostMemory("value_index"), \
                          MergeOp);

<<<<<<< HEAD
#define REGISTER_GPU_REF_KERNEL(type)                     \
  REGISTER_KERNEL_BUILDER(Name("RefMerge")                \
                              .Device(DEVICE_GPU)         \
                              .TypeConstraint<type>("T")  \
                              .HostMemory("value_index"), \
                          MergeOp);

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_REF_KERNEL);
TF_CALL_QUANTIZED_TYPES(REGISTER_GPU_KERNEL);
TF_CALL_QUANTIZED_TYPES(REGISTER_GPU_REF_KERNEL);
REGISTER_GPU_KERNEL(bool);
REGISTER_GPU_REF_KERNEL(bool);
REGISTER_GPU_KERNEL(uint64);
TF_CALL_variant(REGISTER_GPU_KERNEL);

#undef REGISTER_GPU_KERNEL
#undef REGISTER_GPU_REF_KERNEL

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(type)                        \
  REGISTER_KERNEL_BUILDER(Name("Merge")                   \
                              .Device(DEVICE_SYCL)        \
                              .TypeConstraint<type>("T")  \
                              .HostMemory("value_index"), \
                          MergeOp);
REGISTER_SYCL_KERNEL(bool);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_KERNEL);

#define REGISTER_SYCL_REF_KERNEL(type)                    \
  REGISTER_KERNEL_BUILDER(Name("RefMerge")                \
                              .Device(DEVICE_SYCL)        \
                              .TypeConstraint<type>("T")  \
                              .HostMemory("value_index"), \
                          MergeOp);
REGISTER_SYCL_REF_KERNEL(bool);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_REF_KERNEL);

#undef REGISTER_SYCL_KERNEL
#undef REGISTER_SYCL_REF_KERNEL
#endif  // TENSORFLOW_USE_SYCL

// Special GPU kernels for int32 and string.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
#define REGISTER_GPU_HOST_KERNEL(type)                    \
  REGISTER_KERNEL_BUILDER(Name("Merge")                   \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("inputs")       \
                              .HostMemory("output")       \
                              .HostMemory("value_index")  \
                              .TypeConstraint<type>("T"), \
                          MergeOp);                       \
  REGISTER_KERNEL_BUILDER(Name("RefMerge")                \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("inputs")       \
                              .HostMemory("output")       \
                              .HostMemory("value_index")  \
                              .TypeConstraint<type>("T"), \
                          MergeOp)

REGISTER_GPU_HOST_KERNEL(int32);
REGISTER_GPU_HOST_KERNEL(tstring);
REGISTER_GPU_HOST_KERNEL(ResourceHandle);

#undef REGISTER_GPU_HOST_KERNEL

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_HOST_KERNEL(type)                   \
  REGISTER_KERNEL_BUILDER(Name("Merge")                   \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("inputs")       \
                              .HostMemory("output")       \
                              .HostMemory("value_index")  \
                              .TypeConstraint<type>("T"), \
                          MergeOp);                       \
  REGISTER_KERNEL_BUILDER(Name("RefMerge")                \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("inputs")       \
                              .HostMemory("output")       \
                              .HostMemory("value_index")  \
                              .TypeConstraint<type>("T"), \
                          MergeOp)

REGISTER_SYCL_HOST_KERNEL(int32);
REGISTER_SYCL_HOST_KERNEL(tstring);
REGISTER_SYCL_HOST_KERNEL(ResourceHandle);

#undef REGISTER_SYCL_HOST_KERNEL
#endif  // TENSORFLOW_USE_SYCL

void EnterOp::Compute(OpKernelContext* context) {
  if (IsRefType(context->input_dtype(0))) {
    context->forward_ref_input_to_ref_output(0, 0);
  } else {
    context->set_output(0, context->input(0));
  }
}

REGISTER_KERNEL_BUILDER(Name("Enter").Device(DEVICE_DEFAULT), EnterOp);
=======
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);

#undef REGISTER_GPU_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Merge")
                            .Device(DEVICE_GPU)
                            .HostMemory("inputs")
                            .HostMemory("output")
                            .HostMemory("value_index")
                            .TypeConstraint<int32>("T"),
                        MergeOp);

// An enter op has one input and one output. It creates or finds
// the child frame that is uniquely identified by the frame_name,
// and makes its input available to the child frame.
class EnterOp : public OpKernel {
 public:
  explicit EnterOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    if (IsRefType(context->input_dtype(0))) {
      context->forward_ref_input_to_ref_output(0, 0);
    } else {
      context->set_output(0, context->input(0));
    }
  }

  bool IsExpensive() override { return false; }

  ~EnterOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(EnterOp);
};

REGISTER_KERNEL_BUILDER(Name("Enter").Device(DEVICE_CPU), EnterOp);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
REGISTER_KERNEL_BUILDER(Name("RefEnter").Device(DEVICE_CPU), EnterOp);

#define REGISTER_GPU_KERNEL(type) \
  REGISTER_KERNEL_BUILDER(        \
<<<<<<< HEAD
      Name("Enter").Device(DEVICE_GPU).TypeConstraint<type>("T"), EnterOp)
#define REGISTER_GPU_REF_KERNEL(type) \
  REGISTER_KERNEL_BUILDER(            \
      Name("RefEnter").Device(DEVICE_GPU).TypeConstraint<type>("T"), EnterOp)

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_REF_KERNEL);
REGISTER_GPU_KERNEL(bool);
REGISTER_GPU_REF_KERNEL(bool);
TF_CALL_variant(REGISTER_GPU_KERNEL);
=======
      Name("Enter").Device(DEVICE_GPU).TypeConstraint<type>("T"), EnterOp);
#define REGISTER_GPU_REF_KERNEL(type) \
  REGISTER_KERNEL_BUILDER(            \
      Name("RefEnter").Device(DEVICE_GPU).TypeConstraint<type>("T"), EnterOp);

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_NUMBER_TYPES(REGISTER_GPU_REF_KERNEL);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#undef REGISTER_GPU_KERNEL
#undef REGISTER_GPU_REF_KERNEL

<<<<<<< HEAD
#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(type) \
  REGISTER_KERNEL_BUILDER(         \
      Name("Enter").Device(DEVICE_SYCL).TypeConstraint<type>("T"), EnterOp)
REGISTER_SYCL_KERNEL(bool);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_KERNEL);

#define REGISTER_SYCL_REF_KERNEL(type) \
  REGISTER_KERNEL_BUILDER(             \
      Name("RefEnter").Device(DEVICE_SYCL).TypeConstraint<type>("T"), EnterOp)
REGISTER_SYCL_REF_KERNEL(bool);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_REF_KERNEL);

#undef REGISTER_SYCL_KERNEL
#undef REGISTER_SYCL_REF_KERNEL
#define REGISTER_SYCL_HOST_KERNEL(type)                   \
  REGISTER_KERNEL_BUILDER(Name("Enter")                   \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          EnterOp)

#define REGISTER_SYCL_HOST_REF_KERNEL(type)               \
  REGISTER_KERNEL_BUILDER(Name("RefEnter")                \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          EnterOp)

REGISTER_SYCL_HOST_KERNEL(int32);
REGISTER_SYCL_HOST_REF_KERNEL(int32);
REGISTER_SYCL_HOST_KERNEL(tstring);
REGISTER_SYCL_HOST_REF_KERNEL(tstring);
REGISTER_SYCL_HOST_KERNEL(ResourceHandle);

#undef REGISTER_SYCL_HOST_KERNEL
#undef REGISTER_SYCL_HOST_REF_KERNEL
#endif  // TENSORFLOW_USE_SYCL

// Special GPU kernels for int32 and string.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
#define REGISTER_GPU_HOST_KERNEL(type)                    \
  REGISTER_KERNEL_BUILDER(Name("Enter")                   \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          EnterOp)

#define REGISTER_GPU_HOST_REF_KERNEL(type)                \
  REGISTER_KERNEL_BUILDER(Name("RefEnter")                \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          EnterOp)

REGISTER_GPU_HOST_KERNEL(int32);
REGISTER_GPU_HOST_REF_KERNEL(int32);
REGISTER_GPU_HOST_KERNEL(tstring);
REGISTER_GPU_HOST_REF_KERNEL(tstring);
REGISTER_GPU_HOST_KERNEL(ResourceHandle);

#undef REGISTER_GPU_HOST_KERNEL
#undef REGISTER_GPU_HOST_REF_KERNEL

void ExitOp::Compute(OpKernelContext* context) {
  if (IsRefType(context->input_dtype(0))) {
    context->forward_ref_input_to_ref_output(0, 0);
  } else {
    context->set_output(0, context->input(0));
  }
}

REGISTER_KERNEL_BUILDER(Name("Exit").Device(DEVICE_DEFAULT), ExitOp);
REGISTER_KERNEL_BUILDER(Name("RefExit").Device(DEVICE_CPU), ExitOp);
=======
// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Enter")
                            .Device(DEVICE_GPU)
                            .HostMemory("data")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        EnterOp);

// An exit op has one input and one output. It exits the current
// frame to its parent frame, and makes its input available to the
// parent frame.
class ExitOp : public OpKernel {
 public:
  explicit ExitOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    context->set_output(0, context->input(0));
  }

  bool IsExpensive() override { return false; }

  ~ExitOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(ExitOp);
};

REGISTER_KERNEL_BUILDER(Name("Exit").Device(DEVICE_CPU), ExitOp);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#define REGISTER_GPU_KERNEL(type) \
  REGISTER_KERNEL_BUILDER(        \
      Name("Exit").Device(DEVICE_GPU).TypeConstraint<type>("T"), ExitOp);
<<<<<<< HEAD
#define REGISTER_GPU_REF_KERNEL(type) \
  REGISTER_KERNEL_BUILDER(            \
      Name("RefExit").Device(DEVICE_GPU).TypeConstraint<type>("T"), ExitOp);

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_REF_KERNEL);
REGISTER_GPU_KERNEL(bool);
REGISTER_GPU_REF_KERNEL(bool);
TF_CALL_variant(REGISTER_GPU_KERNEL);

#undef REGISTER_GPU_KERNEL
#undef REGISTER_GPU_REF_KERNEL

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(type)                                         \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Exit").Device(DEVICE_SYCL).TypeConstraint<type>("T"), ExitOp); \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("RefExit").Device(DEVICE_SYCL).TypeConstraint<type>("T"), ExitOp);
REGISTER_SYCL_KERNEL(bool);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_KERNEL);

#undef REGISTER_SYCL_KERNEL
#undef REGISTER_SYCL_REF_KERNEL

#define REGISTER_SYCL_HOST_KERNEL(type)                   \
  REGISTER_KERNEL_BUILDER(Name("Exit")                    \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          ExitOp);                        \
  REGISTER_KERNEL_BUILDER(Name("RefExit")                 \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          ExitOp)

REGISTER_SYCL_HOST_KERNEL(int32);
REGISTER_SYCL_HOST_KERNEL(tstring);
#undef REGISTER_SYCL_HOST_KERNEL
#endif  // TENSORFLOW_USE_SYCL

// Special GPU kernels for int32 and string.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
#define REGISTER_GPU_HOST_KERNEL(type)                    \
  REGISTER_KERNEL_BUILDER(Name("Exit")                    \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          ExitOp);                        \
  REGISTER_KERNEL_BUILDER(Name("RefExit")                 \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          ExitOp)

REGISTER_GPU_HOST_KERNEL(int32);
REGISTER_GPU_HOST_KERNEL(tstring);
REGISTER_GPU_HOST_KERNEL(ResourceHandle);

#undef REGISTER_GPU_HOST_KERNEL

void NextIterationOp::Compute(OpKernelContext* context) {
  if (IsRefType(context->input_dtype(0))) {
    context->forward_ref_input_to_ref_output(0, 0);
  } else {
    context->set_output(0, context->input(0));
  }
}

REGISTER_KERNEL_BUILDER(Name("NextIteration").Device(DEVICE_DEFAULT),
                        NextIterationOp);
REGISTER_KERNEL_BUILDER(Name("RefNextIteration").Device(DEVICE_CPU),
                        NextIterationOp);

#define REGISTER_GPU_KERNEL(type)                                            \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("NextIteration").Device(DEVICE_GPU).TypeConstraint<type>("T"),    \
      NextIterationOp);                                                      \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("RefNextIteration").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      NextIterationOp)

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
REGISTER_GPU_KERNEL(bool);
TF_CALL_variant(REGISTER_GPU_KERNEL);

#undef REGISTER_GPU_KERNEL

// Special GPU kernels for int32 and string.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
#define REGISTER_GPU_HOST_KERNEL(type)                    \
  REGISTER_KERNEL_BUILDER(Name("NextIteration")           \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          NextIterationOp);               \
  REGISTER_KERNEL_BUILDER(Name("RefNextIteration")        \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          NextIterationOp)

REGISTER_GPU_HOST_KERNEL(int32);
REGISTER_GPU_HOST_KERNEL(tstring);
REGISTER_GPU_HOST_KERNEL(ResourceHandle);

#undef REGISTER_GPU_HOST_KERNEL

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(type)                                            \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("NextIteration").Device(DEVICE_SYCL).TypeConstraint<type>("T"),    \
      NextIterationOp);                                                       \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("RefNextIteration").Device(DEVICE_SYCL).TypeConstraint<type>("T"), \
      NextIterationOp)
REGISTER_SYCL_KERNEL(bool);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_KERNEL);

#undef REGISTER_SYCL_KERNEL

#define REGISTER_SYCL_HOST_KERNEL(type)                   \
  REGISTER_KERNEL_BUILDER(Name("NextIteration")           \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          NextIterationOp);               \
  REGISTER_KERNEL_BUILDER(Name("RefNextIteration")        \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          NextIterationOp)

REGISTER_SYCL_HOST_KERNEL(int32);
REGISTER_SYCL_HOST_KERNEL(tstring);
#undef REGISTER_SYCL_HOST_KERNEL
#endif  // TENSORFLOW_USE_SYCL

LoopCondOp::LoopCondOp(OpKernelConstruction* context) : OpKernel(context) {}
LoopCondOp::~LoopCondOp() = default;

void LoopCondOp::Compute(OpKernelContext* context) {
  CancellationManager* cm = context->cancellation_manager();
  if (cm != nullptr) {
    bool already_cancelled = cm->IsCancelled();
    OP_REQUIRES(context, !already_cancelled,
                errors::Cancelled("Loop execution was cancelled."));
  }

  context->set_output(0, context->input(0));
}

bool LoopCondOp::IsExpensive() { return false; }

REGISTER_KERNEL_BUILDER(Name("LoopCond").Device(DEVICE_CPU), LoopCondOp);
REGISTER_KERNEL_BUILDER(Name("LoopCond")
                            .Device(DEVICE_DEFAULT)
=======

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);

#undef REGISTER_GPU_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Exit")
                            .Device(DEVICE_GPU)
                            .HostMemory("data")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        ExitOp);

// A next_iteration op has one input and one output. It makes its input
// available to the next iteration.
class NextIterationOp : public OpKernel {
 public:
  explicit NextIterationOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    context->set_output(0, context->input(0));
  }

  bool IsExpensive() override { return false; }

  ~NextIterationOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(NextIterationOp);
};

REGISTER_KERNEL_BUILDER(Name("NextIteration").Device(DEVICE_CPU),
                        NextIterationOp);

#define REGISTER_GPU_KERNEL(type)                                         \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("NextIteration").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      NextIterationOp);

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);

#undef REGISTER_GPU_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("NextIteration")
                            .Device(DEVICE_GPU)
                            .HostMemory("data")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        NextIterationOp);

// A LoopCond op has one input and one output. The input is a boolean
// scalar representing the taken branches of the "pivot" Switch that
// determines loop termination. As a contract, any high-level front-end
// should always use port '0' of the "pivot" switches for loop exit.
class LoopCondOp : public OpKernel {
 public:
  explicit LoopCondOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    context->set_output(0, context->input(0));
  }

  bool IsExpensive() override { return false; }

  ~LoopCondOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(LoopCondOp);
};

REGISTER_KERNEL_BUILDER(Name("LoopCond").Device(DEVICE_CPU), LoopCondOp);
REGISTER_KERNEL_BUILDER(Name("LoopCond")
                            .Device(DEVICE_GPU)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                            .HostMemory("input")
                            .HostMemory("output"),
                        LoopCondOp);

<<<<<<< HEAD
// ControlTrigger kernel
REGISTER_KERNEL_BUILDER(Name("ControlTrigger").Device(DEVICE_DEFAULT),
                        ControlTriggerOp);

// When called, abort op will abort the current process. This can be used to
// abort remote PSs when needed.
class AbortOp : public OpKernel {
 public:
  explicit AbortOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("error_msg", &error_msg_));
    OP_REQUIRES_OK(
        context, context->GetAttr("exit_without_error", &exit_without_error_));
  }

  void Compute(OpKernelContext* context) override {
    if (!exit_without_error_) {
      LOG(FATAL) << "Abort_op intentional failure; " << error_msg_;
    } else {
      LOG(WARNING) << "Exiting the process: " << error_msg_;
      exit(0);
    }
  }

 private:
  string error_msg_;
  bool exit_without_error_;
};

REGISTER_KERNEL_BUILDER(Name("Abort").Device(DEVICE_CPU), AbortOp);
=======
// ControlTrigger kernels
REGISTER_KERNEL_BUILDER(Name("ControlTrigger").Device(DEVICE_CPU),
                        ControlTriggerOp);

REGISTER_KERNEL_BUILDER(Name("ControlTrigger").Device(DEVICE_GPU),
                        ControlTriggerOp);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

}  // namespace tensorflow
