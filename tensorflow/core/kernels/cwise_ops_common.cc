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
#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {

BinaryOpShared::BinaryOpShared(OpKernelConstruction* ctx, DataType out,
                               DataType in)
    : OpKernel(ctx) {
<<<<<<< HEAD
#if !defined(INTEL_MKL) || !defined(ENABLE_MKL)
  OP_REQUIRES_OK(ctx, ctx->MatchSignature({in, in}, {out}));
#endif  // !INTEL_MKL || !ENABLE_MKL
=======
  OP_REQUIRES_OK(ctx, ctx->MatchSignature({in, in}, {out}));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

void BinaryOpShared::SetUnimplementedError(OpKernelContext* ctx) {
  ctx->SetStatus(errors::Unimplemented(
<<<<<<< HEAD
      "Broadcast between ", ctx->input(0).shape().DebugString(), " and ",
      ctx->input(1).shape().DebugString(), " is not supported yet."));
}

void BinaryOpShared::SetComputeError(OpKernelContext* ctx) {
  // For speed, errors during compute are caught only via boolean flag, with no
  // associated information.  This is sufficient for now, since the only binary
  // ops that have compute errors are integer division and mod, and the only
  // error they produce is zero division.
  const string& op = ctx->op_kernel().type_string();
  if ((op == "Div" || op == "Mod" || op == "FloorMod" || op == "FloorDiv") &&
      DataTypeIsInteger(ctx->op_kernel().input_type(0))) {
    ctx->CtxFailure(errors::InvalidArgument("Integer division by zero"));
  } else if ((op == "Pow") &&
             DataTypeIsInteger(ctx->op_kernel().input_type(0)) &&
             DataTypeIsSigned(ctx->op_kernel().input_type(1))) {
    ctx->CtxFailure(errors::InvalidArgument(
        "Integers to negative integer powers are not allowed"));
  } else {
    ctx->CtxFailure(
        errors::Internal("Unexpected error in binary operator "
                         "(only integer div and mod should have errors)"));
  }
}

BinaryOpShared::BinaryOpState::BinaryOpState(OpKernelContext* ctx)
    : in0(ctx->input(0)),
      in1(ctx->input(1)),
      bcast(BCast::FromShape(in0.shape()), BCast::FromShape(in1.shape())) {
  if (!bcast.IsValid()) {
    bool incompatible_shape_error;
    bool has_attr =
        TryGetNodeAttr(ctx->op_kernel().def(), "incompatible_shape_error",
                       &(incompatible_shape_error));
    if (has_attr && !incompatible_shape_error) {
      const string& op = ctx->op_kernel().type_string();
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
      result = (op == "NotEqual");
      return;
    }

    ctx->SetStatus(errors::InvalidArgument(
        "Incompatible shapes: ", in0.shape().DebugString(), " vs. ",
        in1.shape().DebugString()));
    return;
  }

  const TensorShape output_shape = BCast::ToShape(bcast.output_shape());
  out_num_elements = output_shape.num_elements();
  in0_num_elements = in0.NumElements();
  in1_num_elements = in1.NumElements();
  OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                          {0, 1}, 0, output_shape, &out));

  ndims = static_cast<int>(bcast.x_reshape().size());
=======
      "Broadcast between ", ctx->input(0).shape().ShortDebugString(), " and ",
      ctx->input(1).shape().ShortDebugString(), " is not supported yet."));
}

static BCast::Vec FromShape(const TensorShape& shape) {
  BCast::Vec ret;
  for (int i = 0; i < shape.dims(); ++i) ret.push_back(shape.dim_size(i));
  return ret;
}

static TensorShape ToShape(const BCast::Vec& vec) {
  TensorShape shape;
  for (auto elem : vec) shape.AddDim(elem);
  return shape;
}

BinaryOpShared::BinaryOpState::BinaryOpState(OpKernelContext* ctx)
    : bcast(FromShape(ctx->input(0).shape()),
            FromShape(ctx->input(1).shape())) {
  if (!bcast.IsValid()) {
    ctx->SetStatus(errors::InvalidArgument(
        "Incompatible shapes: ", ctx->input(0).shape().ShortDebugString(),
        " vs. ", ctx->input(1).shape().ShortDebugString()));
    return;
  }
  OP_REQUIRES_OK(ctx,
                 ctx->allocate_output(0, ToShape(bcast.output_shape()), &out));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

}  // namespace tensorflow
