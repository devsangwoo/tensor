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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

=======
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
REGISTER_OP("AddN")
    .Input("inputs: N * T")
    .Output("sum: T")
    .Attr("N: int >= 1")
<<<<<<< HEAD
    .Attr("T: {numbertype, variant}")
    .SetIsCommutative()
    .SetIsAggregate()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle cur = c->input(c->num_inputs() - 1);
      for (int i = c->num_inputs() - 2; i >= 0; --i) {
        TF_RETURN_WITH_CONTEXT_IF_ERROR(c->Merge(c->input(i), cur, &cur),
                                        "From merging shape ", i,
                                        " with other shapes.");
      }
      c->set_output(0, cur);

      DataType dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("T", &dtype));

      if (dtype != DT_VARIANT) {
        // Exit early if not DT_VARIANT.
        return Status::OK();
      } else {
        // DT_VARIANT shape handle shape inference.  All sizes and dtypes must
        // be the same; all shapes must be compatible via Merge.
        std::vector<shape_inference::ShapeAndType> cur_shapes_and_types;
        auto* shapes_and_types =
            c->input_handle_shapes_and_types(c->num_inputs() - 1);
        if (shapes_and_types) {
          cur_shapes_and_types = *shapes_and_types;
        }

        for (int i = c->num_inputs() - 2; i >= 0; --i) {
          auto shapes_and_types_i = c->input_handle_shapes_and_types(i);
          if (!shapes_and_types && shapes_and_types_i) {
            // TODO(ebrevdo): Find cases where this happens and fix their shape
            // inference.  If we are calling AddN on variant types, they should
            // all have consistent shape_and_type info.
            shapes_and_types = shapes_and_types_i;
          } else if (shapes_and_types && shapes_and_types_i) {
            if (shapes_and_types_i->size() != shapes_and_types->size()) {
              return errors::InvalidArgument(
                  "shapes_and_types[", i,
                  "].size() == ", shapes_and_types_i->size(),
                  " != shapes_and_types[0].size() == ",
                  shapes_and_types->size());
            }
            for (int j = 0; j < shapes_and_types->size(); ++j) {
              if (shapes_and_types->at(j).dtype !=
                  shapes_and_types_i->at(j).dtype) {
                return errors::InvalidArgument(
                    "shapes_and_types[", i, "][", j, "].dtype() == ",
                    DataTypeString(shapes_and_types_i->at(j).dtype),
                    " != shapes_and_types[0][", j, "].dtype == ",
                    DataTypeString(shapes_and_types->at(j).dtype));
              }
              TF_RETURN_WITH_CONTEXT_IF_ERROR(
                  c->Merge(shapes_and_types_i->at(j).shape,
                           cur_shapes_and_types.at(j).shape,
                           &cur_shapes_and_types.at(j).shape),
                  "From merging shapes_and_types[", i, "][", j, "].shape with ",
                  "shapes_and_types[0][", j, "].shape");
            }
          }
        }
        if (shapes_and_types) {
          c->set_output_handle_shapes_and_types(0, cur_shapes_and_types);
        }
        return Status::OK();
      }
    });

// --------------------------------------------------------------------------

// Note that the following operator is just a placeholder and has no
// associated kernel. The code in accumulate_n_optimizer.cc replaces
// this placeholder with a graph of operators that do have kernels.
// The Python code that generates instances of this op is currently in
// contrib/framework/python/ops/accumulate_n_v2.py
REGISTER_OP("AccumulateNV2")
    .Input("inputs: N * T")
    .Output("sum: T")
    .Attr("N: int >= 1")
    .Attr("T: numbertype")
    .Attr("shape: shape")
    .SetIsCommutative()
    .SetIsAggregate()
    .SetShapeFn(shape_inference::ExplicitShape);
=======
    .Attr("T: numbertype")
    .SetIsCommutative()
    .SetIsAggregate()
    .Doc(R"doc(
Add all input tensors element wise.

inputs: Must all be the same size and shape.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

// --------------------------------------------------------------------------

REGISTER_OP("BatchMatMul")
    .Input("x: T")
    .Input("y: T")
<<<<<<< HEAD
    .Output("output: T")
    .Attr(
        "T: {bfloat16, half, float, double, int32, int64, complex64, "
        "complex128}")
    .Attr("adj_x: bool = false")
    .Attr("adj_y: bool = false")
    .SetShapeFn(shape_inference::BatchMatMulShape);

REGISTER_OP("BatchMatMulV2")
    .Input("x: T")
    .Input("y: T")
    .Output("output: T")
    .Attr(
        "T: {bfloat16, half, float, double, int32, int64, complex64, "
        "complex128}")
    .Attr("adj_x: bool = false")
    .Attr("adj_y: bool = false")
    .SetShapeFn(shape_inference::BatchMatMulV2Shape);

#ifdef INTEL_MKL
REGISTER_OP("_MklBatchMatMul")
    .Input("x: T")
    .Input("y: T")
    .Output("output: T")
    .Attr(
        "T: {bfloat16, half, float, double, int32, int64, complex64, "
        "complex128}")
    .Attr("adj_x: bool = false")
    .Attr("adj_y: bool = false")
    .SetShapeFn(shape_inference::BatchMatMulShape);

REGISTER_OP("_MklBatchMatMulV2")
    .Input("x: T")
    .Input("y: T")
    .Output("output: T")
    .Attr(
        "T: {bfloat16, half, float, double, int32, int64, complex64, "
        "complex128}")
    .Attr("adj_x: bool = false")
    .Attr("adj_y: bool = false")
    .SetShapeFn(shape_inference::BatchMatMulV2Shape);
#endif  // INTEL_MKL
=======
    .Output("out: T")
    .Attr("T: {float, double, int32, complex64}")
    .Attr("adj_x: bool = false")
    .Attr("adj_y: bool = false")
    .Doc(R"doc(
Multiplies slices of two tensors in batches.

Multiplies all slices of `Tensor` `x` and `y` (each slice can be
viewed as an element of a batch), and arranges the individual results
in a single output tensor of the same batch size. Each of the
individual slices can optionally be adjointed (to adjoint a matrix
means to transpose and conjugate it) before multiplication by setting
the `adj_x` or `adj_y` flag to `True`, which are by default `False`.

The input tensors `x` and `y` are 3-D or higher with shape `[..., r_x, c_x]`
and `[..., r_y, c_y]`.

The output tensor is 3-D or higher with shape `[..., r_o, c_o]`, where:

    r_o = c_x if adj_x else r_x
    c_o = r_y if adj_y else c_y

It is computed as:

    out[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])

x: 3-D or higher with shape `[..., r_x, c_x]`.
y: 3-D or higher with shape `[..., r_y, c_y]`.
out: 3-D or higher with shape `[..., r_o, c_o]`
adj_x: If `True`, adjoint the slices of `x`. Defaults to `False`.
adj_y: If `True`, adjoint the slices of `y`. Defaults to `False`.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

// --------------------------------------------------------------------------
// Casting Ops
//
// NOTE: Only a smaller number of types are supported by
// Cast. The exact casting rule is TBD. The current
// implementation uses C++ static cast rules for numeric
// types, which may be changed in the future.
REGISTER_OP("Cast")
    .Input("x: SrcT")
    .Output("y: DstT")
    .Attr("SrcT: type")
    .Attr("DstT: type")
<<<<<<< HEAD
    .Attr("Truncate: bool = false")
    .SetShapeFn(shape_inference::UnchangedShape);
=======
    .Doc(R"doc(
Cast x of type SrcT to y of DstT.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("_HostCast")
    .Input("x: SrcT")
    .Output("y: DstT")
    .Attr("SrcT: type")
    .Attr("DstT: type")
<<<<<<< HEAD
    .Attr("Truncate: bool = false")
    .SetShapeFn(shape_inference::UnchangedShape)
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    .Doc(R"doc(
Cast x of type SrcT to y of DstT.

_HostCast requires its input and produces its output in host memory.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("Abs")
    .Input("x: T")
    .Output("y: T")
<<<<<<< HEAD
    .Attr("T: {bfloat16, half, float, double, int8, int16, int32, int64}")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("ComplexAbs")
    .Input("x: T")
    .Output("y: Tout")
    .Attr("T: {complex64, complex128} = DT_COMPLEX64")
    .Attr("Tout: {float, double} = DT_FLOAT")
    .SetShapeFn(shape_inference::UnchangedShape);

// Declares cwise unary operations signature: 't -> 't
#define UNARY()                                                          \
  Input("x: T")                                                          \
      .Output("y: T")                                                    \
      .Attr(                                                             \
          "T: {bfloat16, half, float, double, int32, int64, complex64, " \
          "complex128}")                                                 \
      .SetShapeFn(shape_inference::UnchangedShape)

#define UNARY_REAL()                              \
  Input("x: T")                                   \
      .Output("y: T")                             \
      .Attr("T: {bfloat16, half, float, double}") \
      .SetShapeFn(shape_inference::UnchangedShape)

#define UNARY_COMPLEX()                                                  \
  Input("x: T")                                                          \
      .Output("y: T")                                                    \
      .Attr("T: {bfloat16, half, float, double, complex64, complex128}") \
      .SetShapeFn(shape_inference::UnchangedShape)

#define UNARY_GRADIENT_COMPLEX()                                         \
  Input("y: T")                                                          \
      .Input("dy: T")                                                    \
      .Output("z: T")                                                    \
      .Attr("T: {bfloat16, half, float, double, complex64, complex128}") \
      .SetShapeFn(shape_inference::UnchangedShape)

REGISTER_OP("Neg").UNARY();

REGISTER_OP("Inv").UNARY();

REGISTER_OP("InvGrad").UNARY_GRADIENT_COMPLEX();

REGISTER_OP("Reciprocal").UNARY();

REGISTER_OP("ReciprocalGrad").UNARY_GRADIENT_COMPLEX();

REGISTER_OP("Square").UNARY();

REGISTER_OP("Sqrt").UNARY_COMPLEX();

REGISTER_OP("SqrtGrad").UNARY_GRADIENT_COMPLEX();

REGISTER_OP("Rsqrt").UNARY_COMPLEX();

REGISTER_OP("Round").UNARY();

REGISTER_OP("RsqrtGrad").UNARY_GRADIENT_COMPLEX();

REGISTER_OP("Exp").UNARY_COMPLEX();

REGISTER_OP("Expm1").UNARY_COMPLEX();

REGISTER_OP("Log").UNARY_COMPLEX();

REGISTER_OP("Log1p").UNARY_COMPLEX();

REGISTER_OP("Sinh").UNARY_COMPLEX();

REGISTER_OP("Cosh").UNARY_COMPLEX();

REGISTER_OP("Tanh").UNARY_COMPLEX();

REGISTER_OP("Asinh").UNARY_COMPLEX();

REGISTER_OP("Acosh").UNARY_COMPLEX();

REGISTER_OP("Atanh").UNARY_COMPLEX();

REGISTER_OP("TanhGrad").UNARY_GRADIENT_COMPLEX();

REGISTER_OP("Lgamma").UNARY_REAL();

REGISTER_OP("Digamma").UNARY_REAL();

REGISTER_OP("Erf").UNARY_REAL();
REGISTER_OP("Erfinv").UNARY_REAL();
REGISTER_OP("Ndtri").UNARY_REAL();
REGISTER_OP("Erfc").UNARY_REAL();

REGISTER_OP("Sigmoid").UNARY_COMPLEX();

REGISTER_OP("SigmoidGrad").UNARY_GRADIENT_COMPLEX();

REGISTER_OP("Sin").UNARY_COMPLEX();

REGISTER_OP("Cos").UNARY_COMPLEX();

REGISTER_OP("Tan").UNARY();

REGISTER_OP("Asin").UNARY();

REGISTER_OP("Acos").UNARY();

REGISTER_OP("Atan").UNARY();

REGISTER_OP("BesselI0e").UNARY_REAL();

REGISTER_OP("BesselI1e").UNARY_REAL();

REGISTER_OP("_UnaryOpsComposition")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {float, half, double}")
    .Attr("op_names: list(string)")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
*NOTE*: Do not invoke this operator directly in Python. Graph rewrite pass is
expected to create these operators.
)doc");

#undef UNARY
#undef UNARY_REAL
#undef UNARY_COMPLEX
=======
    .Attr("T: {float, double, int32, int64}")
    .Doc(R"doc(
Computes the absolute value of a tensor.

Given a tensor `x`, this operation returns a tensor containing the absolute
value of each element in `x`. For example, if x is an input element and y is
an output element, this operation computes \\(y = |x|\\).
)doc");

REGISTER_OP("ComplexAbs")
    .Input("x: complex64")
    .Output("y: float")
    .Doc(R"doc(
Computes the complex absolute value of a tensor.

Given a tensor `x` of complex numbers, this operation returns a tensor of type
`float` that is the absolute value of each element in `x`. All elements in `x`
must be complex numbers of the form \\(a + bj\\). The absolute value is
computed as \\( \sqrt{a^2 + b^2}\\).

For example:

```
# tensor 'x' is [[-2.25 + 4.75j], [-3.25 + 5.75j]]
tf.complex_abs(x) ==> [5.25594902, 6.60492229]
```
)doc");

// Declares cwise unary operations signature: 't -> 't
#define UNARY()                      \
  Input("x: T").Output("y: T").Attr( \
      "T: {float, double, int32, complex64, int64}")

REGISTER_OP("Neg")
    .UNARY()
    .Doc(R"doc(
Computes numerical negative value element-wise.
I.e., \\(y = -x\\).
)doc");

REGISTER_OP("Inv")
    .UNARY()
    .Doc(R"doc(
Computes the reciprocal of x element-wise.
I.e., \\(y = 1 / x\\).
)doc");

REGISTER_OP("Square")
    .UNARY()
    .Doc(R"doc(
Computes square of x element-wise.
I.e., \\(y = x * x = x^2\\).
)doc");

REGISTER_OP("Sqrt")
    .UNARY()
    .Doc(R"doc(
Computes square root of x element-wise.
I.e., \\(y = \sqrt{x} = x^{1/2}\\).
)doc");

REGISTER_OP("Rsqrt")
    .UNARY()
    .Doc(R"doc(
Computes reciprocal of square root of x element-wise.
I.e., \\(y = 1 / \sqrt{x}\\).
)doc");

REGISTER_OP("Exp")
    .UNARY()
    .Doc(R"doc(
Computes exponential of x element-wise.  \\(y = e^x\\).
)doc");

REGISTER_OP("Log")
    .UNARY()
    .Doc(R"doc(
Computes natural logrithm of x element-wise.
I.e., \\(y = \log_e x\\).
)doc");

REGISTER_OP("Tanh")
    .UNARY()
    .Doc(R"doc(
Computes hyperbolic tangent of `x` element-wise.
)doc");

REGISTER_OP("Sigmoid")
    .UNARY()
    .Doc(R"doc(
Computes sigmoid of `x` element-wise.

Specifically, `y = 1 / (1 + exp(-x))`.
)doc");

REGISTER_OP("Sin")
    .UNARY()
    .Doc(R"doc(
Computes sin of x element-wise.
)doc");

REGISTER_OP("Cos")
    .UNARY()
    .Doc(R"doc(
Computes cos of x element-wise.
)doc");

#undef UNARY
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("IsNan")
    .Input("x: T")
    .Output("y: bool")
<<<<<<< HEAD
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);
=======
    .Attr("T: {float, double}")
    .Doc(R"doc(
Returns which elements of x are NaN.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("IsInf")
    .Input("x: T")
    .Output("y: bool")
<<<<<<< HEAD
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);
=======
    .Attr("T: {float, double}")
    .Doc(R"doc(
Returns which elements of x are Inf.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("IsFinite")
    .Input("x: T")
    .Output("y: bool")
<<<<<<< HEAD
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);
=======
    .Attr("T: {float, double}")
    .Doc(R"doc(
Returns which elements of x are finite.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("Sign")
    .Input("x: T")
    .Output("y: T")
<<<<<<< HEAD
    .Attr(
        "T: {bfloat16, half, float, double, int32, int64, complex64, "
        "complex128}")
    .SetShapeFn(shape_inference::UnchangedShape);
=======
    .Attr("T: {float, double, int32, int64}")
    .Doc(R"doc(
Returns an element-wise indication of the sign of a number.

y = sign(x) = -1 if x < 0; 0 if x == 0; 1 if x > 0.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("Floor")
    .Input("x: T")
    .Output("y: T")
<<<<<<< HEAD
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);
=======
    .Attr("T: {float, double}")
    .Doc(R"doc(
Returns element-wise largest integer not greater than x.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("Ceil")
    .Input("x: T")
    .Output("y: T")
<<<<<<< HEAD
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("Rint")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

// Declares cwise binary operations signature: 't, 't -> 't.

#define BINARY_MORE()                                                          \
  Input("x: T").Input("y: T").Output("z: T").Attr(                             \
      "T: {bfloat16, half, float, double, uint8, int8, uint16, int16, int32, " \
      "int64, complex64, complex128}")

#define BINARY_FEWER()                                               \
  Input("x: T").Input("y: T").Output("z: T").Attr(                   \
      "T: {bfloat16, half, float, double, int32, int64, complex64, " \
      "complex128}")

REGISTER_OP("Add")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr(
        "T: {bfloat16, half, float, double, uint8, int8, int16, int32, int64, "
        "complex64, complex128, string}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("AddV2")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr(
        "T: {bfloat16, half, float, double, uint8, int8, int16, int32, int64, "
        "complex64, complex128}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn)
    .SetIsAggregate()
    .SetIsCommutative();

#ifdef INTEL_MKL
REGISTER_OP("_MklAdd")
    .Input("x: T")
    .Input("y: T")
    .Input("mkl_x: uint8")
    .Input("mkl_y: uint8")
    .Output("z: T")
    .Output("mkl_z: uint8")
    .Attr(
        "T: {half, float, double, uint8, int8, int16, int32, int64, complex64, "
        "complex128, string, bfloat16}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn)
    .Doc(R"doc(
Returns `x` + `y` element-wise.

*NOTE*: `tf.math.add` supports broadcasting. `tf.math.add_n` does not. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
)doc");

REGISTER_OP("_MklAddV2")
    .Input("x: T")
    .Input("y: T")
    .Input("mkl_x: uint8")
    .Input("mkl_y: uint8")
    .Output("z: T")
    .Output("mkl_z: uint8")
    .Attr(
        "T: {bfloat16, half, float, double, uint8, int8, int16, int32, int64, "
        "complex64, complex128}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn)
    .SetIsAggregate()
    .SetIsCommutative()
    .Doc(R"doc(
Returns `x` + `y` element-wise.
*NOTE*: `tf.math.add` supports broadcasting. `tf.math.add_n` does not. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
)doc");
#endif  // INTEL_MKL

REGISTER_OP("Sub").BINARY_MORE().SetShapeFn(
    shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("_MklSub")
    .BINARY_FEWER()
    .Input("mkl_x: uint8")
    .Input("mkl_y: uint8")
    .Output("mkl_z: uint8")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn)
    .Doc(R"doc(
Returns x - y element-wise.

*NOTE*: `Sub` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
)doc");

REGISTER_OP("Mul").BINARY_MORE().SetIsCommutative().SetShapeFn(
    shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("MulNoNan")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {half, float, double, complex64, complex128}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

// Note: This op is not commutative w.r.t. to all its inputs.
REGISTER_OP("_MklMul")
    .BINARY_MORE()
    .Input("mkl_x: uint8")
    .Input("mkl_y: uint8")
    .Output("mkl_z: uint8")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn)
    .Doc(R"doc(
Returns x * y element-wise.

*NOTE*: `Mul` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
)doc");

REGISTER_OP("Div").BINARY_MORE().SetShapeFn(
    shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("DivNoNan")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {half, float, double, complex64, complex128}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("FloorDiv")
    .BINARY_MORE()
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("TruncateDiv")
    .BINARY_MORE()
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("RealDiv").BINARY_MORE().SetShapeFn(
    shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("SquaredDifference")
    .BINARY_FEWER()
    .SetIsCommutative()
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

// Note: This op is not commutative w.r.t. to all its inputs.
REGISTER_OP("_MklSquaredDifference")
    .BINARY_FEWER()
    .Input("mkl_x: uint8")
    .Input("mkl_y: uint8")
    .Output("mkl_z: uint8")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn)
    .Doc(R"doc(
Returns (x - y)(x - y) element-wise.

*NOTE*: `SquaredDifference` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
)doc");

REGISTER_OP("Xlogy")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {half, float, double, complex64, complex128}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("Xlog1py")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {half, float, double, complex64, complex128}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("Xdivy")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {half, float, double, complex64, complex128}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

=======
    .Attr("T: {float, double}")
    .Doc(R"doc(
Returns element-wise smallest integer in not less than x.
)doc");

// Declares cwise binary operations signature: 't, 't -> 't.

#define BINARY_MORE()                              \
  Input("x: T").Input("y: T").Output("z: T").Attr( \
      "T: {float, double, int8, int16, int32, complex64, int64}")

#define BINARY_FEWER()                             \
  Input("x: T").Input("y: T").Output("z: T").Attr( \
      "T: {float, double, int32, complex64, int64}")

REGISTER_OP("Add")
    .BINARY_MORE()
    .SetIsCommutative()
    .Doc(R"doc(
Returns x + y element-wise.

*NOTE*: Add supports broadcasting. AddN does not.
)doc");

REGISTER_OP("Sub")
    .BINARY_FEWER()
    .Doc(R"doc(
Returns x - y element-wise.
)doc");

REGISTER_OP("Mul")
    .BINARY_MORE()
    .SetIsCommutative()
    .Doc(R"doc(
Returns x * y element-wise.
)doc");

REGISTER_OP("Div")
    .BINARY_FEWER()
    .Doc(R"doc(
Returns x / y element-wise.
)doc");

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#undef BINARY_FEWER
#undef BINARY_MORE

REGISTER_OP("Maximum")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
<<<<<<< HEAD
    .Attr("T: {bfloat16, half, float, double, int32, int64}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

// Note: This op is not commutative w.r.t. to all its inputs.
REGISTER_OP("_MklMaximum")
    .Input("x: T")
    .Input("y: T")
    .Input("mkl_x: uint8")
    .Input("mkl_y: uint8")
    .Output("z: T")
    .Output("mkl_z: uint8")
    .Attr("T: {half, float, double, int32, int64, bfloat16}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn)
    .Doc(R"doc(
Returns the max of x and y (i.e. x > y ? x : y) element-wise.

*NOTE*: `Maximum` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
=======
    .Attr("T: {float, double, int32, int64}")
    .SetIsCommutative()
    .Doc(R"doc(
Returns the max of x and y (i.e. x > y ? x : y) element-wise, broadcasts.
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
)doc");

REGISTER_OP("Minimum")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
<<<<<<< HEAD
    .Attr("T: {bfloat16, half, float, double, int32, int64}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);
=======
    .Attr("T: {float, double, int32, int64}")
    .SetIsCommutative()
    .Doc(R"doc(
Returns the min of x and y (i.e. x < y ? x : y) element-wise, broadcasts.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("Mod")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
<<<<<<< HEAD
    .Attr("T: {int32, int64, float16, half, bfloat16, float, double}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("FloorMod")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {int32, int64, bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("TruncateMod")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {int32, int64, bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);
=======
    .Attr("T: {int32, int64, float, double}")
    .Doc(R"doc(
Returns element-wise remainder of division.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("Pow")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
<<<<<<< HEAD
    .Attr(
        "T: {bfloat16, float, half, double, int32, int64, complex64, "
        "complex128}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("Igammac")
    .Input("a: T")
    .Input("x: T")
    .Output("z: T")
    .Attr("T: {float, double}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("Igamma")
    .Input("a: T")
    .Input("x: T")
    .Output("z: T")
    .Attr("T: {float, double}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("IgammaGradA")
    .Input("a: T")
    .Input("x: T")
    .Output("z: T")
    .Attr("T: {float, double}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("Zeta")
    .Input("x: T")
    .Input("q: T")
    .Output("z: T")
    .Attr("T: {float, double}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("Polygamma")
    .Input("a: T")
    .Input("x: T")
    .Output("z: T")
    .Attr("T: {float, double}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("Atan2")
    .Input("y: T")
    .Input("x: T")
    .Output("z: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("Betainc")
    .Input("a: T")
    .Input("b: T")
    .Input("x: T")
    .Output("z: T")
    .Attr("T: {float, double}")
    .SetShapeFn([](InferenceContext* c) {
      const int num_inputs = 3;
      ShapeHandle output = c->UnknownShape();
      int num_scalars = 0;
      ShapeHandle some_non_scalar;
      for (int i = 0; i < num_inputs; ++i) {
        ShapeHandle in = c->input(i);
        if (!c->RankKnown(in)) {
          some_non_scalar = in;
          // An input with unknown rank could be either a scalar (to be
          // broadcast) or some other shape.
        } else if (c->Rank(in) == 0) {
          // Input is a scalar, it will be broadcast to the output shape.
          ++num_scalars;
        } else {
          TF_RETURN_IF_ERROR(c->Merge(output, in, &output));
          some_non_scalar = output;
        }
      }

      if (num_scalars == num_inputs - 1) {
        // If all but one input is known to be a scalar, then output is the
        // remaining input.
        output = some_non_scalar;
      } else if (num_scalars == num_inputs) {
        // If all are scalars, output is scalar; pick the first one arbitrarily.
        output = c->input(0);
      }

      c->set_output(0, output);
      return Status::OK();
    });
=======
    .Attr("T: {float, double, int32, complex64, int64}")
    .Doc(R"doc(
Computes the power of one value to another.

Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
corresponding elements in `x` and `y`. For example:

```
# tensor 'x' is [[2, 2]], [3, 3]]
# tensor 'y' is [[8, 16], [2, 3]]
tf.pow(x, y) ==> [[256, 65536], [9, 27]]
```
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

// --------------------------------------------------------------------------

// Declares cwise binary comparison operations signature: 't, 't -> bool,
// where 't has a natural total order.
<<<<<<< HEAD
#define COMPARISON()             \
  Input("x: T")                  \
      .Input("y: T")             \
      .Output("z: bool")         \
      .Attr("T: realnumbertype") \
      .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn)

REGISTER_OP("Less").COMPARISON();

REGISTER_OP("LessEqual").COMPARISON();

REGISTER_OP("Greater").COMPARISON();

REGISTER_OP("GreaterEqual").COMPARISON();
=======
#define COMPARISON()                                  \
  Input("x: T").Input("y: T").Output("z: bool").Attr( \
      "T: {float, double, int32, int64}")

REGISTER_OP("Less")
    .COMPARISON()
    .Doc(R"doc(
Returns the truth value of (x < y) element-wise.
)doc");

REGISTER_OP("LessEqual")
    .COMPARISON()
    .Doc(R"doc(
Returns the truth value of (x <= y) element-wise.
)doc");

REGISTER_OP("Greater")
    .COMPARISON()
    .Doc(R"doc(
Returns the truth value of (x > y) element-wise.
)doc");

REGISTER_OP("GreaterEqual")
    .COMPARISON()
    .Doc(R"doc(
Returns the truth value of (x >= y) element-wise.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#undef COMPARISON

// --------------------------------------------------------------------------

<<<<<<< HEAD
#define EQUALITY_COMPARISON()                                              \
  Input("x: T")                                                            \
      .Input("y: T")                                                       \
      .Output("z: bool")                                                   \
      .SetIsCommutative()                                                  \
      .Attr(                                                               \
          "T: {bfloat16, half, float, double, uint8, int8, int16, int32, " \
          "int64, complex64, quint8, qint8, qint32, string, bool, "        \
          "complex128}")                                                   \
      .Attr("incompatible_shape_error: bool = true")                       \
      .SetShapeFn([](InferenceContext* c) {                                \
        ShapeHandle x = c->input(0);                                       \
        ShapeHandle y = c->input(1);                                       \
        ShapeHandle output;                                                \
        bool incompatible_shape_error;                                     \
        TF_RETURN_IF_ERROR(c->GetAttr("incompatible_shape_error",          \
                                      &incompatible_shape_error));         \
        TF_RETURN_IF_ERROR(BroadcastBinaryOpOutputShapeFnHelper(           \
            c, x, y, incompatible_shape_error, &output));                  \
        c->set_output(0, output);                                          \
        return Status::OK();                                               \
      })

REGISTER_OP("Equal").EQUALITY_COMPARISON();

REGISTER_OP("NotEqual").EQUALITY_COMPARISON();

#undef EQUALITY_COMPARISON

REGISTER_OP("ApproximateEqual")
    .Input("x: T")
    .Input("y: T")
    .Output("z: bool")
    .SetIsCommutative()
    .Attr("T: numbertype")
    .Attr("tolerance: float = 0.00001")
    .SetShapeFn([](InferenceContext* c) {
      // The inputs 'x' and 'y' must have the same shape.
      ShapeHandle data_x = c->input(0);
      ShapeHandle data_y = c->input(1);
      TF_RETURN_IF_ERROR(c->Merge(data_x, data_y, &data_x));
      return shape_inference::UnchangedShape(c);
    });
=======
#define COMPARISON()                                                     \
  Input("x: T").Input("y: T").Output("z: bool").SetIsCommutative().Attr( \
      "T: {float, double, int32, int64, complex64, quint8, qint8, qint32}")

REGISTER_OP("Equal")
    .COMPARISON()
    .Doc(R"doc(
Returns the truth value of (x == y) element-wise.
)doc");

REGISTER_OP("NotEqual")
    .COMPARISON()
    .Doc(R"doc(
Returns the truth value of (x != y) element-wise.
)doc");

#undef COMPARISON
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

// --------------------------------------------------------------------------

REGISTER_OP("LogicalNot")
    .Input("x: bool")
    .Output("y: bool")
<<<<<<< HEAD
    .SetShapeFn(shape_inference::UnchangedShape);

#define BINARY_LOGICAL()  \
  Input("x: bool")        \
      .Input("y: bool")   \
      .Output("z: bool")  \
      .SetIsCommutative() \
      .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn)

REGISTER_OP("LogicalAnd").BINARY_LOGICAL();

REGISTER_OP("LogicalOr").BINARY_LOGICAL();
=======
    .Doc(R"doc(
Returns the truth value of NOT x element-wise.
)doc");

#define BINARY_LOGICAL() \
  Input("x: bool").Input("y: bool").Output("z: bool").SetIsCommutative()

REGISTER_OP("LogicalAnd")
    .BINARY_LOGICAL()
    .Doc(R"doc(
Returns the truth value of x AND y element-wise.
)doc");

REGISTER_OP("LogicalOr")
    .BINARY_LOGICAL()
    .Doc(R"doc(
Returns the truth value of x OR y element-wise.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#undef BINARY_LOGICAL

// --------------------------------------------------------------------------

REGISTER_OP("Select")
    .Input("condition: bool")
    .Input("t: T")
    .Input("e: T")
<<<<<<< HEAD
    .Output("output: T")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      auto* handle_data_1 = c->input_handle_shapes_and_types(1);
      auto* handle_data_2 = c->input_handle_shapes_and_types(2);
      // Merge handle shape and dtype if applicable.
      if (handle_data_1 != nullptr && handle_data_2 != nullptr) {
        const auto size = handle_data_1->size();
        std::vector<shape_inference::ShapeAndType> merged_handle_data(size);
        if (size != handle_data_2->size()) {
          return errors::InvalidArgument(
              "Trying to merge handles pointing to different numbers of "
              "tensors.");
        }

        for (int i = 0; i < size; ++i) {
          const shape_inference::ShapeAndType& s1 = (*handle_data_1)[i];
          const shape_inference::ShapeAndType& s2 = (*handle_data_2)[i];
          if (s1.dtype != s2.dtype) {
            // TODO(apassos) resolve this in the manner of b/32476923
            return errors::InvalidArgument(
                "Trying to merge handles pointing to different dtypes.");
          }
          merged_handle_data[i].dtype = s1.dtype;
          TF_RETURN_IF_ERROR(
              c->Merge(s1.shape, s2.shape, &merged_handle_data[i].shape));
        }

        c->set_output_handle_shapes_and_types(0, merged_handle_data);
      }

      // The inputs 'then' and 'else' must have the same shape.
      ShapeHandle data = c->input(1);
      ShapeHandle other = c->input(2);
      TF_RETURN_IF_ERROR(c->Merge(data, other, &data));

      // The input 'cond' must either have the same shape as 'then' and
      // 'else', or be a vector if 'then' and 'else' are at least vectors.
      ShapeHandle cond = c->input(0);

      if (!c->RankKnown(cond) || !c->RankKnown(data)) {
        c->set_output(0, data);
        return Status::OK();
      }

      // rank of shape and data is known.

      const int32 cond_rank = c->Rank(cond);
      const int32 data_rank = c->Rank(data);

      if (cond_rank == 0) {
        // The rank of 'cond' is a scalar.
        // t and e can have any shape.
        c->set_output(0, data);
        return Status::OK();
      }

      if (cond_rank != 1) {
        // If 'cond' is not a vector, and not a scalar,
        // then shape must match 'then' and 'else'
        TF_RETURN_IF_ERROR(c->Merge(data, cond, &data));
        c->set_output(0, data);
        return Status::OK();
      }

      if (data_rank == 0) {
        // if 'then' and 'else' are scalar also the cond must be
        TF_RETURN_IF_ERROR(c->Merge(data, cond, &data));
        c->set_output(0, data);
        return Status::OK();
      }

      if (cond_rank == 1) {
        // if the cond is a vector and the 'then' is not a scalar,
        // the first dimension of 'then' and 'else'
        TF_RETURN_IF_ERROR(c->Merge(cond, c->Vector(c->Dim(data, 0)), &cond));
        c->set_output(0, data);
        return Status::OK();
      }

      c->set_output(0, data);

      return Status::OK();
    });

REGISTER_OP("SelectV2")
    .Input("condition: bool")
    .Input("t: T")
    .Input("e: T")
    .Output("output: T")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      auto* handle_data_1 = c->input_handle_shapes_and_types(1);
      auto* handle_data_2 = c->input_handle_shapes_and_types(2);
      // Merge handle shape and dtype if applicable.
      if (handle_data_1 != nullptr && handle_data_2 != nullptr) {
        const auto size = handle_data_1->size();
        std::vector<shape_inference::ShapeAndType> merged_handle_data(size);
        if (size != handle_data_2->size()) {
          return errors::InvalidArgument(
              "Trying to merge handles pointing to different numbers of "
              "tensors.");
        }

        for (int i = 0; i < size; ++i) {
          const shape_inference::ShapeAndType& s1 = (*handle_data_1)[i];
          const shape_inference::ShapeAndType& s2 = (*handle_data_2)[i];
          if (s1.dtype != s2.dtype) {
            // TODO(apassos) resolve this in the manner of b/32476923
            return errors::InvalidArgument(
                "Trying to merge handles pointing to different dtypes.");
          }
          merged_handle_data[i].dtype = s1.dtype;
          TF_RETURN_IF_ERROR(
              c->Merge(s1.shape, s2.shape, &merged_handle_data[i].shape));
        }

        c->set_output_handle_shapes_and_types(0, merged_handle_data);
      }

      // The inputs 'cond', 'then', and 'else' must be broadcastable.
      // TODO (yongtang): Consolidate 3-ary broadcast instead of
      // multiple 2-ary broadcast.
      ShapeHandle cond = c->input(0);
      ShapeHandle then = c->input(1);
      ShapeHandle else_ = c->input(2);
      ShapeHandle other;
      TF_RETURN_IF_ERROR(
          BroadcastBinaryOpOutputShapeFnHelper(c, then, else_, true, &other));
      ShapeHandle output;
      TF_RETURN_IF_ERROR(
          BroadcastBinaryOpOutputShapeFnHelper(c, cond, other, true, &output));
      c->set_output(0, output);
      return Status::OK();
    });
=======
    .Output("out: T")
    .Attr("T: type")
    .Doc(R"doc(
Selects elements from `t` or `e`, depending on `condition`.

The `condition`, `t`, and `e` tensors must all have the same shape,
and the output will also have that shape. The `condition` tensor acts
as an element-wise mask that chooses, based on the value at each
element, whether the corresponding element in the output should be
taken from `t` (if true) or `e` (if false). For example:

For example:

```prettyprint
# 'condition' tensor is [[True, False]
#                        [True, False]]
# 't' is [[1, 1],
#         [1, 1]]
# 'e' is [[2, 2],
#         [2, 2]]
select(condition, t, e) ==> [[1, 2],
                             [1, 2]]
```

t:= A `Tensor` with the same shape as `condition`.
e:= A `Tensor` with the same type and shape as `t`.
out:= A `Tensor` with the same type and shape as `t` and `e`.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

// --------------------------------------------------------------------------

REGISTER_OP("MatMul")
    .Input("a: T")
    .Input("b: T")
    .Output("product: T")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
<<<<<<< HEAD
    .Attr(
        "T: {bfloat16, half, float, double, int32, int64, complex64, "
        "complex128}")
    .SetShapeFn(shape_inference::MatMulShape);

#ifdef INTEL_MKL
REGISTER_OP("_MklMatMul")
    .Input("a: T")
    .Input("b: T")
    .Output("product: T")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("T: {bfloat16, float, double, complex64, complex128}")
    .SetShapeFn(shape_inference::MatMulShape);
#endif  // INTEL_MKL

REGISTER_OP("SparseMatMul")
    .Input("a: Ta")
    .Input("b: Tb")
=======
    .Attr("T: {float, double, int32, complex64}")
    .Doc(R"doc(
Multiply the matrix "a" by the matrix "b".

The inputs must be two-dimensional matrices and the inner dimension of
"a" (after being transposed if transpose_a is true) must match the
outer dimension of "b" (after being transposed if transposed_b is
true).

*Note*: The default kernel implementation for MatMul on GPUs uses
cublas.

transpose_a: If true, "a" is transposed before multiplication.
transpose_b: If true, "b" is transposed before multiplication.
)doc");

REGISTER_OP("SparseMatMul")
    .Input("a: float")
    .Input("b: float")
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    .Output("product: float")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("a_is_sparse: bool = false")
    .Attr("b_is_sparse: bool = false")
<<<<<<< HEAD
    .Attr("Ta: {float, bfloat16} = DT_FLOAT")
    .Attr("Tb: {float, bfloat16} = DT_FLOAT")
    .SetShapeFn(shape_inference::MatMulShape);

REGISTER_OP("_FusedMatMul")
    .Input("a: T")
    .Input("b: T")
    .Input("args: num_args * T")
    .Output("product: T")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("T: {float}")
    .Attr("num_args: int >= 0")
    .Attr("fused_ops: list(string) = []")
    // Attributes for the FusedBatchNorm ----------- //
    .Attr("epsilon: float = 0.0001")
    // --------------------------------------------- //
    .SetShapeFn(shape_inference::MatMulShape)
    .Doc(R"doc(
*NOTE*: Do not invoke this operator directly in Python. Grappler is
expected to create these operators.
=======
    .Doc(R"doc(
Multiply matrix "a" by matrix "b".

The inputs must be two-dimensional matrices and the inner dimension of "a" must
match the outer dimension of "b". This op is optimized for the case where at
least one of "a" or "b" is sparse. The breakeven for using this versus a dense
matrix multiply on one platform was 30% zero values in the sparse matrix.
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
)doc");

// --------------------------------------------------------------------------

// For operations where the output is a reduction function along some
// dimensions of the input.
REGISTER_OP("Sum")
    .Input("input: T")
<<<<<<< HEAD
    .Input("reduction_indices: Tidx")
    .Output("output: T")
    .Attr("keep_dims: bool = false")
    .Attr("T: numbertype")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::ReductionShape);

REGISTER_OP("EuclideanNorm")
    .Input("input: T")
    .Input("reduction_indices: Tidx")
    .Output("output: T")
    .Attr("keep_dims: bool = false")
    .Attr("T: numbertype")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::ReductionShape);

REGISTER_OP("Mean")
    .Input("input: T")
    .Input("reduction_indices: Tidx")
    .Output("output: T")
    .Attr("keep_dims: bool = false")
    .Attr("T: numbertype")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::ReductionShape);

REGISTER_OP("Prod")
    .Input("input: T")
    .Input("reduction_indices: Tidx")
    .Output("output: T")
    .Attr("keep_dims: bool = false")
    .Attr("T: numbertype")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::ReductionShape);

REGISTER_OP("Min")
    .Input("input: T")
    .Input("reduction_indices: Tidx")
    .Output("output: T")
    .Attr("keep_dims: bool = false")
    .Attr("T: numbertype")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::ReductionShape);

REGISTER_OP("Max")
    .Input("input: T")
    .Input("reduction_indices: Tidx")
    .Output("output: T")
    .Attr("keep_dims: bool = false")
    .Attr("T: numbertype")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::ReductionShape);

namespace {

Status ArgOpShape(shape_inference::InferenceContext* c) {
  ShapeHandle dimension_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &dimension_shape));

  ShapeHandle input_shape = c->input(0);
  if (!c->RankKnown(input_shape)) {
    return shape_inference::UnknownShape(c);
  }

  const int32 input_rank = c->Rank(input_shape);
  if (input_rank <= 1) {
    // Reducing a scalar/vector must return a scalar.
    return shape_inference::ScalarShape(c);
  }

  const Tensor* dim_t = c->input_tensor(1);
  if (dim_t == nullptr) {
    // We don't know the value of the dimension, but we
    // know the rank of the input, so return the correct
    // rank with unknown dimensions.
    std::vector<DimensionHandle> dims(input_rank - 1);
    for (int i = 0; i < dims.size(); ++i) {
      dims[i] = c->UnknownDim();
    }

    c->set_output(0, c->MakeShape(dims));
    return Status::OK();
  }

  int64 dimension_val;
  if (dim_t->dtype() == DT_INT32) {
    dimension_val = dim_t->scalar<int32>()();
  } else {
    dimension_val = dim_t->scalar<int64>()();
  }

  int64 axis = dimension_val < 0 ? dimension_val + input_rank : dimension_val;
  if (axis < 0 || axis >= input_rank) {
    return errors::InvalidArgument(
        "Dimension (", dimension_val, ") must be in the range [", -input_rank,
        ", ", input_rank, "), where ", input_rank,
        " is the number of dimensions in the input.");
  }

  // Return the input shape without the dimension being reduced.
  std::vector<DimensionHandle> dims;
  for (int i = 0; i < input_rank; ++i) {
    if (axis != i) {
      dims.emplace_back(c->Dim(input_shape, i));
    }
  }
  c->set_output(0, c->MakeShape(dims));
  return Status::OK();
}

}  // namespace

REGISTER_OP("ArgMax")
    .Input("input: T")
    .Input("dimension: Tidx")
    .Output("output: output_type")
    .Attr("T: numbertype")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Attr("output_type: {int32, int64} = DT_INT64")
    .SetShapeFn(ArgOpShape);

REGISTER_OP("ArgMin")
    .Input("input: T")
    .Input("dimension: Tidx")
    .Output("output: output_type")
    .Attr("T: numbertype")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Attr("output_type: {int32, int64} = DT_INT64")
    .SetShapeFn(ArgOpShape);

namespace {

Status SegmentReductionShapeFn(InferenceContext* c) {
  ShapeHandle data_shape;
  ShapeHandle segment_ids_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &data_shape));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &segment_ids_shape));

  ShapeHandle subshape;
  TF_RETURN_IF_ERROR(c->Subshape(data_shape, 1, &subshape));

  ShapeHandle out;
  TF_RETURN_IF_ERROR(
      c->Concatenate(c->Vector(InferenceContext::kUnknownDim), subshape, &out));
  c->set_output(0, out);
  return Status::OK();
}

Status SparseSegmentReductionShapeFn(InferenceContext* c) {
  ShapeHandle data_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &data_shape));

  ShapeHandle indices_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &indices_shape));

  ShapeHandle segment_ids_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &segment_ids_shape));

  // indices and segment_ids should merge cleanly.
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->Merge(indices_shape, segment_ids_shape, &unused));

  ShapeHandle subshape;
  TF_RETURN_IF_ERROR(c->Subshape(data_shape, 1, &subshape));

  ShapeHandle out;
  TF_RETURN_IF_ERROR(
      c->Concatenate(c->Vector(InferenceContext::kUnknownDim), subshape, &out));
  c->set_output(0, out);
  return Status::OK();
}

Status SparseSegmentReductionGradShapeFn(InferenceContext* c) {
  ShapeHandle data_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &data_shape));

  ShapeHandle indices_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &indices_shape));

  // indices and segment_ids should merge cleanly.
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->Merge(c->input(2), indices_shape, &unused));

  // output_dim0 should be a scalar
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));

  ShapeHandle subshape;
  TF_RETURN_IF_ERROR(c->Subshape(data_shape, 1, &subshape));

  const Tensor* dim0 = c->input_tensor(3);
  ShapeHandle dim0_shape;
  if (dim0 == nullptr) {
    // We don't have the value at inference time, so the output
    // shape is unknown.
    dim0_shape = c->Vector(InferenceContext::kUnknownDim);
  } else {
    auto dim0_value = dim0->scalar<int32>()();
    if (dim0_value < 0) {
      return errors::InvalidArgument(
          "Cannot specify a negative value for output_dim0");
    }
    dim0_shape = c->Vector(dim0_value);
  }

  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->Concatenate(dim0_shape, subshape, &out));
  c->set_output(0, out);
  return Status::OK();
}

Status SparseSegmentReductionWithNumSegmentsShapeFn(InferenceContext* c) {
  ShapeHandle data_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &data_shape));

  ShapeHandle indices_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &indices_shape));

  ShapeHandle segment_ids_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &segment_ids_shape));

  ShapeHandle num_segments_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &num_segments_shape));

  // indices and segment_ids should merge cleanly.
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->Merge(indices_shape, segment_ids_shape, &unused));

  ShapeHandle subshape;
  TF_RETURN_IF_ERROR(c->Subshape(data_shape, 1, &subshape));

  ShapeHandle out;
  const Tensor* dim0 = c->input_tensor(3);
  if (dim0 == nullptr) {
    // We don't have the value at inference time, so the output
    // shape is unknown.
    TF_RETURN_IF_ERROR(c->Concatenate(c->Vector(InferenceContext::kUnknownDim),
                                      subshape, &out));
  } else {
    auto dim0_value = dim0->scalar<int32>()();
    if (dim0_value < 0) {
      return errors::InvalidArgument(
          "Cannot specify a negative value for num_segments");
    }
    TF_RETURN_IF_ERROR(c->Concatenate(c->Vector(dim0_value), subshape, &out));
  }
  c->set_output(0, out);
  return Status::OK();
}
}  // namespace
=======
    .Input("reduction_indices: int32")
    .Output("output: T")
    .Attr("keep_dims: bool = false")
    .Attr("T: numbertype")
    .Doc(R"doc(
Computes the sum of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

input: The tensor to reduce.
reduction_indices: The dimensions to reduce.
keep_dims: If true, retain reduced dimensions with length 1.
output: The reduced tensor.
)doc");

REGISTER_OP("Mean")
    .Input("input: T")
    .Input("reduction_indices: int32")
    .Output("output: T")
    .Attr("keep_dims: bool = false")
    .Attr("T: numbertype")
    .Doc(R"doc(
Computes the mean of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

input: The tensor to reduce.
reduction_indices: The dimensions to reduce.
keep_dims: If true, retain reduced dimensions with length 1.
output: The reduced tensor.
)doc");

REGISTER_OP("Prod")
    .Input("input: T")
    .Input("reduction_indices: int32")
    .Output("output: T")
    .Attr("keep_dims: bool = false")
    .Attr("T: numbertype")
    .Doc(R"doc(
Computes the product of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

input: The tensor to reduce.
reduction_indices: The dimensions to reduce.
keep_dims: If true, retain reduced dimensions with length 1.
output: The reduced tensor.
)doc");

REGISTER_OP("Min")
    .Input("input: T")
    .Input("reduction_indices: int32")
    .Output("output: T")
    .Attr("keep_dims: bool = false")
    .Attr("T: numbertype")
    .Doc(R"doc(
Computes the minimum of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

input: The tensor to reduce.
reduction_indices: The dimensions to reduce.
keep_dims: If true, retain reduced dimensions with length 1.
output: The reduced tensor.
)doc");

REGISTER_OP("Max")
    .Input("input: T")
    .Input("reduction_indices: int32")
    .Output("output: T")
    .Attr("keep_dims: bool = false")
    .Attr("T: numbertype")
    .Doc(R"doc(
Computes the maximum of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

input: The tensor to reduce.
reduction_indices: The dimensions to reduce.
keep_dims: If true, retain reduced dimensions with length 1.
output: The reduced tensor.
)doc");

REGISTER_OP("ArgMax")
    .Input("input: T")
    .Input("dimension: int32")
    .Output("output: int64")
    .Attr("T: numbertype")
    .Doc(R"doc(
Returns the index with the largest value across dimensions of a tensor.

dimension: int32, 0 <= dimension < rank(input).  Describes which dimension
  of the input Tensor to reduce across. For vectors, use dimension = 0.
)doc");

REGISTER_OP("ArgMin")
    .Input("input: T")
    .Input("dimension: int32")
    .Output("output: int64")
    .Attr("T: numbertype")
    .Doc(R"doc(
Returns the index with the smallest value across dimensions of a tensor.

dimension: int32, 0 <= dimension < rank(input).  Describes which dimension
  of the input Tensor to reduce across. For vectors, use dimension = 0.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("SegmentSum")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Output("output: T")
<<<<<<< HEAD
    .Attr("T: numbertype")
    .Attr("Tindices: {int32,int64}")
    .SetShapeFn(SegmentReductionShapeFn);
=======
    .Attr("T: realnumbertype")
    .Attr("Tindices: {int32,int64}")
    .Doc(R"doc(
Computes the sum along segments of a tensor.

Read [the section on Segmentation](../python/math_ops.md#segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output_i = \sum_j data_j\\) where sum is over `j` such
that `segment_ids[j] == i`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../images/SegmentSum.png" alt>
</div>

segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
first dimension.  Values should be sorted and can be repeated.

output: Has same shape as data, except for dimension_0 which
has size `k`, the number of segments.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("SegmentMean")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Output("output: T")
<<<<<<< HEAD
    .Attr("T: numbertype")
    .Attr("Tindices: {int32,int64}")
    .SetShapeFn(SegmentReductionShapeFn);
=======
    .Attr("T: realnumbertype")
    .Attr("Tindices: {int32,int64}")
    .Doc(R"doc(
Computes the mean along segments of a tensor.

Read [the section on Segmentation](../python/math_ops.md#segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output_i = \frac{\sum_j data_j}{N}\\) where `mean` is
over `j` such that `segment_ids[j] == i` and `N` is the total number of
values summed.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../images/SegmentMean.png" alt>
</div>

segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
first dimension.  Values should be sorted and can be repeated.

output: Has same shape as data, except for dimension_0 which
has size `k`, the number of segments.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("SegmentProd")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Output("output: T")
<<<<<<< HEAD
    .Attr("T: numbertype")
    .Attr("Tindices: {int32,int64}")
    .SetShapeFn(SegmentReductionShapeFn);
=======
    .Attr("T: realnumbertype")
    .Attr("Tindices: {int32,int64}")
    .Doc(R"doc(
Computes the product along segments of a tensor.

Read [the section on Segmentation](../python/math_ops.md#segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output_i = \prod_j data_j\\) where the product is over `j` such
that `segment_ids[j] == i`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../images/SegmentProd.png" alt>
</div>

segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
first dimension.  Values should be sorted and can be repeated.

output: Has same shape as data, except for dimension_0 which
has size `k`, the number of segments.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("SegmentMin")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Attr("Tindices: {int32,int64}")
<<<<<<< HEAD
    .SetShapeFn(SegmentReductionShapeFn);
=======
    .Doc(R"doc(
Computes the minimum along segments of a tensor.

Read [the section on Segmentation](../python/math_ops.md#segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output_i = \min_j(data_j)\\) where `min` is over `j` such
that `segment_ids[j] == i`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../images/SegmentMin.png" alt>
</div>

segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
first dimension.  Values should be sorted and can be repeated.

output: Has same shape as data, except for dimension_0 which
has size `k`, the number of segments.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("SegmentMax")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Attr("Tindices: {int32,int64}")
<<<<<<< HEAD
    .SetShapeFn(SegmentReductionShapeFn);

REGISTER_OP("UnsortedSegmentSum")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Input("num_segments: Tnumsegments")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32,int64}")
    .Attr("Tnumsegments: {int32,int64} = DT_INT32")
    .SetShapeFn(shape_inference::UnsortedSegmentReductionShapeFn);

REGISTER_OP("UnsortedSegmentMax")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Input("num_segments: Tnumsegments")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Attr("Tindices: {int32,int64}")
    .Attr("Tnumsegments: {int32,int64} = DT_INT32")
    .SetShapeFn(shape_inference::UnsortedSegmentReductionShapeFn);

REGISTER_OP("UnsortedSegmentMin")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Input("num_segments: Tnumsegments")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Attr("Tindices: {int32,int64}")
    .Attr("Tnumsegments: {int32,int64} = DT_INT32")
    .SetShapeFn(shape_inference::UnsortedSegmentReductionShapeFn);

REGISTER_OP("UnsortedSegmentProd")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Input("num_segments: Tnumsegments")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32,int64}")
    .Attr("Tnumsegments: {int32,int64} = DT_INT32")
    .SetShapeFn(shape_inference::UnsortedSegmentReductionShapeFn);

REGISTER_OP("SparseSegmentSum")
    .Input("data: T")
    .Input("indices: Tidx")
    .Input("segment_ids: int32")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(SparseSegmentReductionShapeFn);

REGISTER_OP("SparseSegmentSumWithNumSegments")
    .Input("data: T")
    .Input("indices: Tidx")
    .Input("segment_ids: int32")
    .Input("num_segments: Tnumsegments")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Attr("Tnumsegments: {int32,int64} = DT_INT32")
    .SetShapeFn(SparseSegmentReductionWithNumSegmentsShapeFn);

REGISTER_OP("SparseSegmentMean")
    .Input("data: T")
    .Input("indices: Tidx")
    .Input("segment_ids: int32")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(SparseSegmentReductionShapeFn);

REGISTER_OP("SparseSegmentMeanWithNumSegments")
    .Input("data: T")
    .Input("indices: Tidx")
    .Input("segment_ids: int32")
    .Input("num_segments: Tnumsegments")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Attr("Tnumsegments: {int32,int64} = DT_INT32")
    .SetShapeFn(SparseSegmentReductionWithNumSegmentsShapeFn);

REGISTER_OP("SparseSegmentMeanGrad")
    .Input("grad: T")
    .Input("indices: Tidx")
    .Input("segment_ids: int32")
    .Input("output_dim0: int32")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(SparseSegmentReductionGradShapeFn);

REGISTER_OP("SparseSegmentSqrtN")
    .Input("data: T")
    .Input("indices: Tidx")
    .Input("segment_ids: int32")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(SparseSegmentReductionShapeFn);

REGISTER_OP("SparseSegmentSqrtNWithNumSegments")
    .Input("data: T")
    .Input("indices: Tidx")
    .Input("segment_ids: int32")
    .Input("num_segments: Tnumsegments")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Attr("Tnumsegments: {int32,int64} = DT_INT32")
    .SetShapeFn(SparseSegmentReductionWithNumSegmentsShapeFn);

REGISTER_OP("SparseSegmentSqrtNGrad")
    .Input("grad: T")
    .Input("indices: Tidx")
=======
    .Doc(R"doc(
Computes the maximum along segments of a tensor.

Read [the section on Segmentation](../python/math_ops.md#segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output_i = \max_j(data_j)\\) where `max` is over `j` such
that `segment_ids[j] == i`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../images/SegmentMax.png" alt>
</div>

segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
first dimension.  Values should be sorted and can be repeated.

output: Has same shape as data, except for dimension_0 which
has size `k`, the number of segments.
)doc");

REGISTER_OP("UnsortedSegmentSum")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Input("num_segments: int32")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Attr("Tindices: {int32,int64}")
    .Doc(R"doc(
Computes the sum along segments of a tensor.

Read [the section on Segmentation](../python/math_ops.md#segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output_i = \sum_j data_j\\) where sum is over `j` such
that `segment_ids[j] == i`. Unlike `SegmentSum`, `segment_ids`
need not be sorted and need not cover all values in the full
  range of valid values.

If the sum is empty for a given segment ID `i`, `output[i] = 0`.

`num_segments` should equal the number of distinct segment IDs.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../images/UnsortedSegmentSum.png" alt>
</div>

segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
first dimension.

output: Has same shape as data, except for dimension_0 which
has size `num_segments`.

)doc");

REGISTER_OP("SparseSegmentSum")
    .Input("data: T")
    .Input("indices: int32")
    .Input("segment_ids: int32")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Doc(R"doc(
Computes the sum along sparse segments of a tensor.

Read [the section on Segmentation](../python/math_ops.md#segmentation)
for an explanation of segments.

Like `SegmentSum`, but `segment_ids` can have rank less than `data`'s first
dimension, selecting a subset of dimension_0, specified by `indices`.

For example:

```prettyprint
c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])

# Select two rows, one segment.
tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 0]))
  ==> [[0 0 0 0]]

# Select two rows, two segment.
tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 1]))
  ==> [[ 1  2  3  4]
       [-1 -2 -3 -4]]

# Select all rows, two segments.
tf.sparse_segment_sum(c, tf.constant([0, 1, 2]), tf.constant([0, 0, 1]))
  ==> [[0 0 0 0]
       [5 6 7 8]]

# Which is equivalent to:
tf.segment_sum(c, tf.constant([0, 0, 1]))
```

indices: A 1-D tensor. Has same rank as `segment_ids`.

segment_ids: A 1-D tensor. Values should be sorted and can be repeated.

output: Has same shape as data, except for dimension_0 which
has size `k`, the number of segments.
)doc");

REGISTER_OP("SparseSegmentMean")
    .Input("data: T")
    .Input("indices: int32")
    .Input("segment_ids: int32")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Computes the mean along sparse segments of a tensor.

Read [the section on Segmentation](../python/math_ops.md#segmentation)
for an explanation of segments.

Like `SegmentMean`, but `segment_ids` can have rank less than `data`'s first
dimension, selecting a subset of dimension_0, specified by `indices`.

indices: A 1-D tensor. Has same rank as `segment_ids`.

segment_ids: A 1-D tensor. Values should be sorted and can be repeated.

output: Has same shape as data, except for dimension_0 which
has size `k`, the number of segments.

)doc");

REGISTER_OP("SparseSegmentMeanGrad")
    .Input("grad: T")
    .Input("indices: int32")
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    .Input("segment_ids: int32")
    .Input("output_dim0: int32")
    .Output("output: T")
    .Attr("T: {float, double}")
<<<<<<< HEAD
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(SparseSegmentReductionGradShapeFn);

REGISTER_OP("All")
    .Input("input: bool")
    .Input("reduction_indices: Tidx")
    .Output("output: bool")
    .Attr("keep_dims: bool = false")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::ReductionShape);

REGISTER_OP("Any")
    .Input("input: bool")
    .Input("reduction_indices: Tidx")
    .Attr("keep_dims: bool = false")
    .Output("output: bool")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::ReductionShape);

// --------------------------------------------------------------------------

namespace {

template <typename T>
Status RangeSize(const Tensor* start_t, const Tensor* limit_t,
                 const Tensor* delta_t, InferenceContext* const c) {
  T start = start_t->scalar<T>()();
  T limit = limit_t->scalar<T>()();
  T delta = delta_t->scalar<T>()();
  if (start > limit && delta > T(0)) {
    return errors::InvalidArgument(
        "Requires start <= limit when delta > 0: ", start, "/", limit);
  }
  if (start < limit && delta < T(0)) {
    return errors::InvalidArgument(
        "Requires start >= limit when delta < 0: ", start, "/", limit);
  }
  if (delta == T(0)) {
    return errors::InvalidArgument("Requires delta != 0");
  }

  auto size = (std::is_integral<T>::value
                   ? ((std::abs(limit - start) + std::abs(delta) - T(1)) /
                      std::abs(delta))
                   : (std::ceil(std::abs((limit - start) / delta))));
  c->set_output(0, c->Vector(static_cast<int64>(size)));
  return Status::OK();
}

}  // namespace

REGISTER_OP("Range")
    .Input("start: Tidx")
    .Input("limit: Tidx")
    .Input("delta: Tidx")
    .Output("output: Tidx")
    .Attr("Tidx: {bfloat16, half, float, double, int32, int64} = DT_INT32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(c->input(0), 0, &unused),
                                      " for 'start'");
      TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(c->input(1), 0, &unused),
                                      " for 'limit'");
      TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(c->input(2), 0, &unused),
                                      " for 'delta'");
      const Tensor* start_t = c->input_tensor(0);
      const Tensor* limit_t = c->input_tensor(1);
      const Tensor* delta_t = c->input_tensor(2);
      DataType dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("Tidx", &dtype));
      if (start_t == nullptr || limit_t == nullptr || delta_t == nullptr) {
        c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
        return Status::OK();
      }
      if (dtype == DT_INT32) {
        return RangeSize<int32>(start_t, limit_t, delta_t, c);
      } else if (dtype == DT_INT64) {
        return RangeSize<int64>(start_t, limit_t, delta_t, c);
      } else if (dtype == DT_FLOAT) {
        return RangeSize<float>(start_t, limit_t, delta_t, c);
      } else if (dtype == DT_DOUBLE) {
        return RangeSize<double>(start_t, limit_t, delta_t, c);
      } else if (dtype == DT_BFLOAT16) {
        return RangeSize<bfloat16>(start_t, limit_t, delta_t, c);
      } else {
        return errors::InvalidArgument("Unsupported dtype", dtype);
      }
      return Status::OK();
    });
=======
    .Doc(R"doc(
Computes gradients for SparseSegmentMean.

Returns tensor "output" with same shape as grad, except for dimension_0 whose
value is output_dim0.

grad: gradient propagated to the SparseSegmentMean op.
indices: indices passed to the corresponding SparseSegmentMean op.
segment_ids: segment_ids passed to the corresponding SparseSegmentMean op.
output_dim0: dimension_0 of "data" passed to SparseSegmentMean op.
)doc");

REGISTER_OP("All")
    .Input("input: bool")
    .Input("reduction_indices: int32")
    .Output("output: bool")
    .Attr("keep_dims: bool = false")
    .Doc(R"doc(
Computes the "logical and" of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

input: The tensor to reduce.
reduction_indices: The dimensions to reduce.
keep_dims: If true, retain reduced dimensions with length 1.
output: The reduced tensor.
)doc");

REGISTER_OP("Any")
    .Input("input: bool")
    .Input("reduction_indices: int32")
    .Attr("keep_dims: bool = false")
    .Output("output: bool")
    .Doc(R"doc(
Computes the "logical or" of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

input: The tensor to reduce.
reduction_indices: The dimensions to reduce.
keep_dims: If true, retain reduced dimensions with length 1.
output: The reduced tensor.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("Range")
    .Input("start: int32")
    .Input("limit: int32")
    .Input("delta: int32")
    .Output("output: int32")
    .Doc(R"doc(
Creates a sequence of integers.

This operation creates a sequence of integers that begins at `start` and
extends by increments of `delta` up to but not including `limit`.

For example:

```
# 'start' is 3
# 'limit' is 18
# 'delta' is 3
tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]
```

start: 0-D (scalar). First entry in the sequence.
limit: 0-D (scalar). Upper limit of sequence, exclusive.
delta: 0-D (scalar). Optional. Default is 1. Number that increments `start`.
output: 1-D.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("LinSpace")
    .Input("start: T")
    .Input("stop: T")
<<<<<<< HEAD
    .Input("num: Tidx")
    .Output("output: T")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(c->input(0), 0, &unused),
                                      " for 'start'");
      TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(c->input(1), 0, &unused),
                                      " for 'stop'");
      TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(c->input(2), 0, &unused),
                                      " for 'num'");
      const Tensor* num_t = c->input_tensor(2);
      if (num_t == nullptr) {
        c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
        return Status::OK();
      }

      int64 num;
      if (num_t->dtype() == DT_INT32) {
        num = num_t->scalar<int32>()();
      } else {
        num = num_t->scalar<int64>()();
      }
      if (num <= 0) return errors::InvalidArgument("Requires num > 0: ", num);
      c->set_output(0, c->Vector(num));
      return Status::OK();
    });

REGISTER_OP("Complex")
    .Input("real: T")
    .Input("imag: T")
    .Output("out: Tout")
    .Attr("T: {float, double} = DT_FLOAT")
    .Attr("Tout: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("Real")
    .Input("input: T")
    .Output("output: Tout")
    .Attr("T: {complex64, complex128} = DT_COMPLEX64")
    .Attr("Tout: {float, double} = DT_FLOAT")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("Imag")
    .Input("input: T")
    .Output("output: Tout")
    .Attr("T: {complex64, complex128} = DT_COMPLEX64")
    .Attr("Tout: {float, double} = DT_FLOAT")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("Angle")
    .Input("input: T")
    .Output("output: Tout")
    .Attr("T: {complex64, complex128} = DT_COMPLEX64")
    .Attr("Tout: {float, double} = DT_FLOAT")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("Conj")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {complex64, complex128, variant} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      auto* handle_data = c->input_handle_shapes_and_types(0);
      if (handle_data != nullptr) {
        c->set_output_handle_shapes_and_types(0, *handle_data);
      }
      return Status::OK();
    });

// --------------------------------------------------------------------------

REGISTER_OP("Cross")
    .Input("a: T")
    .Input("b: T")
    .Output("product: T")
    .Attr("T: realnumbertype")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle a_shape;
      ShapeHandle b_shape;
      // * Input rank >= 1.
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &a_shape));
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &b_shape));

      // * Both inputs have the same shape.
      TF_RETURN_IF_ERROR(c->Merge(a_shape, b_shape, &a_shape));

      // * input_shape[-1] == 3.
      if (c->RankKnown(a_shape)) {
        int rank = c->Rank(a_shape);
        auto dim = c->Dim(a_shape, rank - 1);
        TF_RETURN_IF_ERROR(c->WithValue(dim, 3, &dim));
      }
      c->set_output(0, a_shape);
      return Status::OK();
    });

// --------------------------------------------------------------------------

REGISTER_OP("HistogramFixedWidth")
    .Input("values: T")
    .Input("value_range: T")
    .Input("nbins: int32")
    .Output("out: dtype")
    .Attr("T: {int32, int64, float32, float64}")
    .Attr("dtype: {int32, int64} = DT_INT32")
    .SetShapeFn([](InferenceContext* c) {
      // value_range should be a vector.
      ShapeHandle value_range_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &value_range_shape));
      // value_range should have two elements.
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(
          c->WithValue(c->Dim(value_range_shape, 0), 2, &unused));
      // nbins should be a scalar.
      ShapeHandle nbins_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &nbins_shape));

      // If nbins is available, set the shape from nbins.
      const Tensor* nbins_input = c->input_tensor(2);
      if (nbins_input != nullptr) {
        int64 nbins;
        TF_RETURN_IF_ERROR(c->GetScalarFromTensor(nbins_input, &nbins));
        // nbins has to be positive.
        if (nbins <= 0) {
          return errors::InvalidArgument("Requires nbins > 0: ", nbins);
        }
        c->set_output(0, c->Vector(nbins));
      } else {
        c->set_output(0, c->UnknownShapeOfRank(1));
      }
      return Status::OK();
    });

REGISTER_OP("Bincount")
    .Input("arr: int32")
    .Input("size: int32")
    .Input("weights: T")
    .Attr("T: {int32, int64, float32, float64}")
    .Output("bins: T")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      // The input `size` must be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));

      const Tensor* size_tensor = c->input_tensor(1);
      if (size_tensor == nullptr) {
        // Return unknown shape if size is not known.
        c->set_output(0, c->UnknownShapeOfRank(1));
        return Status::OK();
      }

      // Return `[size]` shape if size is known.
      int32 size_val = size_tensor->scalar<int32>()();
      if (size_val < 0) {
        return errors::InvalidArgument("size (", size_val,
                                       ") must be non-negative");
      }
      c->set_output(0, c->MakeShape({size_val}));
      return Status::OK();
    });

REGISTER_OP("Cumsum")
    .Input("x: T")
    .Input("axis: Tidx")
    .Attr("exclusive: bool = false")
    .Attr("reverse: bool = false")
    .Output("out: T")
    .Attr("T: numbertype")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("Cumprod")
    .Input("x: T")
    .Input("axis: Tidx")
    .Attr("exclusive: bool = false")
    .Attr("reverse: bool = false")
    .Output("out: T")
    .Attr("T: numbertype")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("CumulativeLogsumexp")
    .Input("x : T")
    .Input("axis: Tidx")
    .Attr("exclusive: bool = false")
    .Attr("reverse: bool = false")
    .Output("out: T")
    .Attr("T: {float16, float32, float64}")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("QuantizedMatMul")
    .Input("a: T1")
    .Input("b: T2")
    .Input("min_a: float")
    .Input("max_a: float")
    .Input("min_b: float")
    .Input("max_b: float")
    .Output("out: Toutput")
    .Output("min_out: float")
    .Output("max_out: float")
    .Attr("T1: quantizedtype")
    .Attr("T2: quantizedtype")
    .Attr("Toutput: quantizedtype = DT_QINT32")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("Tactivation: quantizedtype = DT_QUINT8")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::MatMulShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));

      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    });

// Note: This op is not commutative w.r.t. to all its inputs.
REGISTER_OP("QuantizedMul")
    .Input("x: T1")
    .Input("y: T2")
    .Input("min_x: float")
    .Input("max_x: float")
    .Input("min_y: float")
    .Input("max_y: float")
    .Output("z: Toutput")
    .Output("min_z: float")
    .Output("max_z: float")
    .Attr("T1: quantizedtype")
    .Attr("T2: quantizedtype")
    .Attr("Toutput: quantizedtype = DT_QINT32")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::BroadcastBinaryOpShapeFn(c));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    });

// Note: This op is not commutative w.r.t. to all its inputs.
REGISTER_OP("QuantizedAdd")
    .Input("x: T1")
    .Input("y: T2")
    .Input("min_x: float")
    .Input("max_x: float")
    .Input("min_y: float")
    .Input("max_y: float")
    .Output("z: Toutput")
    .Output("min_z: float")
    .Output("max_z: float")
    .Attr("T1: quantizedtype")
    .Attr("T2: quantizedtype")
    .Attr("Toutput: quantizedtype = DT_QINT32")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::BroadcastBinaryOpShapeFn(c));
      // min_x, max_x, min_y, max_y should be scalar.
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));

      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("QuantizeDownAndShrinkRange")
    .Input("input: Tinput")
    .Input("input_min: float")
    .Input("input_max: float")
    .Output("output: out_type")
    .Output("output_min: float")
    .Output("output_max: float")
    .Attr("Tinput: quantizedtype")
    .Attr("out_type: quantizedtype")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::UnchangedShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("Requantize")
    .Input("input: Tinput")
    .Input("input_min: float")
    .Input("input_max: float")
    .Input("requested_output_min: float")
    .Input("requested_output_max: float")
    .Output("output: out_type")
    .Output("output_min: float")
    .Output("output_max: float")
    .Attr("Tinput: quantizedtype")
    .Attr("out_type: quantizedtype")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::UnchangedShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("CompareAndBitpack")
    .Input("input: T")
    .Input("threshold: T")
    .Output("output: uint8")
    .Attr("T: {bool, float16, float32, float64, int8, int16, int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &input));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      ShapeHandle output = input;
      if (c->RankKnown(input)) {
        int rank = c->Rank(input);
        auto inner_dim = c->Dim(input, rank - 1);
        DimensionHandle inferred_dim;
        TF_RETURN_IF_ERROR(c->Divide(inner_dim, 8,
                                     /* evenly_divisible */ true,
                                     &inferred_dim));
        TF_RETURN_IF_ERROR(
            c->ReplaceDim(output, rank - 1, inferred_dim, &output));
      }
      c->set_output(0, output);

      return Status::OK();
    });

REGISTER_OP("RequantizationRange")
    .Input("input: Tinput")
    .Input("input_min: float")
    .Input("input_max: float")
    .Output("output_min: float")
    .Output("output_max: float")
    .Attr("Tinput: quantizedtype")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Scalar());
      return Status::OK();
    });

// --------------------------------------------------------------------------

REGISTER_OP("Bucketize")
    .Input("input: T")
    .Output("output: int32")
    .Attr("T: {int32, int64, float, double}")
    .Attr("boundaries: list(float)")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("ClipByValue")
    .Input("t: T")
    .Input("clip_value_min: T")
    .Input("clip_value_max: T")
    .Output("output: T")
    .Attr("T: numbertype")
    .SetShapeFn(shape_inference::UnchangedShape);

#ifdef INTEL_MKL
// Note: This op is not commutative w.r.t. to all its inputs.
REGISTER_OP("_MklAddN")
    .Input("inputs: N * T")
    .Input("mkl_input: N * uint8")
    .Output("sum: T")
    .Output("mkl_sum: uint8")
    .Attr("N: int >= 1")
    .Attr("T: numbertype")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle cur = c->input(c->num_inputs() - 1);
      for (int i = c->num_inputs() - 2; i >= 0; --i) {
        TF_RETURN_WITH_CONTEXT_IF_ERROR(c->Merge(c->input(i), cur, &cur),
                                        "From merging shape ", i,
                                        " with other shapes.");
      }
      c->set_output(0, cur);
      return Status::OK();
    })
    .Doc(R"doc(
Add two input tensors element wise using mkl kernel sum.
inputs: Must all be the same size and shape.
)doc");

#endif  // INTEL_MKL

REGISTER_OP("RequantizePerChannel")
    .Input("input: T")
    .Input("input_min: float")
    .Input("input_max: float")
    .Input("requested_output_min: float")
    .Input("requested_output_max: float")
    .Output("output: out_type")
    .Output("output_min: float")
    .Output("output_max: float")
    .Attr("T: quantizedtype = DT_QINT32")
    .Attr("out_type: quantizedtype = DT_QUINT8")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::UnchangedShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    });
REGISTER_OP("RequantizationRangePerChannel")
    .Input("input: T")
    .Input("input_min: float")
    .Input("input_max: float")
    .Output("output_min: float")
    .Output("output_max: float")
    .Attr("T: quantizedtype = DT_QINT32")
    .Attr("clip_value_max: float")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("NextAfter")
    .Attr("T: {float64, float32} = DT_FLOAT")
    .Input("x1: T")
    .Input("x2: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("SobolSample")
    .Input("dim: int32")
    .Input("num_results: int32")
    .Input("skip: int32")
    .Attr("dtype: {float, double} = DT_DOUBLE")
    .Output("samples: dtype")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      ShapeHandle unused;
      // inputs must be  scalars
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      const Tensor* dim_t = c->input_tensor(0);
      const Tensor* num_results_t = c->input_tensor(1);
      if (dim_t == nullptr || num_results_t == nullptr) {
        c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
        return Status::OK();
      }
      const int32 output_size =
          dim_t->scalar<int32>()() * num_results_t->scalar<int32>()();
      c->set_output(0, c->Vector(output_size));
      return Status::OK();
    });
=======
    .Input("num: int32")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Generates values in an interval.

A sequence of `num` evenly-spaced values are generated beginning at `start`.
If `num > 1`, the values in the sequence increase by `stop - start / num - 1`,
so that the last one is exactly `stop`.

For example:

```
tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
```

start: First entry in the range.
stop: Last entry in the range.
num: Number of values to generate.
output: 1-D. The generated values.
)doc");

REGISTER_OP("Complex")
    .Input("real: float")
    .Input("imag: float")
    .Output("out: complex64")
    .Doc(R"doc(
Converts two real numbers to a complex number.

Given a tensor `real` representing the real part of a complex number, and a
tensor `imag` representing the imaginary part of a complex number, this
operation returns complex numbers elementwise of the form \\(a + bj\\), where
*a* represents the `real` part and *b* represents the `imag` part.

The input tensors `real` and `imag` must have the same shape.

For example:

```
# tensor 'real' is [2.25, 3.25]
# tensor `imag` is [4.75, 5.75]
tf.complex(real, imag) ==> [[2.25 + 4.75j], [3.25 + 5.75j]]
```
)doc");

REGISTER_OP("Real")
    .Input("in: complex64")
    .Output("out: float")
    .Doc(R"doc(
Returns the real part of a complex number.

Given a tensor `in` of complex numbers, this operation returns a tensor of type
`float` that is the real part of each element in `in`. All elements in `in`
must be complex numbers of the form \\(a + bj\\), where *a* is the real part
returned by this operation and *b* is the imaginary part.

For example:

```
# tensor 'in' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.real(in) ==> [-2.25, 3.25]
```
)doc");

REGISTER_OP("Imag")
    .Input("in: complex64")
    .Output("out: float")
    .Doc(R"doc(
Returns the imaginary part of a complex number.

Given a tensor `in` of complex numbers, this operation returns a tensor of type
`float` that is the imaginary part of each element in `in`. All elements in `in`
must be complex numbers of the form \\(a + bj\\), where *a* is the real part
and *b* is the imaginary part returned by this operation.

For example:

```
# tensor 'in' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.imag(in) ==> [4.75, 5.75]
```
)doc");

REGISTER_OP("Conj")
    .Input("in: complex64")
    .Output("out: complex64")
    .Doc(R"doc(
Returns the complex conjugate of a complex number.

Given a tensor `in` of complex numbers, this operation returns a tensor of
complex numbers that are the complex conjugate of each element in `in`. The
complex numbers in `in` must be of the form \\(a + bj\\), where *a* is the real
part and *b* is the imaginary part.

The complex conjugate returned by this operation is of the form \\(a - bj\\).

For example:

```
# tensor 'in' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.conj(in) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
```
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

}  // namespace tensorflow
