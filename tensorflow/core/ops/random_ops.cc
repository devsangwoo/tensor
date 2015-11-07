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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

=======
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
REGISTER_OP("RandomUniform")
    .Input("shape: T")
    .SetIsStateful()
    .Output("output: dtype")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
<<<<<<< HEAD
    .Attr("dtype: {half,bfloat16,float,double}")
    .Attr("T: {int32, int64}")
    .SetShapeFn(shape_inference::RandomShape);

REGISTER_OP("RandomUniformInt")
    .Input("shape: T")
    .Input("minval: Tout")
    .Input("maxval: Tout")
    .SetIsStateful()
    .Output("output: Tout")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("Tout: {int32, int64}")
    .Attr("T: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      return shape_inference::RandomShape(c);
    });
=======
    .Attr("dtype: {float,double}")
    .Attr("T: {int32, int64}")
    .Doc(R"doc(
Outputs random values from a uniform distribution.

The generated values follow a uniform distribution in the range `[0, 1)`. The
lower bound 0 is included in the range, while the upper bound 1 is excluded.

shape: The shape of the output tensor.
dtype: The type of the output.
seed: If either `seed` or `seed2` are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: A second seed to avoid seed collision.

output: A tensor of the specified shape filled with uniform random values.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("RandomStandardNormal")
    .Input("shape: T")
    .SetIsStateful()
    .Output("output: dtype")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
<<<<<<< HEAD
    .Attr("dtype: {half,bfloat16,float,double}")
    .Attr("T: {int32, int64}")
    .SetShapeFn(shape_inference::RandomShape);

REGISTER_OP("ParameterizedTruncatedNormal")
    .Input("shape: T")
    .Input("means: dtype")
    .Input("stdevs: dtype")
    .Input("minvals: dtype")
    .Input("maxvals: dtype")
    .SetIsStateful()
    .Output("output: dtype")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("dtype: {half,bfloat16,float,double}")
    .Attr("T: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      // Parameters must be 0-d or 1-d.
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(3), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(4), 1, &unused));
      return shape_inference::RandomShape(c);
    });
=======
    .Attr("dtype: {float,double}")
    .Attr("T: {int32, int64}")
    .Doc(R"doc(
Outputs random values from a normal distribution.

The generated values will have mean 0 and standard deviation 1.

shape: The shape of the output tensor.
dtype: The type of the output.
seed: If either `seed` or `seed2` are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: A second seed to avoid seed collision.

output: A tensor of the specified shape filled with random normal values.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("TruncatedNormal")
    .Input("shape: T")
    .SetIsStateful()
    .Output("output: dtype")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
<<<<<<< HEAD
    .Attr("dtype: {half,bfloat16,float,double}")
    .Attr("T: {int32, int64}")
    .SetShapeFn(shape_inference::RandomShape);
=======
    .Attr("dtype: {float,double}")
    .Attr("T: {int32, int64}")
    .Doc(R"doc(
Outputs random values from a truncated normal distribution.

The generated values follow a normal distribution with mean 0 and standard
deviation 1, except that values whose magnitude is more than 2 standard
deviations from the mean are dropped and re-picked.

shape: The shape of the output tensor.
dtype: The type of the output.
seed: If either `seed` or `seed2` are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: A second seed to avoid seed collision.

output: A tensor of the specified shape filled with random truncated normal
  values.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("RandomShuffle")
    .Input("value: T")
    .SetIsStateful()
    .Output("output: T")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("T: type")
<<<<<<< HEAD
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("Multinomial")
    .SetIsStateful()
    .Input("logits: T")
    .Input("num_samples: int32")
    .Output("output: output_dtype")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("T: realnumbertype")
    .Attr("output_dtype: {int32, int64} = DT_INT64")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle logits_shape;
      ShapeHandle unused;
      DimensionHandle num_samples;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &logits_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(1, &num_samples));
      c->set_output(0, c->Matrix(c->Dim(logits_shape, 0), num_samples));
      return Status::OK();
    });

REGISTER_OP("RandomGamma")
    .SetIsStateful()
    .Input("shape: S")
    .Input("alpha: T")
    .Output("output: T")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("S: {int32, int64}")
    .Attr("T: {half, float, double}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
      TF_RETURN_IF_ERROR(c->Concatenate(out, c->input(1), &out));
      c->set_output(0, out);
      return Status::OK();
    });

REGISTER_OP("RandomGammaGrad")
    .Input("alpha: T")
    .Input("sample: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("RandomPoisson")
    .SetIsStateful()
    .Input("shape: S")
    .Input("rate: dtype")
    .Output("output: dtype")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("S: {int32, int64}")
    .Attr("dtype: {half, float, double}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
      TF_RETURN_IF_ERROR(c->Concatenate(out, c->input(1), &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Deprecated(25, "Replaced by RandomPoissonV2");

REGISTER_OP("RandomPoissonV2")
    .SetIsStateful()
    .Input("shape: S")
    .Input("rate: R")
    .Output("output: dtype")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("S: {int32, int64}")
    .Attr("R: {half, float, double, int32, int64} = DT_DOUBLE")
    .Attr("dtype: {half, float, double, int32, int64} = DT_INT64")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
      TF_RETURN_IF_ERROR(c->Concatenate(out, c->input(1), &out));
      c->set_output(0, out);
      return Status::OK();
    });
=======
    .Doc(R"doc(
Randomly shuffles a tensor along its first dimension.

  The tensor is shuffled along dimension 0, such that each `value[j]` is mapped
  to one and only one `output[i]`. For example, a mapping that might occur for a
  3x2 tensor is:

```prettyprint
[[1, 2],       [[5, 6],
 [3, 4],  ==>   [1, 2],
 [5, 6]]        [3, 4]]
```

value: The tensor to be shuffled.
seed: If either `seed` or `seed2` are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: A second seed to avoid seed collision.

output: A tensor of same shape and type as `value`, shuffled along its first
  dimension.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

}  // namespace tensorflow
