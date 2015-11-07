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
// See docs in ../ops/io_ops.cc.

#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
<<<<<<< HEAD
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
=======
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/match.h"
#include "tensorflow/core/public/env.h"
#include "tensorflow/core/public/tensor_shape.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

namespace tensorflow {

class MatchingFilesOp : public OpKernel {
 public:
  using OpKernel::OpKernel;
  void Compute(OpKernelContext* context) override {
<<<<<<< HEAD
    const Tensor* patterns_t;
    // NOTE(ringwalt): Changing the input name "pattern" to "patterns" would
    // break existing graphs.
    OP_REQUIRES_OK(context, context->input("pattern", &patterns_t));
    OP_REQUIRES(
        context,
        TensorShapeUtils::IsScalar(patterns_t->shape()) ||
            TensorShapeUtils::IsVector(patterns_t->shape()),
        errors::InvalidArgument(
            "Input patterns tensor must be scalar or vector, but had shape: ",
            patterns_t->shape().DebugString()));
    const auto patterns = patterns_t->flat<tstring>();
    int num_patterns = patterns.size();
    int num_files = 0;
    std::vector<std::vector<string>> all_fnames(num_patterns);
    for (int i = 0; i < num_patterns; i++) {
      OP_REQUIRES_OK(context, context->env()->GetMatchingPaths(patterns(i),
                                                               &all_fnames[i]));
      num_files += all_fnames[i].size();
    }
    Tensor* output_t = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output("filenames", TensorShape({num_files}),
                                          &output_t));
    auto output = output_t->vec<tstring>();
    int index = 0;
    for (int i = 0; i < num_patterns; ++i) {
      for (int j = 0; j < all_fnames[i].size(); j++) {
        output(index++) = all_fnames[i][j];
      }
    }
    std::sort(&output(0), &output(0) + num_files);
=======
    const Tensor* pattern;
    OP_REQUIRES_OK(context, context->input("pattern", &pattern));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(pattern->shape()),
                errors::InvalidArgument(
                    "Input pattern tensor must be scalar, but had shape: ",
                    pattern->shape().DebugString()));
    std::vector<string> fnames;
    OP_REQUIRES_OK(context,
                   io::GetMatchingFiles(context->env(),
                                        pattern->scalar<string>()(), &fnames));
    const int num_out = fnames.size();
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                "filenames", TensorShape({num_out}), &output));
    auto output_vec = output->vec<string>();
    for (int i = 0; i < num_out; ++i) {
      output_vec(i) = fnames[i];
    }
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }
};

REGISTER_KERNEL_BUILDER(Name("MatchingFiles").Device(DEVICE_CPU),
                        MatchingFilesOp);

}  // namespace tensorflow
