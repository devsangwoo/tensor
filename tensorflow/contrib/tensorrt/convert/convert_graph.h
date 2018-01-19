/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CONTRIB_TENSORRT_CONVERT_CONVERT_GRAPH_H_
#define TENSORFLOW_CONTRIB_TENSORRT_CONVERT_CONVERT_GRAPH_H_

#include <string>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorrt {
namespace convert {

tensorflow::Status ConvertGraphDefToTensorRT(
    const tensorflow::GraphDef& graph_def,
    const std::vector<std::string>& output_names, size_t max_batch_size,
    size_t max_workspace_size, tensorflow::GraphDef* new_graph_def);
}
}  // namespace tensorrt

#endif  // TENSORFLOW_CONTRIB_TENSORRT_CONVERT_CONVERT_GRAPH_H_
