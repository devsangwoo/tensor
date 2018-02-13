/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_TENSORRT_CONVERT_CONVERT_NODES_H_
#define TENSORFLOW_CONTRIB_TENSORRT_CONVERT_CONVERT_NODES_H_

#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/lib/core/status.h"

//#if GOOGLE_CUDA
//#if GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {
namespace convert {

struct SubGraphParams {
  SubGraphParams(tensorflow::Graph& graph_,
                 const std::set<int>& subgraph_node_ids_,
                 const std::vector<std::pair<int, int>>& input_inds_,
                 const std::vector<std::pair<int, int>>& output_inds_,
                 size_t max_batch_size_, size_t max_workspace_size_bytes_,
                 const tensorflow::grappler::GraphProperties& graph_properties_,
                 std::unordered_map<string, std::pair<int, string>>*
                 output_edge_map_,
                 tensorflow::NodeDef* trt_node_,
                 int precision_mode_ = 0)
      : graph(graph_),
        subgraph_node_ids(subgraph_node_ids_),
        input_inds(input_inds_),
        output_inds(output_inds_),
        max_batch_size(max_batch_size_),
        max_workspace_size_bytes(max_workspace_size_bytes_),
        graph_properties(graph_properties_),
        output_edge_map(output_edge_map_),
        trt_node(trt_node_),
        precision_mode(precision_mode_) {}

  tensorflow::Graph& graph;
  const std::set<int>& subgraph_node_ids;
  const std::vector<std::pair<int, int>>& input_inds;   // {node_id, output_idx}
  const std::vector<std::pair<int, int>>& output_inds;  // {node_id, output_idx}
  size_t max_batch_size;
  size_t max_workspace_size_bytes;
  const tensorflow::grappler::GraphProperties& graph_properties;
  std::unordered_map<string, std::pair<int, string>>* output_edge_map;
  tensorflow::NodeDef* trt_node;
  const int precision_mode;
};

tensorflow::Status ConvertSubGraphToTensorRTNodeDef(SubGraphParams &params);
tensorflow::Status InjectCalibrationNode(SubGraphParams& params);
tensorflow::Status ConvertCalibrationNodeToEngineNode(tensorflow::Graph& graph,
                                                      tensorflow::Node* c_node);
}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

//#endif  // GOOGLE_TENSORRT
//#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CONTRIB_TENSORRT_CONVERT_CONVERT_NODES_H_
