/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_GRAPPLER_UTILS_TRANSITIVE_FANIN_H_
#define TENSORFLOW_CORE_GRAPPLER_UTILS_TRANSITIVE_FANIN_H_

#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {

// Computes the transitive fanin of the graph based on reachability from the
// specified terminal nodes. ill_formed will be set to true if the graph is
// deemed structurally invalid. Returns the set of nodes comprising the
// transitive fanin.
std::vector<const NodeDef*> ComputeTransitiveFanin(
    const GraphDef& graph, const std::vector<string>& terminal_nodes,
    bool* ill_formed);

// Creates output_graph from input_graph using the transitive fanin from the
// specified terminal nodes. Returns error if the input_graph is deemed
// structurally invalid.
Status SetTransitiveFaninGraph(const GraphDef& input_graph,
                               GraphDef* output_graph,
                               const std::vector<string>& terminal_nodes);

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_UTILS_TRANSITIVE_FANIN_H_
