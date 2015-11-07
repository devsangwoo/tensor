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

#ifndef TENSORFLOW_CORE_GRAPH_DEFAULT_DEVICE_H_
#define TENSORFLOW_CORE_GRAPH_DEFAULT_DEVICE_H_
=======
#ifndef TENSORFLOW_GRAPH_DEFAULT_DEVICE_H_
#define TENSORFLOW_GRAPH_DEFAULT_DEVICE_H_
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#include <string>

#include "tensorflow/core/framework/graph.pb.h"
<<<<<<< HEAD
#include "tensorflow/core/framework/node_def.pb.h"
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

namespace tensorflow {
namespace graph {

// Sets the default device for all nodes in graph_def to "device",
// only if not already set.
inline void SetDefaultDevice(const string& device, GraphDef* graph_def) {
  for (int i = 0; i < graph_def->node_size(); ++i) {
    auto node = graph_def->mutable_node(i);
    if (node->device().empty()) {
      node->set_device(device);
    }
  }
}

}  // namespace graph
}  // namespace tensorflow

<<<<<<< HEAD
#endif  // TENSORFLOW_CORE_GRAPH_DEFAULT_DEVICE_H_
=======
#endif  // TENSORFLOW_GRAPH_DEFAULT_DEVICE_H_
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
