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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_PIN_TO_HOST_OPTIMIZER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_PIN_TO_HOST_OPTIMIZER_H_

#include <unordered_set>
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {
namespace internal {
// Try and find an appropriate Host device in `devices` given `device`.
bool TrySwapToHostDevice(const gtl::FlatSet<string>& devices,
                         bool has_device_cpu, string* device);
}  // end namespace internal

// Optimize TensorFlow ops that should be swapped into the CPU to avoid
// excessive cpu<->gpu memcpy/sync.
//
// TODO(williamchan): The current heuristic will swap any small integer Const to
// CPU. This may cause a problem cpu->cpu->gpu wherein the original behaviour of
// gpu->gpu->gpu may have been better/faster. We should probably fix this.
class PinToHostOptimizer : public GraphOptimizer {
 public:
  PinToHostOptimizer() : opt_level_(RewriterConfig::DEFAULT) {}
  explicit PinToHostOptimizer(RewriterConfig::Toggle opt_level)
      : opt_level_(opt_level) {}

  ~PinToHostOptimizer() override {}

  string name() const override { return "pin_to_host_optimizer"; };

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimized_graph, double result) override {}

 private:
  RewriterConfig::Toggle opt_level_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_PIN_TO_HOST_OPTIMIZER_H_
