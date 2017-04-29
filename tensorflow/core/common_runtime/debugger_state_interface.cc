/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/debugger_state_interface.h"

namespace tensorflow {

// static
DebuggerStateFactory* DebuggerStateRegistry::factory_ = nullptr;

// static
DebugGraphDecoratorFactory* DebugGraphDecoratorRegistry::factory_ = nullptr;

const string SummarizeDebugTensorWatches(
    const protobuf::RepeatedPtrField<DebugTensorWatch>& watches) {
  std::ostringstream oss;

  for (const DebugTensorWatch& watch : watches) {
    string tensor_name =
        strings::StrCat(watch.node_name(), ":", watch.output_slot());
    if (watch.tolerate_debug_op_creation_failures()) {
      oss << "(TOL)";  // Shorthand for "tolerate".
    }
    oss << tensor_name << "|";

    for (const string& debug_op : watch.debug_ops()) {
      oss << debug_op << ",";
    }

    oss << "@";
    for (const string& debug_url : watch.debug_urls()) {
      oss << debug_url << ",";
    }

    oss << ";";
  }

  return oss.str();
}

// static
void DebuggerStateRegistry::RegisterFactory(
    const DebuggerStateFactory& factory) {
  delete factory_;
  factory_ = new DebuggerStateFactory(factory);
}

// static
std::unique_ptr<DebuggerStateInterface> DebuggerStateRegistry::CreateState(
    const DebugOptions& debug_options) {
  return (factory_ == nullptr || *factory_ == nullptr)
             ? nullptr
             : (*factory_)(debug_options);
}

// static
void DebugGraphDecoratorRegistry::RegisterFactory(
    const DebugGraphDecoratorFactory& factory) {
  delete factory_;
  factory_ = new DebugGraphDecoratorFactory(factory);
}

// static
std::unique_ptr<DebugGraphDecoratorInterface>
DebugGraphDecoratorRegistry::CreateDecorator(const DebugOptions& options) {
  return (factory_ == nullptr || *factory_ == nullptr) ? nullptr
                                                       : (*factory_)(options);
}

}  // end namespace tensorflow
