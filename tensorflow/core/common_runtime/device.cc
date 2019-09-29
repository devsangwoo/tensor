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

#include "tensorflow/core/common_runtime/device.h"

#include "tensorflow/core/framework/op_segment.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

Device::Device(Env* env, const DeviceAttributes& device_attributes)
    : DeviceBase(env), device_attributes_(device_attributes) {
  CHECK(DeviceNameUtils::ParseFullName(name(), &parsed_name_))
      << "Invalid device name: " << name();
  rmgr_ = new ResourceMgr(parsed_name_.job);
}

Device::~Device() {
  if (rmgr_ != nullptr) {
    DeleteResourceMgr();
  }
}

void Device::Sync(const DoneCallback& done) { done(Sync()); }

// static
DeviceAttributes Device::BuildDeviceAttributes(
    const string& name, DeviceType device, Bytes memory_limit,
    const DeviceLocality& locality, const string& physical_device_desc) {
  DeviceAttributes da;
  da.set_name(name);
  do {
    da.set_incarnation(random::New64());
  } while (da.incarnation() == 0);  // This proto field must not be zero
  da.set_device_type(device.type());
  da.set_memory_limit(memory_limit.value());
  *da.mutable_locality() = locality;
  da.set_physical_device_desc(physical_device_desc);
  return da;
}

Status Device::FillContextMap(const Graph* graph,
                              DeviceContextMap* device_context_map) {
  DeviceContext* device_context = nullptr;
  TF_RETURN_IF_ERROR(TryGetDeviceContext(&device_context));
  if (device_context) {
    device_context_map->resize(graph->num_node_ids());
    for (Node* n : graph->nodes()) {
      // Increment the refcount for every assignment to a node.
      device_context->Ref();
      // Transfers ownership to value in the DeviceContextMap.
      (*device_context_map)[n->id()] = device_context;
    }

    // Decrement the refcount, since each node_id in the returned
    // map has a reference to the context.
    device_context->Unref();
  }
  return Status::OK();
}

}  // namespace tensorflow
