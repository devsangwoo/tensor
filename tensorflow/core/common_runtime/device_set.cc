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
#include "tensorflow/core/common_runtime/device_set.h"

#include <set>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
<<<<<<< HEAD
#include "tensorflow/core/common_runtime/device_factory.h"
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"

namespace tensorflow {

DeviceSet::DeviceSet() {}

DeviceSet::~DeviceSet() {}

void DeviceSet::AddDevice(Device* device) {
  devices_.push_back(device);
<<<<<<< HEAD
  for (const string& name :
       DeviceNameUtils::GetNamesForDeviceMappings(device->parsed_name())) {
    device_by_name_.insert({name, device});
  }
=======
  device_by_name_.insert({device->name(), device});
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

void DeviceSet::FindMatchingDevices(const DeviceNameUtils::ParsedName& spec,
                                    std::vector<Device*>* devices) const {
  // TODO(jeff): If we are going to repeatedly lookup the set of devices
  // for the same spec, maybe we should have a cache of some sort
  devices->clear();
  for (Device* d : devices_) {
    if (DeviceNameUtils::IsCompleteSpecification(spec, d->parsed_name())) {
      devices->push_back(d);
    }
  }
}

Device* DeviceSet::FindDeviceByName(const string& name) const {
  return gtl::FindPtrOrNull(device_by_name_, name);
}

<<<<<<< HEAD
// static
int DeviceSet::DeviceTypeOrder(const DeviceType& d) {
  return DeviceFactory::DevicePriority(d.type_string());
}

static bool DeviceTypeComparator(const DeviceType& a, const DeviceType& b) {
  // First sort by prioritized device type (higher is preferred) and
  // then by device name (lexicographically).
  auto a_priority = DeviceSet::DeviceTypeOrder(a);
  auto b_priority = DeviceSet::DeviceTypeOrder(b);
  if (a_priority != b_priority) {
    return a_priority > b_priority;
  }

  return StringPiece(a.type()) < StringPiece(b.type());
=======
// Higher result implies lower priority.
static int Order(const DeviceType& d) {
  if (StringPiece(d.type()) == DEVICE_CPU) {
    return 3;
  } else if (StringPiece(d.type()) == DEVICE_GPU) {
    return 2;
  } else {
    return 1;
  }
}

static bool ByPriority(const DeviceType& a, const DeviceType& b) {
  // Order by "order number"; break ties lexicographically.
  return std::make_pair(Order(a), StringPiece(a.type())) <
         std::make_pair(Order(b), StringPiece(b.type()));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

std::vector<DeviceType> DeviceSet::PrioritizedDeviceTypeList() const {
  std::vector<DeviceType> result;
  std::set<string> seen;
  for (Device* d : devices_) {
<<<<<<< HEAD
    const auto& t = d->device_type();
    if (seen.insert(t).second) {
      result.emplace_back(t);
    }
  }
  std::sort(result.begin(), result.end(), DeviceTypeComparator);
=======
    auto t = d->device_type();
    if (seen.insert(t).second) {
      result.emplace_back(DeviceType(t));
    }
  }
  std::sort(result.begin(), result.end(), ByPriority);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  return result;
}

}  // namespace tensorflow
