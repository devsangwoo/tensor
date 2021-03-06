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

#include "tensorflow/core/common_runtime/device_mgr.h"

#include <memory>
#include <vector>
=======
#include "tensorflow/core/common_runtime/device_mgr.h"

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

<<<<<<< HEAD
DeviceMgr::~DeviceMgr() {}

StaticDeviceMgr::StaticDeviceMgr(std::vector<std::unique_ptr<Device>> devices)
    : devices_(std::move(devices)), name_backing_store_(128) {
  for (auto& d : devices_) {
    // Register under the (1) full name and (2) canonical name.
    for (const string& name :
         DeviceNameUtils::GetNamesForDeviceMappings(d->parsed_name())) {
      device_map_[CopyToBackingStore(name)] = d.get();
    }
    // Register under the (3) local name and (4) legacy local name.
    for (const string& name :
         DeviceNameUtils::GetLocalNamesForDeviceMappings(d->parsed_name())) {
      device_map_[CopyToBackingStore(name)] = d.get();
    }
=======
DeviceMgr::DeviceMgr(const std::vector<Device*>& devices) {
  for (Device* d : devices) {
    devices_.push_back(d);

    // Register under both the full name and the local name.
    device_map_[d->name()] = d;
    device_map_[DeviceNameUtils::LocalName(d->name())] = d;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    device_type_counts_[d->device_type()]++;
  }
}

<<<<<<< HEAD
StaticDeviceMgr::StaticDeviceMgr(std::unique_ptr<Device> device)
    : StaticDeviceMgr([&device] {
        std::vector<std::unique_ptr<Device>> vector;
        vector.push_back(std::move(device));
        return vector;
      }()) {}

StaticDeviceMgr::~StaticDeviceMgr() {
  // Release resources ahead of destroying the device manager as the resource
  // destructors (e.g. ~IteratorResource) assume devices still exist.
  for (auto& device : devices_) {
    device->ClearResourceMgr();
  }
}

StringPiece StaticDeviceMgr::CopyToBackingStore(StringPiece s) {
  size_t n = s.size();
  char* space = name_backing_store_.Alloc(n);
  memcpy(space, s.data(), n);
  return StringPiece(space, n);
}

void StaticDeviceMgr::ListDeviceAttributes(
    std::vector<DeviceAttributes>* devices) const {
  devices->reserve(devices_.size());
  for (const auto& dev : devices_) {
=======
DeviceMgr::~DeviceMgr() {
  for (auto p : devices_) delete p;
}

void DeviceMgr::ListDeviceAttributes(
    std::vector<DeviceAttributes>* devices) const {
  devices->reserve(devices_.size());
  for (Device* dev : devices_) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    devices->emplace_back(dev->attributes());
  }
}

<<<<<<< HEAD
std::vector<Device*> StaticDeviceMgr::ListDevices() const {
  std::vector<Device*> devices(devices_.size());
  for (size_t i = 0; i < devices_.size(); ++i) {
    devices[i] = devices_[i].get();
  }
  return devices;
}

string StaticDeviceMgr::DebugString() const {
  string out;
  for (const auto& dev : devices_) {
=======
std::vector<Device*> DeviceMgr::ListDevices() const {
  return std::vector<Device*>(devices_.begin(), devices_.end());
}

string DeviceMgr::DebugString() const {
  string out;
  for (Device* dev : devices_) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    strings::StrAppend(&out, dev->name(), "\n");
  }
  return out;
}

<<<<<<< HEAD
string StaticDeviceMgr::DeviceMappingString() const {
  string out;
  for (const auto& dev : devices_) {
=======
string DeviceMgr::DeviceMappingString() const {
  string out;
  for (Device* dev : devices_) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    if (!dev->attributes().physical_device_desc().empty()) {
      strings::StrAppend(&out, dev->name(), " -> ",
                         dev->attributes().physical_device_desc(), "\n");
    }
  }
  return out;
}

<<<<<<< HEAD
Status StaticDeviceMgr::LookupDevice(StringPiece name, Device** device) const {
  auto iter = device_map_.find(name);
  if (iter == device_map_.end()) {
    std::vector<StringPiece> device_names;
    for (auto&& itr : device_map_) {
      device_names.push_back(itr.first);
    }
    VLOG(1) << "Unknown device: " << name
            << " all devices: " << absl::StrJoin(device_names, ", ");
=======
Status DeviceMgr::LookupDevice(const string& name, Device** device) const {
  Status s;
  auto iter = device_map_.find(name);
  if (iter == device_map_.end()) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    return errors::InvalidArgument(name, " unknown device.");
  }
  *device = iter->second;
  return Status::OK();
}

<<<<<<< HEAD
void StaticDeviceMgr::ClearContainers(
    gtl::ArraySlice<string> containers) const {
  Status s;
  for (const auto& dev : devices_) {
=======
void DeviceMgr::ClearContainers(gtl::ArraySlice<string> containers) const {
  Status s;
  for (Device* dev : devices_) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    if (containers.empty()) {
      s.Update(dev->resource_manager()->Cleanup(
          dev->resource_manager()->default_container()));
    } else {
      for (const string& c : containers) {
        s.Update(dev->resource_manager()->Cleanup(c));
      }
    }
    if (!s.ok()) {
      LOG(WARNING) << s;
    }
  }
}

<<<<<<< HEAD
int StaticDeviceMgr::NumDeviceType(const string& type) const {
=======
int DeviceMgr::NumDeviceType(const string& type) const {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  auto iter = device_type_counts_.find(type);
  if (iter != device_type_counts_.end()) return iter->second;
  return 0;
}

}  // namespace tensorflow
