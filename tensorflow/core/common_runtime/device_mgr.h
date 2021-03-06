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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_MGR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_MGR_H_

#include <memory>
=======
#ifndef TENSORFLOW_COMMON_RUNTIME_DEVICE_MGR_H_
#define TENSORFLOW_COMMON_RUNTIME_DEVICE_MGR_H_

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
<<<<<<< HEAD
#include "tensorflow/core/lib/core/arena.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/macros.h"
=======
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/public/status.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

namespace tensorflow {

class DeviceAttributes;

<<<<<<< HEAD
// Represents a set of devices.
class DeviceMgr {
 public:
  DeviceMgr() = default;
  virtual ~DeviceMgr();

  // Returns attributes of all devices.
  virtual void ListDeviceAttributes(
      std::vector<DeviceAttributes>* devices) const = 0;

  // Returns raw pointers to the underlying devices.
  virtual std::vector<Device*> ListDevices() const = 0;

  // Returns a string listing all devices.
  virtual string DebugString() const = 0;

  // Returns a string of all the device mapping.
  virtual string DeviceMappingString() const = 0;

  // Assigns *device with pointer to Device of the given name.
  // Accepts either a full device name, or just the replica-local suffix.
  virtual Status LookupDevice(StringPiece name, Device** device) const = 0;

  // Clears given containers of all devices if 'container' is
  // non-empty. Otherwise, clears default containers of all devices.
  virtual void ClearContainers(gtl::ArraySlice<string> containers) const = 0;

  virtual int NumDeviceType(const string& type) const = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(DeviceMgr);
};

// Represents a static set of devices.
class StaticDeviceMgr : public DeviceMgr {
 public:
  // Constructs a StaticDeviceMgr from a list of devices.
  explicit StaticDeviceMgr(std::vector<std::unique_ptr<Device>> devices);

  // Constructs a StaticDeviceMgr managing a single device.
  explicit StaticDeviceMgr(std::unique_ptr<Device> device);

  ~StaticDeviceMgr() override;

  void ListDeviceAttributes(
      std::vector<DeviceAttributes>* devices) const override;
  std::vector<Device*> ListDevices() const override;
  string DebugString() const override;
  string DeviceMappingString() const override;
  Status LookupDevice(StringPiece name, Device** device) const override;
  void ClearContainers(gtl::ArraySlice<string> containers) const override;
  int NumDeviceType(const string& type) const override;

 private:
  const std::vector<std::unique_ptr<Device>> devices_;

  StringPiece CopyToBackingStore(StringPiece s);

  std::unordered_map<StringPiece, Device*, StringPieceHasher> device_map_;
  core::Arena name_backing_store_;  // Storage for keys in device_map_
  std::unordered_map<string, int> device_type_counts_;

  TF_DISALLOW_COPY_AND_ASSIGN(StaticDeviceMgr);
};

// Represents a dynamic set of devices
class DynamicDeviceMgr : public DeviceMgr {
 public:
  // Constructs an empty DynamicDeviceMgr.
  DynamicDeviceMgr() {}

  ~DynamicDeviceMgr() override;

  void ListDeviceAttributes(
      std::vector<DeviceAttributes>* devices) const override;
  std::vector<Device*> ListDevices() const override;
  string DebugString() const override;
  string DeviceMappingString() const override;
  Status LookupDevice(StringPiece name, Device** device) const override;
  void ClearContainers(gtl::ArraySlice<string> containers) const override;
  int NumDeviceType(const string& type) const override;

  // Add devices to device manager. Returns error for repeated device names.
  Status AddDevices(std::vector<std::unique_ptr<Device>> devices);

  // Remove devices from device manager. Returns error for non-existing devices.
  Status RemoveDevices(std::vector<Device*> devices);

  // Remove devices from device manager by their names. Returns error for
  // non-existing devices.
  Status RemoveDevicesByName(const std::vector<string>& device_names);

 private:
  mutable mutex devices_mu_;

  std::unordered_map<Device*, std::unique_ptr<Device>> dynamic_devices_
      GUARDED_BY(devices_mu_);

  std::unordered_map<string, Device*> device_map_ GUARDED_BY(devices_mu_);

  std::unordered_map<string, int> device_type_counts_ GUARDED_BY(devices_mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(DynamicDeviceMgr);
};
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_MGR_H_
=======
class DeviceMgr {
 public:
  // TODO(zhifengc): Other initialization information.
  explicit DeviceMgr(const std::vector<Device*>& devices);
  ~DeviceMgr();

  // Returns attributes of all devices.
  void ListDeviceAttributes(std::vector<DeviceAttributes>* devices) const;

  std::vector<Device*> ListDevices() const;

  // Returns a string listing all devices.
  string DebugString() const;

  // Returns a string of all the device mapping.
  string DeviceMappingString() const;

  // Assigns *device with pointer to Device of the given name.
  // Accepts either a full device name, or just the replica-local suffix.
  Status LookupDevice(const string& name, Device** device) const;

  // Clears given containers of all devices if 'container' is
  // non-empty. Otherwise, clears default containers of all devices.
  void ClearContainers(gtl::ArraySlice<string> containers) const;

  int NumDeviceType(const string& type) const;

 private:
  typedef gtl::InlinedVector<Device*, 8> DeviceVec;
  DeviceVec devices_;
  std::unordered_map<string, Device*> device_map_;
  std::unordered_map<string, int> device_type_counts_;

  TF_DISALLOW_COPY_AND_ASSIGN(DeviceMgr);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_DEVICE_MGR_H_
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
