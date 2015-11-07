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

#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/device_factory.h"

#include <vector>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/test.h"
=======
#include "tensorflow/core/common_runtime/device_set.h"

#include "tensorflow/core/public/status.h"
#include <gtest/gtest.h>
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

namespace tensorflow {
namespace {

// Return a fake device with the specified type and name.
static Device* Dev(const char* type, const char* name) {
  class FakeDevice : public Device {
   public:
<<<<<<< HEAD
    explicit FakeDevice(const DeviceAttributes& attr) : Device(nullptr, attr) {}
=======
    explicit FakeDevice(const DeviceAttributes& attr)
        : Device(nullptr, attr, nullptr) {}
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    Status Sync() override { return Status::OK(); }
    Allocator* GetAllocator(AllocatorAttributes) override { return nullptr; }
  };
  DeviceAttributes attr;
  attr.set_name(name);
  attr.set_device_type(type);
  return new FakeDevice(attr);
}

<<<<<<< HEAD
class DeviceSetTest : public ::testing::Test {
=======
class DeviceSetTest : public testing::Test {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
 public:
  void AddDevice(const char* type, const char* name) {
    Device* d = Dev(type, name);
    owned_.emplace_back(d);
    devices_.AddDevice(d);
  }

  std::vector<DeviceType> types() const {
    return devices_.PrioritizedDeviceTypeList();
  }

 private:
  DeviceSet devices_;
  std::vector<std::unique_ptr<Device>> owned_;
};

<<<<<<< HEAD
class DummyFactory : public DeviceFactory {
 public:
  Status ListPhysicalDevices(std::vector<string>* devices) override {
    return Status::OK();
  }
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override {
    return Status::OK();
  }
};

// Assumes the default priority is '50'.
REGISTER_LOCAL_DEVICE_FACTORY("d1", DummyFactory);
REGISTER_LOCAL_DEVICE_FACTORY("d2", DummyFactory, 51);
REGISTER_LOCAL_DEVICE_FACTORY("d3", DummyFactory, 49);

TEST_F(DeviceSetTest, PrioritizedDeviceTypeList) {
  EXPECT_EQ(50, DeviceSet::DeviceTypeOrder(DeviceType("d1")));
  EXPECT_EQ(51, DeviceSet::DeviceTypeOrder(DeviceType("d2")));
  EXPECT_EQ(49, DeviceSet::DeviceTypeOrder(DeviceType("d3")));

  EXPECT_EQ(std::vector<DeviceType>{}, types());

  AddDevice("d1", "/job:a/replica:0/task:0/device:d1:0");
  EXPECT_EQ(std::vector<DeviceType>{DeviceType("d1")}, types());

  AddDevice("d1", "/job:a/replica:0/task:0/device:d1:1");
  EXPECT_EQ(std::vector<DeviceType>{DeviceType("d1")}, types());

  // D2 is prioritized higher than D1.
  AddDevice("d2", "/job:a/replica:0/task:0/device:d2:0");
  EXPECT_EQ((std::vector<DeviceType>{DeviceType("d2"), DeviceType("d1")}),
            types());

  // D3 is prioritized below D1.
  AddDevice("d3", "/job:a/replica:0/task:0/device:d3:0");
  EXPECT_EQ((std::vector<DeviceType>{
                DeviceType("d2"),
                DeviceType("d1"),
                DeviceType("d3"),
            }),
            types());
=======
TEST_F(DeviceSetTest, PrioritizedDeviceTypeList) {
  EXPECT_EQ(std::vector<DeviceType>{}, types());

  AddDevice("CPU", "/job:a/replica:0/task:0/cpu:0");
  EXPECT_EQ(std::vector<DeviceType>{DeviceType(DEVICE_CPU)}, types());

  AddDevice("CPU", "/job:a/replica:0/task:0/cpu:1");
  EXPECT_EQ(std::vector<DeviceType>{DeviceType(DEVICE_CPU)}, types());

  AddDevice("GPU", "/job:a/replica:0/task:0/gpu:0");
  EXPECT_EQ(
      (std::vector<DeviceType>{DeviceType(DEVICE_GPU), DeviceType(DEVICE_CPU)}),
      types());

  AddDevice("T1", "/job:a/replica:0/task:0/device:T1:0");
  AddDevice("T1", "/job:a/replica:0/task:0/device:T1:1");
  AddDevice("T2", "/job:a/replica:0/task:0/device:T2:0");
  EXPECT_EQ(
      (std::vector<DeviceType>{DeviceType("T1"), DeviceType("T2"),
                               DeviceType(DEVICE_GPU), DeviceType(DEVICE_CPU)}),
      types());
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

}  // namespace
}  // namespace tensorflow
