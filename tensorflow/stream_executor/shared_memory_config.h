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
// This file defines a uniform interface to configuration options for shared
// memory for supported devices. As with many StreamExecutor-supported features,
// support for the options defined herein is device-dependent.
#ifndef TENSORFLOW_STREAM_EXECUTOR_SHARED_MEMORY_CONFIG_H_
#define TENSORFLOW_STREAM_EXECUTOR_SHARED_MEMORY_CONFIG_H_

<<<<<<< HEAD
namespace stream_executor {
=======
namespace perftools {
namespace gputools {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

// SharedMemoryConfig enum describes potential widths of shared memory banks for
// a device or kernel.
enum class SharedMemoryConfig {
  kDefault,    // Use the device default configuration.
  kFourByte,   // Sets shared memory banks to be four bytes wide.
  kEightByte,  // Sets shared memory banks to be eight bytes wide.
};

<<<<<<< HEAD
}  // namespace stream_executor
=======
}  // namespace gputools
}  // namespace perftools
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#endif  // TENSORFLOW_STREAM_EXECUTOR_SHARED_MEMORY_CONFIG_H_
