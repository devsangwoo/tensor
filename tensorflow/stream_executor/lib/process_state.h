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
#ifndef TENSORFLOW_STREAM_EXECUTOR_LIB_PROCESS_STATE_H_
#define TENSORFLOW_STREAM_EXECUTOR_LIB_PROCESS_STATE_H_

#include "tensorflow/stream_executor/platform/port.h"

<<<<<<< HEAD
namespace stream_executor {
=======
namespace perftools {
namespace gputools {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
namespace port {

string Hostname();
bool GetCurrentDirectory(string* dir);

}  // namespace port
<<<<<<< HEAD
}  // namespace stream_executor
=======
}  // namespace gputools
}  // namespace perftools
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#endif  // TENSORFLOW_STREAM_EXECUTOR_LIB_PROCESS_STATE_H_
