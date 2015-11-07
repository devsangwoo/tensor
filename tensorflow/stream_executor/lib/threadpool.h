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
#ifndef TENSORFLOW_STREAM_EXECUTOR_LIB_THREADPOOL_H_
#define TENSORFLOW_STREAM_EXECUTOR_LIB_THREADPOOL_H_

#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/stream_executor/lib/env.h"
<<<<<<< HEAD
#include "tensorflow/stream_executor/lib/thread_options.h"

namespace stream_executor {
namespace port {

using tensorflow::Thread;
using tensorflow::thread::ThreadPool;

}  // namespace port
}  // namespace stream_executor
=======
#include "tensorflow/stream_executor/lib/notification.h"
#include "tensorflow/stream_executor/lib/thread_options.h"

namespace perftools {
namespace gputools {
namespace port {

using tensorflow::thread::ThreadPool;

}  // namespace port
}  // namespace gputools
}  // namespace perftools
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#endif  // TENSORFLOW_STREAM_EXECUTOR_LIB_THREADPOOL_H_
