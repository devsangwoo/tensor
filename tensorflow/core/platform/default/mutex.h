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

#ifndef TENSORFLOW_CORE_PLATFORM_DEFAULT_MUTEX_H_
#define TENSORFLOW_CORE_PLATFORM_DEFAULT_MUTEX_H_

// IWYU pragma: private, include "third_party/tensorflow/core/platform/mutex.h"
// IWYU pragma: friend third_party/tensorflow/core/platform/mutex.h

namespace tensorflow {

namespace internal {
std::cv_status wait_until_system_clock(
    CVData *cv_data, MuData *mu_data,
    const std::chrono::system_clock::time_point timeout_time);
}  // namespace internal

template <class Rep, class Period>
std::cv_status condition_variable::wait_for(
    mutex_lock &lock, std::chrono::duration<Rep, Period> dur) {
  return internal::wait_until_system_clock(
      &this->cv_, &lock.mutex()->mu_, std::chrono::system_clock::now() + dur);
=======
#ifndef TENSORFLOW_PLATFORM_DEFAULT_MUTEX_H_
#define TENSORFLOW_PLATFORM_DEFAULT_MUTEX_H_

#include <chrono>
#include <condition_variable>
#include <mutex>

namespace tensorflow {

enum LinkerInitialized { LINKER_INITIALIZED };

// A class that wraps around the std::mutex implementation, only adding an
// additional LinkerInitialized constructor interface.
class mutex : public std::mutex {
 public:
  mutex() {}
  // The default implementation of std::mutex is safe to use after the linker
  // initializations
  explicit mutex(LinkerInitialized x) {}
};

using std::condition_variable;
typedef std::unique_lock<std::mutex> mutex_lock;

inline ConditionResult WaitForMilliseconds(mutex_lock* mu,
                                           condition_variable* cv, int64 ms) {
  std::cv_status s = cv->wait_for(*mu, std::chrono::milliseconds(ms));
  return (s == std::cv_status::timeout) ? kCond_Timeout : kCond_MaybeNotified;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

}  // namespace tensorflow

<<<<<<< HEAD
#endif  // TENSORFLOW_CORE_PLATFORM_DEFAULT_MUTEX_H_
=======
#endif  // TENSORFLOW_PLATFORM_DEFAULT_MUTEX_H_
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
