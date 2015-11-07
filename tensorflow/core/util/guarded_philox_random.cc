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
#include "tensorflow/core/util/guarded_philox_random.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {

Status GuardedPhiloxRandom::Init(OpKernelConstruction* context) {
  // Grab seed Attrs.
  int64 seed, seed2;
  auto status = context->GetAttr("seed", &seed);
  if (!status.ok()) return status;
  status = context->GetAttr("seed2", &seed2);
  if (!status.ok()) return status;

  // Initialize with the given seeds
  Init(seed, seed2);
  return Status::OK();
}

void GuardedPhiloxRandom::Init(int64 seed, int64 seed2) {
  CHECK(!initialized_);
  if (seed == 0 && seed2 == 0) {
    // If both seeds are unspecified, use completely random seeds.
    seed = random::New64();
    seed2 = random::New64();
  }
  mutex_lock lock(mu_);
  generator_ = random::PhiloxRandom(seed, seed2);
  initialized_ = true;
}

<<<<<<< HEAD
void GuardedPhiloxRandom::Init(random::PhiloxRandom::ResultType counter,
                               random::PhiloxRandom::Key key) {
  CHECK(!initialized_);
  mutex_lock lock(mu_);
  generator_ = random::PhiloxRandom(counter, key);
  initialized_ = true;
}

=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
random::PhiloxRandom GuardedPhiloxRandom::ReserveSamples128(int64 samples) {
  CHECK(initialized_);
  mutex_lock lock(mu_);
  auto local = generator_;
  generator_.Skip(samples);
  return local;
}

}  // namespace tensorflow
