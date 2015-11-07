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

#include "tensorflow/core/kernels/initializable_lookup_table.h"
=======
#include "tensorflow/core/kernels/initializable_lookup_table.h"

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace lookup {

<<<<<<< HEAD
Status InitializableLookupTable::Find(OpKernelContext* ctx, const Tensor& keys,
                                      Tensor* values,
=======
Status InitializableLookupTable::Find(const Tensor& keys, Tensor* values,
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                                      const Tensor& default_value) {
  if (!is_initialized()) {
    return errors::FailedPrecondition("Table not initialized.");
  }
<<<<<<< HEAD
  // Do not let the use migrate before the check;  table is used without
  // a lock by the readers.
  std::atomic_thread_fence(std::memory_order_acquire);
  return DoFind(keys, values, default_value);
}

Status InitializableLookupTable::ImportValues(OpKernelContext* ctx,
                                              const Tensor& keys,
                                              const Tensor& values) {
  lookup::KeyValueTensorIterator iter(&keys, &values);
  return Initialize(iter);
}

=======
  TF_RETURN_IF_ERROR(CheckFindArguments(keys, *values, default_value));
  return DoFind(keys, values, default_value);
}

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
Status InitializableLookupTable::Initialize(InitTableIterator& iter) {
  if (!iter.Valid()) {
    return iter.status();
  }
<<<<<<< HEAD
  TF_RETURN_IF_ERROR(
      CheckKeyAndValueTensorsForInsert(iter.keys(), iter.values()));

  mutex_lock l(mu_);
  if (is_initialized()) {
    bool result;
    TF_RETURN_IF_ERROR(AreEntriesSame(iter, &result));
    // If the table is already initialized, we make sure that the entries in the
    // table are the same that we want to initialize the table with.
    if (!result) {
      return errors::FailedPrecondition(
          "Table was already initialized with "
          "different data.");
    } else {
      return Status::OK();
    }
  }
  TF_RETURN_IF_ERROR(DoLazyPrepare([&iter]() { return iter.total_size(); }));
=======
  TF_RETURN_IF_ERROR(CheckKeyAndValueTensors(iter.keys(), iter.values()));

  mutex_lock l(mu_);
  if (is_initialized()) {
    return errors::FailedPrecondition("Table already initialized.");
  }

  TF_RETURN_IF_ERROR(DoPrepare(iter.total_size()));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  while (iter.Valid()) {
    TF_RETURN_IF_ERROR(DoInsert(iter.keys(), iter.values()));
    iter.Next();
  }
  if (!errors::IsOutOfRange(iter.status())) {
    return iter.status();
  }
<<<<<<< HEAD

  is_initialized_.store(true, std::memory_order_release);
  return Status::OK();
}

Status InitializableLookupTable::AreEntriesSame(const InitTableIterator& iter,
                                                bool* result) {
  *result = iter.total_size() == size();
=======
  is_initialized_ = true;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  return Status::OK();
}

}  // namespace lookup
}  // namespace tensorflow
