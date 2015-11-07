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

#ifndef TENSORFLOW_CORE_KERNELS_INITIALIZABLE_LOOKUP_TABLE_H_
#define TENSORFLOW_CORE_KERNELS_INITIALIZABLE_LOOKUP_TABLE_H_

#include <atomic>

#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/platform/macros.h"
=======
#ifndef TENSORFLOW_KERNELS_INITIALIZABLE_LOOKUP_TABLE_H_
#define TENSORFLOW_KERNELS_INITIALIZABLE_LOOKUP_TABLE_H_

#include "tensorflow/core/framework/lookup_interface.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

namespace tensorflow {
namespace lookup {

// Base class for lookup tables that require initialization.
class InitializableLookupTable : public LookupInterface {
 public:
  class InitTableIterator;

  // Performs batch lookups, for every element in the key tensor, Find returns
  // the corresponding value into the values tensor.
  // If an element is not present in the table, the given default value is used.
  //
  // For tables that require initialization, `Find` is available once the table
  // is marked as initialized.
  //
  // Returns the following statuses:
  // - OK: when the find finishes successfully.
  // - FailedPrecondition: if the table is not initialized.
  // - InvalidArgument: if any of the preconditions on the lookup key or value
  //   fails.
  // - In addition, other implementations may provide another non-OK status
  //   specific to their failure modes.
<<<<<<< HEAD
  Status Find(OpKernelContext* ctx, const Tensor& keys, Tensor* values,
              const Tensor& default_value) final;

  // Returns errors::Unimplemented.
  Status Insert(OpKernelContext* ctx, const Tensor& keys,
                const Tensor& values) final {
    return errors::Unimplemented(
        "Insert not supported by InitializableLookupTable implementations");
  }

  // Returns errors::Unimplemented.
  Status Remove(OpKernelContext* ctx, const Tensor& keys) final {
    return errors::Unimplemented(
        "Remove not supported by InitializableLookupTable implementations");
  }

  Status ExportValues(OpKernelContext* context) override {
    return errors::Unimplemented(
        "ExportValues not supported by InitializableLookupTable "
        "implementations");
  }

  Status ImportValues(OpKernelContext* ctx, const Tensor& keys,
                      const Tensor& values) final;

  TensorShape key_shape() const final { return TensorShape(); }

  TensorShape value_shape() const final { return TensorShape(); }

  // Returns whether the table was initialized and is ready to serve lookups.
  bool is_initialized() const {
    return is_initialized_.load(std::memory_order_acquire);
  }
=======
  Status Find(const Tensor& keys, Tensor* values,
              const Tensor& default_value) final;

  // Returns whether the table was initialized and is ready to serve lookups.
  bool is_initialized() const { return is_initialized_; }
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  // Initializes the table from the given init table iterator.
  //
  // Atomically, this operation prepares the table, populates it with the given
  // iterator, and mark the table as initialized.
  //
  // Returns the following statuses:
  // - OK: when the initialization was successful.
  // - InvalidArgument: if any of the preconditions on the lookup key or value
  //   fails.
  // - FailedPrecondition: if the table is already initialized and
  //   fail_if_initialized is set to true.
  // - In addition, other implementations may provide another non-OK status
  //   specific to their failure modes.
  Status Initialize(InitTableIterator& iter);

  // Basic iterator to initialize lookup tables.
  // It yields a sequence of pairs of `keys()` and `values()` Tensors, so that
  // the consumer may insert key-value pairs in batches.
  //
  // Then the iterator is exhausted, valid returns false and status returns
  // Status::OutOfRange.
<<<<<<< HEAD
  //
  // This class is Thread-unsafe.
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  class InitTableIterator {
   public:
    InitTableIterator() {}

    virtual ~InitTableIterator() {}

    // Prepares the next batch of key and value tensors.
    virtual void Next() = 0;

    // Returns true if keys and values point to valid tensors.
    virtual bool Valid() const = 0;

    // Returns a tensor that contains the current batch of 'key' values.
    virtual const Tensor& keys() const = 0;

    // Returns a tensor that contains the current batch of 'value' values.
    virtual const Tensor& values() const = 0;

<<<<<<< HEAD
    // Returns an error if one has occurred, otherwise returns Status::OK.
    virtual Status status() const = 0;

    // Returns the total number of elements that the iterator will produce.
    // It might return -1 in case of error.
=======
    // Returns an error if one has occurred, otherwire returns Status::OK.
    virtual Status status() const = 0;

    // Returns the total number of elements that the iterator will produce.
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    virtual int64 total_size() const = 0;

   private:
    TF_DISALLOW_COPY_AND_ASSIGN(InitTableIterator);
  };

<<<<<<< HEAD
  InitializableLookupTable* GetInitializableLookupTable() override {
    return this;
  }

=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
 protected:
  // Prepares and allocates the underlying data structure to store the given
  // number of expected elements.
  virtual Status DoPrepare(size_t expected_num_elements) = 0;

<<<<<<< HEAD
  // Same as DoPrepare() but derived implementations might choose to skip
  // calling get_expected_num_elements if size is not needed for DoPrepare.
  virtual Status DoLazyPrepare(
      std::function<int64(void)> get_expected_num_elements) {
    int64 expected_num_elements = get_expected_num_elements();
    if (expected_num_elements < 0) {
      return errors::FailedPrecondition("Got negative expected_num_elements.");
    }
    return DoPrepare(expected_num_elements);
  }

=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  // Populates the table in batches given keys and values as tensors into the
  // underlying data structure.
  virtual Status DoInsert(const Tensor& keys, const Tensor& values) = 0;

  // Performs the batch find operation on the underlying data structure.
  virtual Status DoFind(const Tensor& keys, Tensor* values,
                        const Tensor& default_value) = 0;

<<<<<<< HEAD
  virtual Status AreEntriesSame(const InitTableIterator& iter, bool* result);

  mutex mu_;

 private:
  std::atomic<bool> is_initialized_{false};
};

// Iterator to initialize tables given 'keys' and 'values' tensors.
//
// The two tensors are returned in the first iteration. It doesn't loop
// over each element of the tensor since insertions in the lookup table can
// process batches.
class KeyValueTensorIterator
    : public InitializableLookupTable::InitTableIterator {
 public:
  // keys and values are not owned by the iterator.
  explicit KeyValueTensorIterator(const Tensor* keys, const Tensor* values)
      : keys_(keys), values_(values), valid_(true), status_(Status::OK()) {
    TensorShape key_shape = keys_->shape();
    if (!key_shape.IsSameSize(values_->shape())) {
      valid_ = false;
      status_ = errors::InvalidArgument(
          "keys and values should have the same dimension.",
          key_shape.DebugString(), " vs ", values_->shape().DebugString());
    }
    if (key_shape.num_elements() == 0) {
      valid_ = false;
      status_ =
          errors::InvalidArgument("keys and values cannot be empty tensors.");
    }
  }

  bool Valid() const override { return valid_; }

  void Next() override {
    valid_ = false;
    status_ = errors::OutOfRange("No more data.");
  }

  const Tensor& keys() const override { return *keys_; }

  const Tensor& values() const override { return *values_; }

  Status status() const override { return status_; }

  int64 total_size() const override {
    return keys_ == nullptr ? -1 : keys_->NumElements();
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(KeyValueTensorIterator);

  const Tensor* keys_;    // Doesn't own it.
  const Tensor* values_;  // Doesn't own it.
  bool valid_;            // true if the iterator points to an existing range.
  Status status_;
=======
  mutex mu_;
  bool is_initialized_ = false;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
};

}  // namespace lookup
}  // namespace tensorflow

<<<<<<< HEAD
#endif  // TENSORFLOW_CORE_KERNELS_INITIALIZABLE_LOOKUP_TABLE_H_
=======
#endif  // TENSORFLOW_KERNELS_INITIALIZABLE_LOOKUP_TABLE_H_
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
