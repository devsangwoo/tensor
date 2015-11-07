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
#include "tensorflow/core/kernels/lookup_table_op.h"
#define EIGEN_USE_THREADS

#include <string>
<<<<<<< HEAD
#include <type_traits>
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#include <utility>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
<<<<<<< HEAD
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/kernels/initializable_lookup_table.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
=======
#include "tensorflow/core/kernels/initializable_lookup_table.h"
#include "tensorflow/core/lib/gtl/map_util.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#include "tensorflow/core/lib/hash/hash.h"

namespace tensorflow {
namespace lookup {

// Lookup table that wraps an unordered_map, where the key and value data type
<<<<<<< HEAD
// is specified. Each individual value must be a scalar. If vector values are
// required, use MutableHashTableOfTensors.
//
// This table is mutable and thread safe - Insert can be called at any time.
//
// Sample use case:
//
// MutableHashTableOfScalars<int64, int64> table;  // int64 -> int64.
// // Populate the table, elements could be added in one or multiple calls.
// table.Insert(key_tensor, value_tensor); // Populate the table.
=======
// is specified.
//
// This table is recommened for any variations to key values.
//
// For look up, the table is required to be initialized (allocated
// and populated). Once the table is marked as initialized it becomes read-only.
//
// Sample use case:
//
// HashTable<int64, int64> table;  // int64 -> int64.
// table.Prepare(10); // Prepare the underlying data structure, the number of
//                    // elements is required by interface, but not used.
// // Populate the table, elements could be added in one or multiple calls.
// table.Insert(key_tensor, value_tensor); // Populate the table.
// ...
// table.set_is_initialized();
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
//
// table.Find(in_t, &out_t, default_t)
//
template <class K, class V>
<<<<<<< HEAD
class MutableHashTableOfScalars final : public LookupInterface {
 public:
  MutableHashTableOfScalars(OpKernelContext* ctx, OpKernel* kernel) {}

  size_t size() const override {
    tf_shared_lock l(mu_);
    return table_.size();
  }

  Status Find(OpKernelContext* ctx, const Tensor& key, Tensor* value,
              const Tensor& default_value) override {
    const V default_val = default_value.flat<V>()(0);
    const auto key_values = key.flat<K>();
    auto value_values = value->flat<V>();

    tf_shared_lock l(mu_);
    for (int64 i = 0; i < key_values.size(); ++i) {
      value_values(i) = gtl::FindWithDefault(
          table_, SubtleMustCopyIfIntegral(key_values(i)), default_val);
    }

    return Status::OK();
  }

  Status DoInsert(bool clear, const Tensor& keys, const Tensor& values) {
    const auto key_values = keys.flat<K>();
    const auto value_values = values.flat<V>();

    mutex_lock l(mu_);
    if (clear) {
      table_.clear();
    }
    for (int64 i = 0; i < key_values.size(); ++i) {
      gtl::InsertOrUpdate(&table_, SubtleMustCopyIfIntegral(key_values(i)),
                          SubtleMustCopyIfIntegral(value_values(i)));
    }
    return Status::OK();
  }

  Status Insert(OpKernelContext* ctx, const Tensor& keys,
                const Tensor& values) override {
    return DoInsert(false, keys, values);
  }

  Status Remove(OpKernelContext* ctx, const Tensor& keys) override {
    const auto key_values = keys.flat<K>();

    mutex_lock l(mu_);
    for (int64 i = 0; i < key_values.size(); ++i) {
      table_.erase(SubtleMustCopyIfIntegral(key_values(i)));
    }
    return Status::OK();
  }

  Status ImportValues(OpKernelContext* ctx, const Tensor& keys,
                      const Tensor& values) override {
    return DoInsert(true, keys, values);
  }

  Status ExportValues(OpKernelContext* ctx) override {
    tf_shared_lock l(mu_);
    int64 size = table_.size();

    Tensor* keys;
    Tensor* values;
    TF_RETURN_IF_ERROR(
        ctx->allocate_output("keys", TensorShape({size}), &keys));
    TF_RETURN_IF_ERROR(
        ctx->allocate_output("values", TensorShape({size}), &values));

    auto keys_data = keys->flat<K>();
    auto values_data = values->flat<V>();
    int64 i = 0;
    for (auto it = table_.begin(); it != table_.end(); ++it, ++i) {
      keys_data(i) = it->first;
      values_data(i) = it->second;
    }
    return Status::OK();
  }
=======
class HashTable : public InitializableLookupTable {
 public:
  size_t size() const override { return table_ ? table_->size() : 0; }
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  DataType key_dtype() const override { return DataTypeToEnum<K>::v(); }

  DataType value_dtype() const override { return DataTypeToEnum<V>::v(); }

<<<<<<< HEAD
  TensorShape key_shape() const final { return TensorShape(); }

  TensorShape value_shape() const override { return TensorShape(); }

  int64 MemoryUsed() const override {
    int64 ret = 0;
    tf_shared_lock l(mu_);
    for (unsigned i = 0; i < table_.bucket_count(); ++i) {
      size_t bucket_size = table_.bucket_size(i);
      if (bucket_size == 0) {
        ret++;
      } else {
        ret += bucket_size;
      }
    }
    return sizeof(MutableHashTableOfScalars) + ret;
  }

 private:
  mutable mutex mu_;
  std::unordered_map<K, V> table_ GUARDED_BY(mu_);
};

// Lookup table that wraps an unordered_map. Behaves identical to
// MutableHashTableOfScalars except that each value must be a vector.
template <class K, class V>
class MutableHashTableOfTensors final : public LookupInterface {
 public:
  MutableHashTableOfTensors(OpKernelContext* ctx, OpKernel* kernel) {
    OP_REQUIRES_OK(ctx,
                   GetNodeAttr(kernel->def(), "value_shape", &value_shape_));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(value_shape_),
        errors::InvalidArgument("Default value must be a vector, got shape ",
                                value_shape_.DebugString()));
  }

  size_t size() const override {
    tf_shared_lock l(mu_);
    return table_.size();
  }

  Status Find(OpKernelContext* ctx, const Tensor& key, Tensor* value,
              const Tensor& default_value) override {
    const auto default_flat = default_value.flat<V>();
    const auto key_values = key.flat<K>();
    auto value_values = value->flat_inner_dims<V, 2>();
    int64 value_dim = value_shape_.dim_size(0);

    tf_shared_lock l(mu_);
    for (int64 i = 0; i < key_values.size(); ++i) {
      ValueArray* value_vec =
          gtl::FindOrNull(table_, SubtleMustCopyIfIntegral(key_values(i)));
      if (value_vec != nullptr) {
        for (int64 j = 0; j < value_dim; j++) {
          value_values(i, j) = value_vec->at(j);
        }
      } else {
        for (int64 j = 0; j < value_dim; j++) {
          value_values(i, j) = default_flat(j);
        }
      }
    }

    return Status::OK();
  }

  Status DoInsert(bool clear, const Tensor& keys, const Tensor& values) {
    const auto key_values = keys.flat<K>();
    const auto value_values = values.flat_inner_dims<V, 2>();
    int64 value_dim = value_shape_.dim_size(0);

    mutex_lock l(mu_);
    if (clear) {
      table_.clear();
    }
    for (int64 i = 0; i < key_values.size(); ++i) {
      ValueArray value_vec;
      for (int64 j = 0; j < value_dim; j++) {
        V value = value_values(i, j);
        value_vec.push_back(value);
      }
      gtl::InsertOrUpdate(&table_, SubtleMustCopyIfIntegral(key_values(i)),
                          value_vec);
    }
    return Status::OK();
  }

  Status Insert(OpKernelContext* ctx, const Tensor& keys,
                const Tensor& values) override {
    return DoInsert(false, keys, values);
  }

  Status Remove(OpKernelContext* ctx, const Tensor& keys) override {
    const auto key_values = keys.flat<K>();

    mutex_lock l(mu_);
    for (int64 i = 0; i < key_values.size(); ++i) {
      table_.erase(SubtleMustCopyIfIntegral(key_values(i)));
    }
    return Status::OK();
  }

  Status ImportValues(OpKernelContext* ctx, const Tensor& keys,
                      const Tensor& values) override {
    return DoInsert(true, keys, values);
  }

  Status ExportValues(OpKernelContext* ctx) override {
    tf_shared_lock l(mu_);
    int64 size = table_.size();
    int64 value_dim = value_shape_.dim_size(0);

    Tensor* keys;
    Tensor* values;
    TF_RETURN_IF_ERROR(
        ctx->allocate_output("keys", TensorShape({size}), &keys));
    TF_RETURN_IF_ERROR(ctx->allocate_output(
        "values", TensorShape({size, value_dim}), &values));

    auto keys_data = keys->flat<K>();
    auto values_data = values->matrix<V>();
    int64 i = 0;
    for (auto it = table_.begin(); it != table_.end(); ++it, ++i) {
      K key = it->first;
      ValueArray value = it->second;
      keys_data(i) = key;
      for (int64 j = 0; j < value_dim; j++) {
        values_data(i, j) = value[j];
      }
    }
    return Status::OK();
  }

  DataType key_dtype() const override { return DataTypeToEnum<K>::v(); }

  DataType value_dtype() const override { return DataTypeToEnum<V>::v(); }

  TensorShape key_shape() const final { return TensorShape(); }

  TensorShape value_shape() const override { return value_shape_; }

  int64 MemoryUsed() const override {
    int64 ret = 0;
    tf_shared_lock l(mu_);
    for (unsigned i = 0; i < table_.bucket_count(); ++i) {
      size_t bucket_size = table_.bucket_size(i);
      if (bucket_size == 0) {
        ret++;
      } else {
        ret += bucket_size;
      }
    }
    return sizeof(MutableHashTableOfTensors) + ret;
  }

 private:
  TensorShape value_shape_;
  mutable mutex mu_;
  typedef gtl::InlinedVector<V, 4> ValueArray;
  std::unordered_map<K, ValueArray> table_ GUARDED_BY(mu_);
};

namespace {

template <typename T>
inline uint64 HashScalar(const T& key) {
  return static_cast<uint64>(key);
}

inline uint64 HashScalar(const tstring& key) { return Hash64(key); }

// If the given shape is a scalar return {1} instead. Otherwise leave it alone.
TensorShape MaybeVectorizeShape(const TensorShape& shape) {
  if (shape.dims() == 0) {
    return TensorShape({1});
  }
  return shape;
}

}  // namespace

// Modeled after densehashtable in https://github.com/sparsehash/sparsehash
template <class K, class V>
class MutableDenseHashTable final : public LookupInterface {
 public:
  MutableDenseHashTable(OpKernelContext* ctx, OpKernel* kernel) {
    OP_REQUIRES_OK(
        ctx, GetNodeAttr(kernel->def(), "max_load_factor", &max_load_factor_));
    OP_REQUIRES(ctx, max_load_factor_ > 0 && max_load_factor_ < 1,
                errors::InvalidArgument(
                    "max_load_factor must be between 0 and 1, got: ",
                    max_load_factor_));

    OP_REQUIRES_OK(ctx,
                   GetNodeAttr(kernel->def(), "value_shape", &value_shape_));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(value_shape_) ||
                    TensorShapeUtils::IsVector(value_shape_),
                errors::InvalidArgument(
                    "Empty value must be a scalar or a vector, got shape ",
                    value_shape_.DebugString()));

    const Tensor* empty_key_input;
    OP_REQUIRES_OK(ctx, ctx->input("empty_key", &empty_key_input));
    key_shape_ = empty_key_input->shape();
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(key_shape_) ||
                    TensorShapeUtils::IsVector(key_shape_),
                errors::InvalidArgument(
                    "Empty key must be a scalar or a vector, got shape ",
                    key_shape_.DebugString()));
    empty_key_ = PersistentTensor(*empty_key_input);
    empty_key_hash_ = HashKey(
        empty_key_input->template shaped<K, 2>({1, key_shape_.num_elements()}),
        0);

    const Tensor* deleted_key_input;
    OP_REQUIRES_OK(ctx, ctx->input("deleted_key", &deleted_key_input));
    OP_REQUIRES(ctx, key_shape_.IsSameSize(deleted_key_input->shape()),
                errors::InvalidArgument(
                    "Empty and deleted keys must have same shape, got shapes: ",
                    key_shape_.DebugString(), " and ",
                    deleted_key_input->shape().DebugString()));
    deleted_key_ = PersistentTensor(*deleted_key_input);
    deleted_key_hash_ = HashKey(deleted_key_input->template shaped<K, 2>(
                                    {1, key_shape_.num_elements()}),
                                0);

    if (empty_key_hash_ == deleted_key_hash_) {
      const int64 key_size = key_shape_.num_elements();
      const auto empty_key_matrix =
          empty_key_.AccessTensor(ctx)->template shaped<K, 2>({1, key_size});
      const auto deleted_key_matrix =
          deleted_key_.AccessTensor(ctx)->template shaped<K, 2>({1, key_size});
      OP_REQUIRES(
          ctx, !IsEqualKey(empty_key_matrix, 0, deleted_key_matrix, 0),
          errors::InvalidArgument("Empty and deleted keys cannot be equal"));
    }

    int64 initial_num_buckets;
    OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "initial_num_buckets",
                                    &initial_num_buckets));
    OP_REQUIRES_OK(ctx, AllocateBuckets(ctx, initial_num_buckets));
  }

  size_t size() const override LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    return num_entries_;
  }

  Status Find(OpKernelContext* ctx, const Tensor& key, Tensor* value,
              const Tensor& default_value) override LOCKS_EXCLUDED(mu_) {
    const int64 num_elements = (key.dims() == 0) ? 1 : key.dim_size(0);
    const int64 key_size = key_shape_.num_elements();
    const int64 value_size = value_shape_.num_elements();
    if (key.NumElements() != num_elements * key_size) {
      TensorShape expected_shape({num_elements});
      expected_shape.AppendShape(key_shape_);
      return errors::InvalidArgument("Expected key shape ",
                                     expected_shape.DebugString(), " got ",
                                     key.shape().DebugString());
    }
    const auto key_matrix = key.shaped<K, 2>({num_elements, key_size});
    auto value_matrix = value->shaped<V, 2>({num_elements, value_size});
    const auto default_flat = default_value.flat<V>();

    tf_shared_lock l(mu_);
    const auto key_buckets_matrix =
        key_buckets_.AccessTensor(ctx)->template matrix<K>();
    const auto value_buckets_matrix =
        value_buckets_.AccessTensor(ctx)->template matrix<V>();
    const auto empty_key_matrix =
        empty_key_.AccessTensor(ctx)->template shaped<K, 2>({1, key_size});
    const auto deleted_key_matrix =
        deleted_key_.AccessTensor(ctx)->template shaped<K, 2>({1, key_size});
    const int64 bit_mask = num_buckets_ - 1;
    // TODO(andreasst): parallelize using work_sharder
    for (int64 i = 0; i < num_elements; ++i) {
      const uint64 key_hash = HashKey(key_matrix, i);
      if (empty_key_hash_ == key_hash &&
          IsEqualKey(empty_key_matrix, 0, key_matrix, i)) {
        return errors::InvalidArgument(
            "Using the empty_key as a table key is not allowed");
      }
      if (deleted_key_hash_ == key_hash &&
          IsEqualKey(deleted_key_matrix, 0, key_matrix, i)) {
        return errors::InvalidArgument(
            "Using the deleted_key as a table key is not allowed");
      }
      int64 bucket_index = key_hash & bit_mask;
      int64 num_probes = 0;
      while (true) {
        if (IsEqualKey(key_buckets_matrix, bucket_index, key_matrix, i)) {
          for (int64 j = 0; j < value_size; ++j) {
            // TODO(andreasst): check if we can get rid of SubtleMustCopy
            // here and elsewhere in this file.
            value_matrix(i, j) =
                SubtleMustCopyIfIntegral(value_buckets_matrix(bucket_index, j));
          }
          break;
        }
        if (IsEqualKey(key_buckets_matrix, bucket_index, empty_key_matrix, 0)) {
          for (int64 j = 0; j < value_size; ++j) {
            value_matrix(i, j) = SubtleMustCopyIfIntegral(default_flat(j));
          }
          break;
        }
        ++num_probes;
        bucket_index =
            (bucket_index + num_probes) & bit_mask;  // quadratic probing
        if (num_probes >= num_buckets_) {
          return errors::Internal(
              "Internal error in MutableDenseHashTable lookup");
        }
      }
    }
    return Status::OK();
  }

  Status Insert(OpKernelContext* ctx, const Tensor& key,
                const Tensor& value) override LOCKS_EXCLUDED(mu_) {
    const int64 batch_size = (key.dims() == 0) ? 1 : key.dim_size(0);
    if (key.NumElements() != batch_size * key_shape_.num_elements()) {
      TensorShape expected_shape({batch_size});
      expected_shape.AppendShape(key_shape_);
      return errors::InvalidArgument("Expected key shape ",
                                     expected_shape.DebugString(), " got ",
                                     key.shape().DebugString());
    }
    mutex_lock l(mu_);
    // For simplicity we assume that all keys in the input result in inserts
    // rather than updates. That means we may grow the table even though we
    // don't need to. As long as the number of keys inserted in one call is
    // small compared to the size of the map, the impact of this is minimal.
    const int64 pending_num_entries = num_entries_ + batch_size;
    if (pending_num_entries > num_buckets_ * max_load_factor_) {
      int64 new_num_buckets = num_buckets_;
      do {
        new_num_buckets <<= 1;
      } while (pending_num_entries > new_num_buckets * max_load_factor_);
      TF_RETURN_IF_ERROR(Rebucket(ctx, new_num_buckets));
    }
    return DoInsert(ctx, key, value, false);
  }

  Status Remove(OpKernelContext* ctx, const Tensor& key) override
      LOCKS_EXCLUDED(mu_) {
    if (key.NumElements() != key.dim_size(0) * key_shape_.num_elements()) {
      TensorShape expected_shape({key.dim_size(0)});
      expected_shape.AppendShape(key_shape_);
      return errors::InvalidArgument("Expected key shape ",
                                     expected_shape.DebugString(), " got ",
                                     key.shape().DebugString());
    }
    mutex_lock l(mu_);
    return DoRemove(ctx, key);
  }

  Status ImportValues(OpKernelContext* ctx, const Tensor& keys,
                      const Tensor& values) override LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    num_buckets_ = keys.dim_size(0);
    key_buckets_ = PersistentTensor(keys);
    value_buckets_ = PersistentTensor(values);
    // Count the number of keys that are not the empty_key or deleted_key.
    // This requires iterating through the whole table but that is OK as we
    // only execute it during checkpoint restore.
    num_entries_ = 0;
    const auto empty_key_tensor =
        empty_key_.AccessTensor(ctx)->template shaped<K, 2>(
            {1, key_shape_.num_elements()});
    const auto deleted_key_tensor =
        deleted_key_.AccessTensor(ctx)->template shaped<K, 2>(
            {1, key_shape_.num_elements()});
    const auto key_buckets_tensor =
        key_buckets_.AccessTensor(ctx)->template matrix<K>();
    for (int64 i = 0; i < num_buckets_; ++i) {
      if (!IsEqualKey(key_buckets_tensor, i, empty_key_tensor, 0) &&
          !IsEqualKey(key_buckets_tensor, i, deleted_key_tensor, 0)) {
        ++num_entries_;
=======
 protected:
  Status DoPrepare(size_t unused) override {
    if (is_initialized_) {
      return errors::Aborted("HashTable already initialized.");
    }
    if (!table_) {
      table_ = std::unique_ptr<std::unordered_map<K, V>>(
          new std::unordered_map<K, V>());
    }
    return Status::OK();
  };

  Status DoInsert(const Tensor& keys, const Tensor& values) override {
    if (!table_) {
      return errors::FailedPrecondition("HashTable is not prepared.");
    }

    const auto key_values = keys.flat<K>();
    const auto value_values = values.flat<V>();
    for (size_t i = 0; i < key_values.size(); ++i) {
      const K& key = key_values(i);
      const V& value = value_values(i);
      const V& previous_value = gtl::LookupOrInsert(table_.get(), key, value);
      if (previous_value != value) {
        return errors::FailedPrecondition(
            "HashTable has different value for same key. Key ", key, " has ",
            previous_value, " and trying to add value ", value);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      }
    }
    return Status::OK();
  }

<<<<<<< HEAD
  Status ExportValues(OpKernelContext* ctx) override LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    Tensor key_buckets_tensor = *key_buckets_.AccessTensor(ctx);
    Tensor value_buckets_tensor = *value_buckets_.AccessTensor(ctx);
    TF_RETURN_IF_ERROR(ctx->set_output("keys", key_buckets_tensor));
    TF_RETURN_IF_ERROR(ctx->set_output("values", value_buckets_tensor));
    return Status::OK();
  }

  Status CheckKeyAndValueTensorsForImport(const Tensor& keys,
                                          const Tensor& values) override {
    TF_RETURN_IF_ERROR(CheckKeyAndValueTypes(keys, values));
    TF_RETURN_IF_ERROR(CheckKeyShape(keys.shape()));

    // The storage format in key_buckets_ and value_buckets_ is always vectors,
    // even if the inputs are scalars. This is what eventually gets exported
    // and is expected by the import method as well.
    TensorShape key_shape = MaybeVectorizeShape(key_shape_);
    TensorShape value_shape = MaybeVectorizeShape(value_shape_);

    // Compute the final expected shape of the value by starting with the shape
    // of all keys, removing the dimensions particular to each key and then
    // appending the shape of a single value.
    TensorShape expected_value_shape = keys.shape();
    expected_value_shape.RemoveLastDims(key_shape.dims());
    expected_value_shape.AppendShape(value_shape);
    if (values.shape() != expected_value_shape) {
      return errors::InvalidArgument(
          "Expected shape ", expected_value_shape.DebugString(),
          " for value, got ", values.shape().DebugString());
=======
  Status DoFind(const Tensor& key, Tensor* value,
                const Tensor& default_value) override {
    const V default_val = default_value.flat<V>()(0);
    const auto key_values = key.flat<K>();
    auto value_values = value->flat<V>();

    for (size_t i = 0; i < key_values.size(); ++i) {
      value_values(i) =
          gtl::FindWithDefault(*table_, key_values(i), default_val);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    }
    return Status::OK();
  }

<<<<<<< HEAD
  DataType key_dtype() const override { return DataTypeToEnum<K>::v(); }

  DataType value_dtype() const override { return DataTypeToEnum<V>::v(); }

  TensorShape key_shape() const override { return key_shape_; }

  TensorShape value_shape() const override { return value_shape_; }

  int64 MemoryUsed() const override {
    tf_shared_lock l(mu_);
    return sizeof(MutableDenseHashTable) + key_buckets_.AllocatedBytes() +
           value_buckets_.AllocatedBytes() + empty_key_.AllocatedBytes();
  }

 private:
  Status DoInsert(OpKernelContext* ctx, const Tensor& key, const Tensor& value,
                  bool ignore_empty_and_deleted_key)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    const int64 num_elements = (key.dims() == 0) ? 1 : key.dim_size(0);
    const int64 value_size = value_shape_.num_elements();
    const int64 key_size = key_shape_.num_elements();
    const auto key_matrix = key.shaped<K, 2>({num_elements, key_size});
    auto value_matrix = value.shaped<V, 2>({num_elements, value_size});

    auto key_buckets_matrix =
        key_buckets_.AccessTensor(ctx)->template matrix<K>();
    auto value_buckets_matrix =
        value_buckets_.AccessTensor(ctx)->template matrix<V>();
    const auto empty_key_tensor =
        empty_key_.AccessTensor(ctx)->template shaped<K, 2>({1, key_size});
    const auto deleted_key_tensor =
        deleted_key_.AccessTensor(ctx)->template shaped<K, 2>({1, key_size});
    const int64 bit_mask = num_buckets_ - 1;
    for (int64 i = 0; i < num_elements; ++i) {
      const uint64 key_hash = HashKey(key_matrix, i);
      if (empty_key_hash_ == key_hash &&
          IsEqualKey(empty_key_tensor, 0, key_matrix, i)) {
        if (ignore_empty_and_deleted_key) {
          continue;
        }
        return errors::InvalidArgument(
            "Using the empty_key as a table key is not allowed");
      }
      if (deleted_key_hash_ == key_hash &&
          IsEqualKey(deleted_key_tensor, 0, key_matrix, i)) {
        if (ignore_empty_and_deleted_key) {
          continue;
        }
        return errors::InvalidArgument(
            "Using the deleted_key as a table key is not allowed");
      }
      int64 bucket_index = key_hash & bit_mask;
      int64 num_probes = 0;
      while (true) {
        if (IsEqualKey(key_buckets_matrix, bucket_index, key_matrix, i)) {
          for (int64 j = 0; j < value_size; ++j) {
            value_buckets_matrix(bucket_index, j) =
                SubtleMustCopyIfIntegral(value_matrix(i, j));
          }
          break;
        }
        if (IsEqualKey(key_buckets_matrix, bucket_index, empty_key_tensor, 0) ||
            IsEqualKey(key_buckets_matrix, bucket_index, deleted_key_tensor,
                       0)) {
          ++num_entries_;
          for (int64 j = 0; j < key_size; ++j) {
            key_buckets_matrix(bucket_index, j) =
                SubtleMustCopyIfIntegral(key_matrix(i, j));
          }
          for (int64 j = 0; j < value_size; ++j) {
            value_buckets_matrix(bucket_index, j) =
                SubtleMustCopyIfIntegral(value_matrix(i, j));
          }
          break;
        }
        ++num_probes;
        bucket_index =
            (bucket_index + num_probes) & bit_mask;  // quadratic probing
        if (num_probes >= num_buckets_) {
          return errors::Internal(
              "Internal error in MutableDenseHashTable insert");
        }
      }
    }
    return Status::OK();
  }

  Status DoRemove(OpKernelContext* ctx, const Tensor& key)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    const int64 num_elements = key.dim_size(0);
    const int64 key_size = key_shape_.num_elements();
    const auto key_matrix = key.shaped<K, 2>({num_elements, key_size});

    auto key_buckets_matrix =
        key_buckets_.AccessTensor(ctx)->template matrix<K>();
    const auto empty_key_tensor =
        empty_key_.AccessTensor(ctx)->template shaped<K, 2>({1, key_size});
    const auto deleted_key_tensor =
        deleted_key_.AccessTensor(ctx)->template shaped<K, 2>({1, key_size});
    const auto deleted_key_flat =
        deleted_key_.AccessTensor(ctx)->template flat<K>();
    const int64 bit_mask = num_buckets_ - 1;
    for (int64 i = 0; i < num_elements; ++i) {
      const uint64 key_hash = HashKey(key_matrix, i);
      if (empty_key_hash_ == key_hash &&
          IsEqualKey(empty_key_tensor, 0, key_matrix, i)) {
        return errors::InvalidArgument(
            "Using the empty_key as a table key is not allowed");
      }
      if (deleted_key_hash_ == key_hash &&
          IsEqualKey(deleted_key_tensor, 0, key_matrix, i)) {
        return errors::InvalidArgument(
            "Using the deleted_key as a table key is not allowed");
      }
      int64 bucket_index = key_hash & bit_mask;
      int64 num_probes = 0;
      while (true) {
        if (IsEqualKey(key_buckets_matrix, bucket_index, key_matrix, i)) {
          --num_entries_;
          for (int64 j = 0; j < key_size; ++j) {
            key_buckets_matrix(bucket_index, j) =
                SubtleMustCopyIfIntegral(deleted_key_flat(j));
          }
          break;
        }
        if (IsEqualKey(key_buckets_matrix, bucket_index, empty_key_tensor, 0)) {
          break;
        }
        ++num_probes;
        bucket_index =
            (bucket_index + num_probes) & bit_mask;  // quadratic probing
        if (num_probes >= num_buckets_) {
          return errors::Internal(
              "Internal error in MutableDenseHashTable remove");
        }
      }
    }
    return Status::OK();
  }

  Status AllocateBuckets(OpKernelContext* ctx, int64 new_num_buckets)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (new_num_buckets < 4 ||
        ((new_num_buckets & (new_num_buckets - 1)) != 0)) {
      return errors::InvalidArgument(
          "Number of buckets must be at least 4 and a power of 2, got: ",
          new_num_buckets);
    }
    num_buckets_ = new_num_buckets;
    num_entries_ = 0;

    const int64 key_size = key_shape_.num_elements();
    Tensor* key_buckets_tensor;
    TF_RETURN_IF_ERROR(ctx->allocate_persistent(
        key_dtype(), TensorShape({num_buckets_, key_size}), &key_buckets_,
        &key_buckets_tensor));
    auto key_buckets_matrix = key_buckets_tensor->matrix<K>();
    const auto empty_key_flat =
        empty_key_.AccessTensor(ctx)->template flat<K>();
    for (int64 i = 0; i < num_buckets_; ++i) {
      for (int64 j = 0; j < key_size; ++j) {
        key_buckets_matrix(i, j) = empty_key_flat(j);
      }
    }

    const int64 value_size = value_shape_.num_elements();
    Tensor* value_buckets_tensor;
    TF_RETURN_IF_ERROR(ctx->allocate_persistent(
        value_dtype(), TensorShape({num_buckets_, value_size}), &value_buckets_,
        &value_buckets_tensor));
    auto value_buckets_matrix = value_buckets_tensor->matrix<V>();
    for (int64 i = 0; i < num_buckets_; ++i) {
      for (int64 j = 0; j < value_size; ++j) {
        // Initialize values to the default value for the type to avoid
        // exposing uninitialized memory in ExportValues().
        value_buckets_matrix(i, j) = V();
      }
    }
    return Status::OK();
  }

  Status Rebucket(OpKernelContext* ctx, int64 num_new_buckets)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    Tensor old_key_buckets = *key_buckets_.AccessTensor(ctx);
    Tensor old_value_buckets = *value_buckets_.AccessTensor(ctx);
    TF_RETURN_IF_ERROR(AllocateBuckets(ctx, num_new_buckets));
    return DoInsert(ctx, old_key_buckets, old_value_buckets, true);
  }

  uint64 HashKey(typename TTypes<K>::ConstMatrix key, int64 index) const {
    if (key_shape_.num_elements() == 1) {
      return HashScalar(key(index, 0));
    }
    uint64 result = 0;
    for (int64 i = 0; i < key_shape_.num_elements(); ++i) {
      result = Hash64Combine(result, HashScalar(key(index, i)));
    }
    return result;
  }

  // Use a template to allow this function to be used both with Matrix and
  // ConstMatrix types.
  template <typename MT2>
  bool IsEqualKey(typename TTypes<K>::Matrix tensor1, int64 index1, MT2 tensor2,
                  int64 index2) const {
    for (int64 i = 0; i < key_shape_.num_elements(); ++i) {
      if (tensor1(index1, i) != tensor2(index2, i)) {
        return false;
      }
    }
    return true;
  }

  TensorShape key_shape_;
  TensorShape value_shape_;
  float max_load_factor_;
  mutable mutex mu_;
  int64 num_entries_ GUARDED_BY(mu_);
  int64 num_buckets_ GUARDED_BY(mu_);
  PersistentTensor key_buckets_ GUARDED_BY(mu_);
  PersistentTensor value_buckets_ GUARDED_BY(mu_);
  PersistentTensor empty_key_;
  uint64 empty_key_hash_;
  PersistentTensor deleted_key_;
  uint64 deleted_key_hash_;
=======
 private:
  std::unique_ptr<std::unordered_map<K, V>> table_;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
};

}  // namespace lookup

<<<<<<< HEAD
// Base class for kernels that take a LookupTable handle as the 0th input.
class LookupTableOpKernel : public OpKernel {
 public:
  explicit LookupTableOpKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx),
        expected_input_0_(ctx->input_type(0) == DT_RESOURCE ? DT_RESOURCE
                                                            : DT_STRING_REF) {}

 protected:
  Status GetTable(OpKernelContext* ctx, lookup::LookupInterface** table) {
    if (expected_input_0_ == DT_RESOURCE) {
      return GetResourceLookupTable("table_handle", ctx, table);
    } else {
      return GetReferenceLookupTable("table_handle", ctx, table);
    }
  }

  // Input 0 could be a STRING_REF or a RESOURCE
  const DataType expected_input_0_;
};

// Table lookup op. Perform the lookup operation on the given table.
class LookupTableFindOp : public LookupTableOpKernel {
 public:
  using LookupTableOpKernel::LookupTableOpKernel;

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype(),
=======
// Table lookup op. Perform the lookup operation on the given table.
class LookupTableFindOp : public OpKernel {
 public:
  explicit LookupTableFindOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    DataTypeVector expected_inputs = {DT_STRING_REF, table->key_dtype(),
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                                      table->value_dtype()};
    DataTypeVector expected_outputs = {table->value_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));

<<<<<<< HEAD
    const Tensor& key = ctx->input(1);
    const Tensor& default_value = ctx->input(2);
    OP_REQUIRES_OK(ctx, table->CheckFindArguments(key, default_value));

    TensorShape output_shape = key.shape();
    output_shape.RemoveLastDims(table->key_shape().dims());
    output_shape.AppendShape(table->value_shape());
    Tensor* out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("values", output_shape, &out));

    OP_REQUIRES_OK(ctx, table->Find(ctx, key, out, default_value));
=======
    const Tensor& input = ctx->input(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(input.shape()),
                errors::InvalidArgument("Input must be a vector, not ",
                                        input.shape().DebugString()));

    const Tensor& default_value = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(default_value.shape()),
                errors::InvalidArgument("Default value must be a scalar, not ",
                                        default_value.shape().DebugString()));

    Tensor* out;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("output_values", input.shape(), &out));

    OP_REQUIRES_OK(ctx, table->Find(input, out, default_value));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }
};

REGISTER_KERNEL_BUILDER(Name("LookupTableFind").Device(DEVICE_CPU),
                        LookupTableFindOp);
<<<<<<< HEAD
REGISTER_KERNEL_BUILDER(Name("LookupTableFindV2").Device(DEVICE_CPU),
                        LookupTableFindOp);

// Table insert op.
class LookupTableInsertOp : public LookupTableOpKernel {
 public:
  using LookupTableOpKernel::LookupTableOpKernel;

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype(),
                                      table->value_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor& keys = ctx->input(1);
    const Tensor& values = ctx->input(2);
    OP_REQUIRES_OK(ctx, table->CheckKeyAndValueTensorsForInsert(keys, values));

    int64 memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = table->MemoryUsed();
    }
    OP_REQUIRES_OK(ctx, table->Insert(ctx, keys, values));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(table->MemoryUsed() -
                                               memory_used_before);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("LookupTableInsert").Device(DEVICE_CPU),
                        LookupTableInsertOp);
REGISTER_KERNEL_BUILDER(Name("LookupTableInsertV2").Device(DEVICE_CPU),
                        LookupTableInsertOp);

// Table remove op.
class LookupTableRemoveOp : public LookupTableOpKernel {
 public:
  using LookupTableOpKernel::LookupTableOpKernel;

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor& key = ctx->input(1);
    OP_REQUIRES_OK(ctx, table->CheckKeyTensorForRemove(key));

    int64 memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = table->MemoryUsed();
    }
    OP_REQUIRES_OK(ctx, table->Remove(ctx, key));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(table->MemoryUsed() -
                                               memory_used_before);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("LookupTableRemoveV2").Device(DEVICE_CPU),
                        LookupTableRemoveOp);

// Op that returns the size of the given table.
class LookupTableSizeOp : public LookupTableOpKernel {
 public:
  using LookupTableOpKernel::LookupTableOpKernel;

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
=======

// Op that returns the size of the given table.
class LookupTableSizeOp : public OpKernel {
 public:
  explicit LookupTableSizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    core::ScopedUnref unref_me(table);

    Tensor* out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("size", TensorShape({}), &out));
    out->flat<int64>().setConstant(table->size());
  }
};

REGISTER_KERNEL_BUILDER(Name("LookupTableSize").Device(DEVICE_CPU),
                        LookupTableSizeOp);
<<<<<<< HEAD
REGISTER_KERNEL_BUILDER(Name("LookupTableSizeV2").Device(DEVICE_CPU),
                        LookupTableSizeOp);

// Op that outputs tensors of all keys and all values.
class LookupTableExportOp : public LookupTableOpKernel {
 public:
  using LookupTableOpKernel::LookupTableOpKernel;

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    OP_REQUIRES_OK(ctx, table->ExportValues(ctx));
  }
};

REGISTER_KERNEL_BUILDER(Name("LookupTableExport").Device(DEVICE_CPU),
                        LookupTableExportOp);
REGISTER_KERNEL_BUILDER(Name("LookupTableExportV2").Device(DEVICE_CPU),
                        LookupTableExportOp);

// Clear the table and insert data.
class LookupTableImportOp : public LookupTableOpKernel {
 public:
  using LookupTableOpKernel::LookupTableOpKernel;

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype(),
                                      table->value_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor& keys = ctx->input(1);
    const Tensor& values = ctx->input(2);
    OP_REQUIRES_OK(ctx, table->CheckKeyAndValueTensorsForImport(keys, values));

    int memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = table->MemoryUsed();
    }
    OP_REQUIRES_OK(ctx, table->ImportValues(ctx, keys, values));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(table->MemoryUsed() -
                                               memory_used_before);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("LookupTableImport").Device(DEVICE_CPU),
                        LookupTableImportOp);
REGISTER_KERNEL_BUILDER(Name("LookupTableImportV2").Device(DEVICE_CPU),
                        LookupTableImportOp);
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

// Register the HashTable op with the currently supported key and value types.
#define REGISTER_KERNEL(key_dtype, value_dtype)                           \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("HashTable")                                                   \
          .Device(DEVICE_CPU)                                             \
          .TypeConstraint<key_dtype>("key_dtype")                         \
          .TypeConstraint<value_dtype>("value_dtype"),                    \
      LookupTableOp<lookup::HashTable<key_dtype, value_dtype>, key_dtype, \
<<<<<<< HEAD
                    value_dtype>)                                         \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("HashTableV2")                                                 \
          .Device(DEVICE_CPU)                                             \
          .TypeConstraint<key_dtype>("key_dtype")                         \
          .TypeConstraint<value_dtype>("value_dtype"),                    \
      LookupTableOp<lookup::HashTable<key_dtype, value_dtype>, key_dtype, \
                    value_dtype>)

REGISTER_KERNEL(int32, double);
REGISTER_KERNEL(int32, float);
REGISTER_KERNEL(int32, int32);
REGISTER_KERNEL(int32, tstring);
REGISTER_KERNEL(int64, double);
REGISTER_KERNEL(int64, float);
REGISTER_KERNEL(int64, int32);
REGISTER_KERNEL(int64, int64);
REGISTER_KERNEL(int64, tstring);
REGISTER_KERNEL(tstring, bool);
REGISTER_KERNEL(tstring, double);
REGISTER_KERNEL(tstring, float);
REGISTER_KERNEL(tstring, int32);
REGISTER_KERNEL(tstring, int64);
REGISTER_KERNEL(tstring, tstring);

#undef REGISTER_KERNEL

// Register the MutableHashTable op.
#define REGISTER_KERNEL(key_dtype, value_dtype)                                \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("MutableHashTable")                                                 \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<key_dtype>("key_dtype")                              \
          .TypeConstraint<value_dtype>("value_dtype"),                         \
      LookupTableOp<lookup::MutableHashTableOfScalars<key_dtype, value_dtype>, \
                    key_dtype, value_dtype>)                                   \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("MutableHashTableV2")                                               \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<key_dtype>("key_dtype")                              \
          .TypeConstraint<value_dtype>("value_dtype"),                         \
      LookupTableOp<lookup::MutableHashTableOfScalars<key_dtype, value_dtype>, \
                    key_dtype, value_dtype>)

REGISTER_KERNEL(int32, double);
REGISTER_KERNEL(int32, float);
REGISTER_KERNEL(int32, int32);
REGISTER_KERNEL(int64, double);
REGISTER_KERNEL(int64, float);
REGISTER_KERNEL(int64, int32);
REGISTER_KERNEL(int64, int64);
REGISTER_KERNEL(int64, tstring);
REGISTER_KERNEL(int64, Variant);
REGISTER_KERNEL(tstring, bool);
REGISTER_KERNEL(tstring, double);
REGISTER_KERNEL(tstring, float);
REGISTER_KERNEL(tstring, int32);
REGISTER_KERNEL(tstring, int64);

#undef REGISTER_KERNEL

// Register the MutableHashTableOfTensors op.
#define REGISTER_KERNEL(key_dtype, value_dtype)                                \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("MutableHashTableOfTensors")                                        \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<key_dtype>("key_dtype")                              \
          .TypeConstraint<value_dtype>("value_dtype"),                         \
      LookupTableOp<lookup::MutableHashTableOfTensors<key_dtype, value_dtype>, \
                    key_dtype, value_dtype>)                                   \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("MutableHashTableOfTensorsV2")                                      \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<key_dtype>("key_dtype")                              \
          .TypeConstraint<value_dtype>("value_dtype"),                         \
      LookupTableOp<lookup::MutableHashTableOfTensors<key_dtype, value_dtype>, \
                    key_dtype, value_dtype>)

REGISTER_KERNEL(int32, double);
REGISTER_KERNEL(int32, float);
REGISTER_KERNEL(int32, int32);
REGISTER_KERNEL(int64, double);
REGISTER_KERNEL(int64, float);
REGISTER_KERNEL(int64, int32);
REGISTER_KERNEL(int64, int64);
REGISTER_KERNEL(int64, tstring);
REGISTER_KERNEL(tstring, bool);
REGISTER_KERNEL(tstring, double);
REGISTER_KERNEL(tstring, float);
REGISTER_KERNEL(tstring, int32);
REGISTER_KERNEL(tstring, int64);

#undef REGISTER_KERNEL

// Register the MutableDenseHashTable op.
#define REGISTER_KERNEL(key_dtype, value_dtype)                            \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("MutableDenseHashTable")                                        \
          .Device(DEVICE_CPU)                                              \
          .TypeConstraint<key_dtype>("key_dtype")                          \
          .TypeConstraint<value_dtype>("value_dtype"),                     \
      LookupTableOp<lookup::MutableDenseHashTable<key_dtype, value_dtype>, \
                    key_dtype, value_dtype>)                               \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("MutableDenseHashTableV2")                                      \
          .Device(DEVICE_CPU)                                              \
          .TypeConstraint<key_dtype>("key_dtype")                          \
          .TypeConstraint<value_dtype>("value_dtype"),                     \
      LookupTableOp<lookup::MutableDenseHashTable<key_dtype, value_dtype>, \
                    key_dtype, value_dtype>)

REGISTER_KERNEL(int32, double);
REGISTER_KERNEL(int32, float);
REGISTER_KERNEL(int32, int32);
REGISTER_KERNEL(int64, bool);
REGISTER_KERNEL(int64, double);
REGISTER_KERNEL(int64, float);
REGISTER_KERNEL(int64, int32);
REGISTER_KERNEL(int64, int64);
REGISTER_KERNEL(int64, Variant);
REGISTER_KERNEL(tstring, bool);
REGISTER_KERNEL(tstring, double);
REGISTER_KERNEL(tstring, float);
REGISTER_KERNEL(tstring, int32);
REGISTER_KERNEL(tstring, int64);
REGISTER_KERNEL(tstring, ResourceHandle);
=======
                    value_dtype>)

REGISTER_KERNEL(string, int64);
REGISTER_KERNEL(int64, string);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#undef REGISTER_KERNEL

}  // namespace tensorflow
