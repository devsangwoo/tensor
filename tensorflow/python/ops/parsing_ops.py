<<<<<<< HEAD
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Parsing Ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_config
# go/tf-wildcard-import
# pylint: disable=wildcard-import,undefined-variable
from tensorflow.python.ops.gen_parsing_ops import *
# pylint: enable=wildcard-import,undefined-variable
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


ops.NotDifferentiable("DecodeRaw")
ops.NotDifferentiable("DecodePaddedRaw")
ops.NotDifferentiable("ParseTensor")
ops.NotDifferentiable("SerializeTensor")
ops.NotDifferentiable("StringToNumber")


VarLenFeature = parsing_config.VarLenFeature
RaggedFeature = parsing_config.RaggedFeature
SparseFeature = parsing_config.SparseFeature
FixedLenFeature = parsing_config.FixedLenFeature
FixedLenSequenceFeature = parsing_config.FixedLenSequenceFeature
# pylint: disable=protected-access
_ParseOpParams = parsing_config._ParseOpParams
_construct_tensors_for_composite_features = (
    parsing_config._construct_tensors_for_composite_features)
# pylint: enable=protected-access


# TODO(b/122887740) Switch files that use this private symbol to use new name.
_construct_sparse_tensors_for_sparse_features = \
    _construct_tensors_for_composite_features


def _prepend_none_dimension(features):
  """Returns a copy of features with adjusted FixedLenSequenceFeature shapes."""
  if features:
    modified_features = dict(features)  # Create a copy to modify
    for key, feature in features.items():
      if isinstance(feature, FixedLenSequenceFeature):
        if not feature.allow_missing:
          raise ValueError("Unsupported: FixedLenSequenceFeature requires "
                           "allow_missing to be True.")
        modified_features[key] = FixedLenSequenceFeature(
            [None] + list(feature.shape),
            feature.dtype,
            feature.allow_missing,
            feature.default_value)
    return modified_features
  else:
    return features


@tf_export("io.parse_example", v1=[])
def parse_example_v2(serialized, features, example_names=None, name=None):
  # pylint: disable=line-too-long
  """Parses `Example` protos into a `dict` of tensors.

  Parses a number of serialized [`Example`](https://www.tensorflow.org/code/tensorflow/core/example/example.proto)
  protos given in `serialized`. We refer to `serialized` as a batch with
  `batch_size` many entries of individual `Example` protos.

  `example_names` may contain descriptive names for the corresponding serialized
  protos. These may be useful for debugging purposes, but they have no effect on
  the output. If not `None`, `example_names` must be the same length as
  `serialized`.

  This op parses serialized examples into a dictionary mapping keys to `Tensor`
  `SparseTensor`, and `RaggedTensor` objects. `features` is a dict from keys to
  `VarLenFeature`, `SparseFeature`, `RaggedFeature`, and `FixedLenFeature`
  objects. Each `VarLenFeature` and `SparseFeature` is mapped to a
  `SparseTensor`; each `FixedLenFeature` is mapped to a `Tensor`; and each
  `RaggedFeature` is mapped to a `RaggedTensor`.

  Each `VarLenFeature` maps to a `SparseTensor` of the specified type
  representing a ragged matrix. Its indices are `[batch, index]` where `batch`
  identifies the example in `serialized`, and `index` is the value's index in
  the list of values associated with that feature and example.

  Each `SparseFeature` maps to a `SparseTensor` of the specified type
  representing a Tensor of `dense_shape` `[batch_size] + SparseFeature.size`.
  Its `values` come from the feature in the examples with key `value_key`.
  A `values[i]` comes from a position `k` in the feature of an example at batch
  entry `batch`. This positional information is recorded in `indices[i]` as
  `[batch, index_0, index_1, ...]` where `index_j` is the `k-th` value of
  the feature in the example at with key `SparseFeature.index_key[j]`.
  In other words, we split the indices (except the first index indicating the
  batch entry) of a `SparseTensor` by dimension into different features of the
  `Example`. Due to its complexity a `VarLenFeature` should be preferred over a
  `SparseFeature` whenever possible.

  Each `FixedLenFeature` `df` maps to a `Tensor` of the specified type (or
  `tf.float32` if not specified) and shape `(serialized.size(),) + df.shape`.

  `FixedLenFeature` entries with a `default_value` are optional. With no default
  value, we will fail if that `Feature` is missing from any example in
  `serialized`.

  Each `FixedLenSequenceFeature` `df` maps to a `Tensor` of the specified type
  (or `tf.float32` if not specified) and shape
  `(serialized.size(), None) + df.shape`.
  All examples in `serialized` will be padded with `default_value` along the
  second dimension.

  Each `RaggedFeature` maps to a `RaggedTensor` of the specified type.  It
  is formed by stacking the `RaggedTensor` for each example, where the
  `RaggedTensor` for each individual example is constructed using the tensors
  specified by `RaggedTensor.values_key` and `RaggedTensor.partition`.  See
  the `tf.io.RaggedFeature` documentation for details and examples.

  Examples:

  For example, if one expects a `tf.float32` `VarLenFeature` `ft` and three
  serialized `Example`s are provided:

  ```
  serialized = [
    features
      { feature { key: "ft" value { float_list { value: [1.0, 2.0] } } } },
    features
      { feature []},
    features
      { feature { key: "ft" value { float_list { value: [3.0] } } }
  ]
  ```

  then the output will look like:

  ```python
  {"ft": SparseTensor(indices=[[0, 0], [0, 1], [2, 0]],
                      values=[1.0, 2.0, 3.0],
                      dense_shape=(3, 2)) }
  ```

  If instead a `FixedLenSequenceFeature` with `default_value = -1.0` and
  `shape=[]` is used then the output will look like:

  ```python
  {"ft": [[1.0, 2.0], [3.0, -1.0]]}
  ```

  Given two `Example` input protos in `serialized`:

  ```
  [
    features {
      feature { key: "kw" value { bytes_list { value: [ "knit", "big" ] } } }
      feature { key: "gps" value { float_list { value: [] } } }
    },
    features {
      feature { key: "kw" value { bytes_list { value: [ "emmy" ] } } }
      feature { key: "dank" value { int64_list { value: [ 42 ] } } }
      feature { key: "gps" value { } }
    }
  ]
  ```

  And arguments

  ```
  example_names: ["input0", "input1"],
  features: {
      "kw": VarLenFeature(tf.string),
      "dank": VarLenFeature(tf.int64),
      "gps": VarLenFeature(tf.float32),
  }
  ```

  Then the output is a dictionary:

  ```python
=======
"""Parsing Ops."""

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import common_shapes
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
# pylint: disable=wildcard-import,undefined-variable
from tensorflow.python.ops.gen_parsing_ops import *


ops.NoGradient("DecodeRaw")
ops.NoGradient("StringToNumber")


# pylint: disable=protected-access
def parse_example(serialized,
                  names=None,
                  sparse_keys=None,
                  sparse_types=None,
                  dense_keys=None,
                  dense_types=None,
                  dense_defaults=None,
                  dense_shapes=None,
                  name="ParseExample"):
  """Parse Example protos.

  Args:
    serialized: string vector, a batch of binary serialized Example protos.
    names: A string vector, the names of the serialized protos.
      "names" may contain, e.g., table key (descriptive) names for the
      corresponding serialized protos.  These are purely useful for debugging
      purposes, and the presence of values here has no effect on the output.
      "names" may be an empty vector, if no names are available.
      If non-empty, this vector must be the same length as "serialized".
    sparse_keys: A string list of keys in the Examples' features.
      These keys are associated with sparse values.
    sparse_types: A list of DTypes.
      This list's length must match that of sparse_keys.  Currently
      parse_example supports tf.float32 (FloatList), tf.int64 (Int64List),
      and tf.string (BytesList).
    dense_keys: A string list of keys in the Examples' features.
      These keys are associated with dense values.
    dense_types: A list of DTypes.
      This list's length must match that of dense_keys.  Currently
      parse_example supports tf.float32 (FloatList), tf.int64 (Int64List),
      and tf.string (BytesList).
    dense_defaults: A dict of {key:Tensor} (some may be missing).
      The keys of the dict must match the dense_keys of the feature.
      If a key is not present in this dictionary, the corresponding dense
      Feature is required in all elements of serialized.
    dense_shapes: A list of tuples.
      Entries provide the shape of data in each dense Feature in features.
      The length of dense_shapes must be the same as the length of dense_keys.
      The number of elements in the Feature corresponding to dense_key[j]
      must always have np.prod(dense_shapes[j]) entries.
      If dense_shapes[j] == (D0, D1, ..., DN) then the the shape of output
      Tensor dense_values[j] will be (|serialized|, D0, D1, ..., DN):
      The dense outputs are just the inputs row-stacked by batch.
    name: (Optional) Name of Op in the graph.

  Returns:
    A dictionary mapping keys to Tensors and SparseTensors.

    The key dense_keys[j] is mapped to a tensor of type dense_types[j] and
    of shape (serialized.size(),) + dense_shapes[j] (i.e., the dense outputs are
    inputs, reshaped in row-major format and then row-stacked by batch).

    The key sparse_keys[j] is mapped to a SparseTensor of type sparse_types[j].
    The SparseTensor represents a ragged matrix.  Its indices are [batch, index]
    where "batch" is is the batch entry the value is from, and "index" is the
    value's index in the list of values associated with that feature
    and example.  For example, if one expects a tf.float32 sparse feature "ft"
    and three serialized examples are provided:

    serialized = [
      features:
        { feature: [ key: { "ft" value: float_list: { value: [1.0, 2.0] } } ] },
      features:
        { feature: [] },
      features:
        { feature: [ key: { "ft" value: float_list: { value: [3.0] } } ] }
    ]

    then the output will look like:

      {"ft": SparseTensor(indices=[[0, 0], [0, 1], [2, 0]],
                          values=[1.0, 2.0, 3.0],
                          shape=(3, 2)) }

  Raises:
    ValueError: If sparse and dense keys intersect, or input lengths do not
      match up for sparse_* (similarly for dense_*).
    TypeError: If an input is malformed.

  Example input, format, and output: Just Sparse Inputs
  ================================================

  Given two brain.Example input protos:

  serialized:  // serialized versions of the protos below
    [features: {
      feature: { key: "kw" value: { bytes_list: { value: [ "knit", "big" ] } } }
      feature: { key: "gps" value: { float_list: { value: [] } } }
     },
     features: {
      feature: { key: "kw" value: { bytes_list: { value: [ "emmy" ] } } }
      feature: { key: "dank" value: { int64_list: { value: [ 42 ] } } }
      feature: { key: "gps" value: { } }
    }]
  names: ["input0", "input1"],
  sparse_keys: ["kw", "dank", "gps"]
  sparse_types: [DT_STRING, DT_INT64, DT_FLOAT]

  Then the expected output is a dictionary:
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  {
    "kw": SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0]],
        values=["knit", "big", "emmy"]
<<<<<<< HEAD
        dense_shape=[2, 2]),
    "dank": SparseTensor(
        indices=[[1, 0]],
        values=[42],
        dense_shape=[2, 1]),
    "gps": SparseTensor(
        indices=[],
        values=[],
        dense_shape=[2, 0]),
  }
  ```

  For dense results in two serialized `Example`s:

  ```
  [
    features {
      feature { key: "age" value { int64_list { value: [ 0 ] } } }
      feature { key: "gender" value { bytes_list { value: [ "f" ] } } }
     },
     features {
      feature { key: "age" value { int64_list { value: [] } } }
      feature { key: "gender" value { bytes_list { value: [ "f" ] } } }
    }
  ]
  ```

  We can use arguments:

  ```
  example_names: ["input0", "input1"],
  features: {
      "age": FixedLenFeature([], dtype=tf.int64, default_value=-1),
      "gender": FixedLenFeature([], dtype=tf.string),
  }
  ```

  And the expected output is:

  ```python
=======
        shape=[2, 2]),
    "dank": SparseTensor(
        indices=[[1, 0]],
        values=[42],
        shape=[2, 1]),
    "gps": SparseTensor(
        indices=[],
        values=[],
        shape=[2, 0]),
  }


  Example input, format, and output: Dense Inputs (without defaults)
  ==================================================================

  Given two brain.Example input protos:

  serialized:  // serialized versions of the protos below
    [features: {
      feature: { key: "age" value: { int64_list: { value: [ 0 ] } } }
      feature: { key: "gender" value: { bytes_list: { value: [ "f" ] } } }
     },
     features: {
      feature: { key: "age" value: { int64_list: { value: [] } } }
      feature: { key: "gender" value: { bytes_list: { value: [ "f" ] } } }
    }]
  names: ["input0", "input1"],
  dense_keys: np.array(["age", "gender"])
  dense_types: [tf.int64, tf.string]
  dense_defaults: {
    "age": -1  # defaults to -1 if missing
               # "gender" has no specified default so it's required
  }
  dense_shapes: [(1,), (1,)]  # age, gender, label, weight

  Then the expected output is a dictionary:
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  {
    "age": [[0], [-1]],
    "gender": [["f"], ["f"]],
  }
<<<<<<< HEAD
  ```

  An alternative to `VarLenFeature` to obtain a `SparseTensor` is
  `SparseFeature`. For example, given two `Example` input protos in
  `serialized`:

  ```
  [
    features {
      feature { key: "val" value { float_list { value: [ 0.5, -1.0 ] } } }
      feature { key: "ix" value { int64_list { value: [ 3, 20 ] } } }
    },
    features {
      feature { key: "val" value { float_list { value: [ 0.0 ] } } }
      feature { key: "ix" value { int64_list { value: [ 42 ] } } }
    }
  ]
  ```

  And arguments

  ```
  example_names: ["input0", "input1"],
  features: {
      "sparse": SparseFeature(
          index_key="ix", value_key="val", dtype=tf.float32, size=100),
  }
  ```

  Then the output is a dictionary:

  ```python
  {
    "sparse": SparseTensor(
        indices=[[0, 3], [0, 20], [1, 42]],
        values=[0.5, -1.0, 0.0]
        dense_shape=[2, 100]),
  }
  ```

  See the `tf.io.RaggedFeature` documentation for examples showing how
  `RaggedFeature` can be used to obtain `RaggedTensor`s.

  Args:
    serialized: A vector (1-D Tensor) of strings, a batch of binary
      serialized `Example` protos.
    features: A `dict` mapping feature keys to `FixedLenFeature`,
      `VarLenFeature`, `SparseFeature`, and `RaggedFeature` values.
    example_names: A vector (1-D Tensor) of strings (optional), the names of
      the serialized protos in the batch.
    name: A name for this operation (optional).

  Returns:
    A `dict` mapping feature keys to `Tensor`, `SparseTensor`, and
    `RaggedTensor` values.

  Raises:
    ValueError: if any feature is invalid.
  """
  if not features:
    raise ValueError("Missing: features was %s." % features)
  features = _prepend_none_dimension(features)
  params = _ParseOpParams.from_features(features, [
      VarLenFeature, SparseFeature, FixedLenFeature, FixedLenSequenceFeature,
      RaggedFeature
  ])

  outputs = _parse_example_raw(serialized, example_names, params, name=name)
  return _construct_tensors_for_composite_features(features, outputs)


@tf_export(v1=["io.parse_example", "parse_example"])
def parse_example(serialized, features, name=None, example_names=None):
  return parse_example_v2(serialized, features, example_names, name)


parse_example.__doc__ = parse_example_v2.__doc__


def _parse_example_raw(serialized, names, params, name):
  """Parses `Example` protos.

  Args:
    serialized: A vector (1-D Tensor) of strings, a batch of binary
      serialized `Example` protos.
    names: A vector (1-D Tensor) of strings (optional), the names of
      the serialized protos.
    params: A `ParseOpParams` containing the parameters for the parse op.
    name: A name for this operation (optional).

  Returns:
    A `dict` mapping keys to `Tensor`s and `SparseTensor`s and `RaggedTensor`s.

  """
  if params.num_features == 0:
    raise ValueError("Must provide at least one feature key")
  with ops.name_scope(name, "ParseExample", [serialized, names]):
    names = [] if names is None else names
    serialized = ops.convert_to_tensor(serialized, name="serialized")
    if params.ragged_keys and serialized.shape.ndims is None:
      raise ValueError("serialized must have statically-known rank to "
                       "parse ragged features.")
    outputs = gen_parsing_ops.parse_example_v2(
        serialized=serialized,
        names=names,
        sparse_keys=params.sparse_keys,
        dense_keys=params.dense_keys,
        ragged_keys=params.ragged_keys,
        dense_defaults=params.dense_defaults_vec,
        num_sparse=len(params.sparse_keys),
        sparse_types=params.sparse_types,
        ragged_value_types=params.ragged_value_types,
        ragged_split_types=params.ragged_split_types,
        dense_shapes=params.dense_shapes_as_proto,
        name=name)
    (sparse_indices, sparse_values, sparse_shapes, dense_values,
     ragged_values, ragged_row_splits) = outputs
    # pylint: disable=protected-access
    ragged_tensors = parsing_config._build_ragged_tensors(
        serialized.shape, ragged_values, ragged_row_splits)

    sparse_tensors = [
        sparse_tensor.SparseTensor(ix, val, shape) for (ix, val, shape)
        in zip(sparse_indices, sparse_values, sparse_shapes)]

    return dict(
        zip(params.sparse_keys + params.dense_keys + params.ragged_keys,
            sparse_tensors + dense_values + ragged_tensors))


@tf_export(v1=["io.parse_single_example", "parse_single_example"])
def parse_single_example(serialized, features, name=None, example_names=None):
  """Parses a single `Example` proto.

  Similar to `parse_example`, except:

  For dense tensors, the returned `Tensor` is identical to the output of
  `parse_example`, except there is no batch dimension, the output shape is the
  same as the shape given in `dense_shape`.

  For `SparseTensor`s, the first (batch) column of the indices matrix is removed
  (the indices matrix is a column vector), the values vector is unchanged, and
  the first (`batch_size`) entry of the shape vector is removed (it is now a
  single element vector).

  One might see performance advantages by batching `Example` protos with
  `parse_example` instead of using this function directly.

  Args:
    serialized: A scalar string Tensor, a single serialized Example.
    features: A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` values.
    name: A name for this operation (optional).
    example_names: (Optional) A scalar string Tensor, the associated name.

  Returns:
    A `dict` mapping feature keys to `Tensor` and `SparseTensor` values.

  Raises:
    ValueError: if any feature is invalid.
  """
  return parse_single_example_v2(serialized, features, example_names, name)


@tf_export("io.parse_single_example", v1=[])
def parse_single_example_v2(
    serialized, features, example_names=None, name=None
    ):
  """Parses a single `Example` proto.

  Similar to `parse_example`, except:

  For dense tensors, the returned `Tensor` is identical to the output of
  `parse_example`, except there is no batch dimension, the output shape is the
  same as the shape given in `dense_shape`.

  For `SparseTensor`s, the first (batch) column of the indices matrix is removed
  (the indices matrix is a column vector), the values vector is unchanged, and
  the first (`batch_size`) entry of the shape vector is removed (it is now a
  single element vector).

  One might see performance advantages by batching `Example` protos with
  `parse_example` instead of using this function directly.

  Args:
    serialized: A scalar string Tensor, a single serialized Example.
    features: A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` values.
    example_names: (Optional) A scalar string Tensor, the associated name.
    name: A name for this operation (optional).

  Returns:
    A `dict` mapping feature keys to `Tensor` and `SparseTensor` values.

  Raises:
    ValueError: if any feature is invalid.
  """
  if not features:
    raise ValueError("Missing features.")
  with ops.name_scope(name, "ParseSingleExample", [serialized, example_names]):
    serialized = ops.convert_to_tensor(serialized, name="serialized")
    serialized = _assert_scalar(serialized, "serialized")
    return parse_example_v2(serialized, features, example_names, name)


@tf_export("io.parse_sequence_example")
def parse_sequence_example(serialized,
                           context_features=None,
                           sequence_features=None,
                           example_names=None,
                           name=None):
  # pylint: disable=line-too-long
  """Parses a batch of `SequenceExample` protos.

  Parses a vector of serialized
  [`SequenceExample`](https://www.tensorflow.org/code/tensorflow/core/example/example.proto)
  protos given in `serialized`.

  This op parses serialized sequence examples into a tuple of dictionaries,
  each mapping keys to `Tensor` and `SparseTensor` objects.
  The first dictionary contains mappings for keys appearing in
  `context_features`, and the second dictionary contains mappings for keys
  appearing in `sequence_features`.

  At least one of `context_features` and `sequence_features` must be provided
  and non-empty.

  The `context_features` keys are associated with a `SequenceExample` as a
  whole, independent of time / frame.  In contrast, the `sequence_features` keys
  provide a way to access variable-length data within the `FeatureList` section
  of the `SequenceExample` proto.  While the shapes of `context_features` values
  are fixed with respect to frame, the frame dimension (the first dimension)
  of `sequence_features` values may vary between `SequenceExample` protos,
  and even between `feature_list` keys within the same `SequenceExample`.

  `context_features` contains `VarLenFeature`, `RaggedFeature`, and
  `FixedLenFeature`  objects. Each `VarLenFeature` is mapped to a
  `SparseTensor`; each `RaggedFeature` is  mapped to a `RaggedTensor`; and each
  `FixedLenFeature` is mapped to a `Tensor`, of the specified type, shape, and
  default value.

  `sequence_features` contains `VarLenFeature`, `RaggedFeature`, and
  `FixedLenSequenceFeature` objects. Each `VarLenFeature` is mapped to a
  `SparseTensor`; each `RaggedFeature` is mapped to a `RaggedTensor; and
  each `FixedLenSequenceFeature` is mapped to a `Tensor`, each of the specified
  type. The shape will be `(B,T,) + df.dense_shape` for
  `FixedLenSequenceFeature` `df`, where `B` is the batch size, and `T` is the
  length of the associated `FeatureList` in the `SequenceExample`. For instance,
  `FixedLenSequenceFeature([])` yields a scalar 2-D `Tensor` of static shape
  `[None, None]` and dynamic shape `[B, T]`, while
  `FixedLenSequenceFeature([k])` (for `int k >= 1`) yields a 3-D matrix `Tensor`
  of static shape `[None, None, k]` and dynamic shape `[B, T, k]`.

  Like the input, the resulting output tensors have a batch dimension. This
  means that the original per-example shapes of `VarLenFeature`s and
  `FixedLenSequenceFeature`s can be lost. To handle that situation, this op also
  provides dicts of shape tensors as part of the output. There is one dict for
  the context features, and one for the feature_list features. Context features
  of type `FixedLenFeature`s will not be present, since their shapes are already
  known by the caller. In situations where the input 'FixedLenFeature`s are of
  different lengths across examples, the shorter examples will be padded with
  default datatype values: 0 for numeric types, and the empty string for string
  types.

  Each `SparseTensor` corresponding to `sequence_features` represents a ragged
  vector.  Its indices are `[time, index]`, where `time` is the `FeatureList`
  entry and `index` is the value's index in the list of values associated with
  that time.

  `FixedLenFeature` entries with a `default_value` and `FixedLenSequenceFeature`
  entries with `allow_missing=True` are optional; otherwise, we will fail if
  that `Feature` or `FeatureList` is missing from any example in `serialized`.

  `example_name` may contain a descriptive name for the corresponding serialized
  proto. This may be useful for debugging purposes, but it has no effect on the
  output. If not `None`, `example_name` must be a scalar.

  Args:
    serialized: A vector (1-D Tensor) of type string containing binary
      serialized `SequenceExample` protos.
    context_features: A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` or `RaggedFeature` values. These features are associated
      with a `SequenceExample` as a whole.
    sequence_features: A `dict` mapping feature keys to
      `FixedLenSequenceFeature` or `VarLenFeature` or `RaggedFeature` values.
      These features are associated with data within the `FeatureList` section
      of the `SequenceExample` proto.
    example_names: A vector (1-D Tensor) of strings (optional), the name of the
      serialized protos.
    name: A name for this operation (optional).

  Returns:
    A tuple of three `dict`s, each mapping keys to `Tensor`s,
    `SparseTensor`s, and `RaggedTensor`. The first dict contains the context
    key/values, the second dict contains the feature_list key/values, and the
    final dict contains the lengths of any dense feature_list features.

  Raises:
    ValueError: if any feature is invalid.
  """
  if not (context_features or sequence_features):
    raise ValueError("Missing features.")
  context_params = _ParseOpParams.from_features(
      context_features, [VarLenFeature, FixedLenFeature, RaggedFeature])
  feature_list_params = _ParseOpParams.from_features(
      sequence_features,
      [VarLenFeature, FixedLenSequenceFeature, RaggedFeature])

  with ops.name_scope(name, "ParseSequenceExample",
                      [serialized, example_names]):
    outputs = _parse_sequence_example_raw(serialized, example_names,
                                          context_params, feature_list_params,
                                          name)
    context_output, feature_list_output, feature_list_lengths = outputs

    if context_params.ragged_keys:
      context_output = _construct_tensors_for_composite_features(
          context_features, context_output)
    if feature_list_params.ragged_keys:
      feature_list_output = _construct_tensors_for_composite_features(
          sequence_features, feature_list_output)

    return context_output, feature_list_output, feature_list_lengths


def _parse_sequence_example_raw(serialized,
                                debug_name,
                                context,
                                feature_list,
                                name=None):
  """Parses a vector of `SequenceExample` protos.

  Args:
    serialized: A vector (1-D Tensor) of type string, containing binary
      serialized `SequenceExample` protos.
    debug_name: A vector (1-D Tensor) of strings (optional), the names of the
      serialized protos.
    context: A `ParseOpParams` containing the parameters for the parse
      op for the context features.
    feature_list: A `ParseOpParams` containing the parameters for the
      parse op for the feature_list features.
    name: A name for this operation (optional).

  Returns:
    A tuple of three `dict`s, each mapping keys to `Tensor`s, `SparseTensor`s,
    and `RaggedTensor`s. The first dict contains the context key/values, the
    second dict contains the feature_list key/values, and the final dict
    contains the lengths of any dense feature_list features.

  Raises:
    TypeError: if feature_list.dense_defaults is not either None or a dict.
  """
  if context.num_features + feature_list.num_features == 0:
    raise ValueError("Must provide at least one feature key")
  with ops.name_scope(name, "ParseSequenceExample", [serialized]):
    debug_name = [] if debug_name is None else debug_name

    # Internal
    feature_list_dense_missing_assumed_empty = []
    for k, v in feature_list.dense_defaults.items():
      if v is not None:
        raise ValueError("Value feature_list.dense_defaults[%s] must be None" %
                         k)
      feature_list_dense_missing_assumed_empty.append(k)

    has_ragged = context.ragged_keys or feature_list.ragged_keys
    serialized = ops.convert_to_tensor(serialized, name="serialized")
    if has_ragged and serialized.shape.ndims is None:
      raise ValueError("serialized must have statically-known rank to "
                       "parse ragged features.")
    feature_list_dense_missing_assumed_empty_vector = [
        key in feature_list_dense_missing_assumed_empty
        for key in feature_list.dense_keys
    ]
    outputs = gen_parsing_ops.parse_sequence_example_v2(
        # Inputs
        serialized=serialized,
        debug_name=debug_name,
        context_sparse_keys=context.sparse_keys,
        context_dense_keys=context.dense_keys,
        context_ragged_keys=context.ragged_keys,
        feature_list_sparse_keys=feature_list.sparse_keys,
        feature_list_dense_keys=feature_list.dense_keys,
        feature_list_ragged_keys=feature_list.ragged_keys,
        feature_list_dense_missing_assumed_empty=(
            feature_list_dense_missing_assumed_empty_vector),
        context_dense_defaults=context.dense_defaults_vec,
        # Attrs
        Ncontext_sparse=len(context.sparse_keys),
        Nfeature_list_sparse=len(feature_list.sparse_keys),
        Nfeature_list_dense=len(feature_list.dense_keys),
        context_sparse_types=context.sparse_types,
        context_ragged_value_types=context.ragged_value_types,
        context_ragged_split_types=context.ragged_split_types,
        feature_list_dense_types=feature_list.dense_types,
        feature_list_sparse_types=feature_list.sparse_types,
        feature_list_ragged_value_types=feature_list.ragged_value_types,
        feature_list_ragged_split_types=feature_list.ragged_split_types,
        context_dense_shapes=context.dense_shapes_as_proto,
        feature_list_dense_shapes=feature_list.dense_shapes,
        name=name)
    (context_sparse_indices, context_sparse_values, context_sparse_shapes,
     context_dense_values, context_ragged_values, context_ragged_row_splits,
     feature_list_sparse_indices, feature_list_sparse_values,
     feature_list_sparse_shapes, feature_list_dense_values,
     feature_list_dense_lengths, feature_list_ragged_values,
     feature_list_ragged_outer_splits,
     feature_list_ragged_inner_splits) = outputs
    # pylint: disable=protected-access
    context_ragged_tensors = parsing_config._build_ragged_tensors(
        serialized.shape, context_ragged_values, context_ragged_row_splits)
    feature_list_ragged_tensors = parsing_config._build_ragged_tensors(
        serialized.shape, feature_list_ragged_values,
        feature_list_ragged_outer_splits, feature_list_ragged_inner_splits)

    # pylint: disable=g-complex-comprehension
    context_sparse_tensors = [
        sparse_tensor.SparseTensor(ix, val, shape)
        for (ix, val,
             shape) in zip(context_sparse_indices, context_sparse_values,
                           context_sparse_shapes)
    ]

    feature_list_sparse_tensors = [
        sparse_tensor.SparseTensor(ix, val, shape)
        for (ix, val, shape
            ) in zip(feature_list_sparse_indices, feature_list_sparse_values,
                     feature_list_sparse_shapes)
    ]
    # pylint: enable=g-complex-comprehension

    context_output = dict(
        zip(
            context.sparse_keys + context.dense_keys + context.ragged_keys,
            context_sparse_tensors + context_dense_values +
            context_ragged_tensors))
    feature_list_output = dict(
        zip(
            feature_list.sparse_keys + feature_list.dense_keys +
            feature_list.ragged_keys, feature_list_sparse_tensors +
            feature_list_dense_values + feature_list_ragged_tensors))
    feature_list_lengths = dict(
        zip(feature_list.dense_keys, feature_list_dense_lengths))

    return (context_output, feature_list_output, feature_list_lengths)


@tf_export("io.parse_single_sequence_example",
           v1=["io.parse_single_sequence_example",
               "parse_single_sequence_example"])
def parse_single_sequence_example(
    serialized, context_features=None, sequence_features=None,
    example_name=None, name=None):
  # pylint: disable=line-too-long
  """Parses a single `SequenceExample` proto.

  Parses a single serialized [`SequenceExample`](https://www.tensorflow.org/code/tensorflow/core/example/example.proto)
  proto given in `serialized`.

  This op parses a serialized sequence example into a tuple of dictionaries,
  each mapping keys to `Tensor` and `SparseTensor` objects.
  The first dictionary contains mappings for keys appearing in
  `context_features`, and the second dictionary contains mappings for keys
  appearing in `sequence_features`.

  At least one of `context_features` and `sequence_features` must be provided
  and non-empty.

  The `context_features` keys are associated with a `SequenceExample` as a
  whole, independent of time / frame.  In contrast, the `sequence_features` keys
  provide a way to access variable-length data within the `FeatureList` section
  of the `SequenceExample` proto.  While the shapes of `context_features` values
  are fixed with respect to frame, the frame dimension (the first dimension)
  of `sequence_features` values may vary between `SequenceExample` protos,
  and even between `feature_list` keys within the same `SequenceExample`.

  `context_features` contains `VarLenFeature`, `RaggedFeature`, and
  `FixedLenFeature` objects. Each `VarLenFeature` is mapped to a `SparseTensor`;
  each `RaggedFeature` is mapped to a `RaggedTensor`; and each `FixedLenFeature`
  is mapped to a `Tensor`, of the specified type, shape, and default value.

  `sequence_features` contains `VarLenFeature`, `RaggedFeature`, and
  `FixedLenSequenceFeature` objects. Each `VarLenFeature` is mapped to a
  `SparseTensor`; each `RaggedFeature` is mapped to a `RaggedTensor`; and each
  `FixedLenSequenceFeature` is mapped to a `Tensor`, each of the specified type.
  The shape will be `(T,) + df.dense_shape` for `FixedLenSequenceFeature` `df`,
  where `T` is the length of the associated `FeatureList` in the
  `SequenceExample`. For instance, `FixedLenSequenceFeature([])` yields a scalar
  1-D `Tensor` of static shape `[None]` and dynamic shape `[T]`, while
  `FixedLenSequenceFeature([k])` (for `int k >= 1`) yields a 2-D matrix `Tensor`
  of static shape `[None, k]` and dynamic shape `[T, k]`.

  Each `SparseTensor` corresponding to `sequence_features` represents a ragged
  vector.  Its indices are `[time, index]`, where `time` is the `FeatureList`
  entry and `index` is the value's index in the list of values associated with
  that time.

  `FixedLenFeature` entries with a `default_value` and `FixedLenSequenceFeature`
  entries with `allow_missing=True` are optional; otherwise, we will fail if
  that `Feature` or `FeatureList` is missing from any example in `serialized`.

  `example_name` may contain a descriptive name for the corresponding serialized
  proto. This may be useful for debugging purposes, but it has no effect on the
  output. If not `None`, `example_name` must be a scalar.

  Note that the batch version of this function, `tf.parse_sequence_example`,
  is written for better memory efficiency and will be faster on large
  `SequenceExample`s.

  Args:
    serialized: A scalar (0-D Tensor) of type string, a single binary
      serialized `SequenceExample` proto.
    context_features: A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` or `RaggedFeature` values. These features are associated
      with a `SequenceExample` as a whole.
    sequence_features: A `dict` mapping feature keys to
      `FixedLenSequenceFeature` or `VarLenFeature` or `RaggedFeature` values.
      These features are associated with data within the `FeatureList` section
      of the `SequenceExample` proto.
    example_name: A scalar (0-D Tensor) of strings (optional), the name of
      the serialized proto.
    name: A name for this operation (optional).

  Returns:
    A tuple of two `dict`s, each mapping keys to `Tensor`s and `SparseTensor`s
    and `RaggedTensor`s.

    * The first dict contains the context key/values.
    * The second dict contains the feature_list key/values.

  Raises:
    ValueError: if any feature is invalid.
  """
  # pylint: enable=line-too-long
  if not (context_features or sequence_features):
    raise ValueError("Missing features.")
  context_params = _ParseOpParams.from_features(
      context_features, [VarLenFeature, FixedLenFeature, RaggedFeature])
  feature_list_params = _ParseOpParams.from_features(
      sequence_features,
      [VarLenFeature, FixedLenSequenceFeature, RaggedFeature])

  with ops.name_scope(name, "ParseSingleSequenceExample",
                      [serialized, example_name]):
    context_output, feature_list_output = (
        _parse_single_sequence_example_raw(serialized, context_params,
                                           feature_list_params, example_name,
                                           name))

    if context_params.ragged_keys:
      context_output = _construct_tensors_for_composite_features(
          context_features, context_output)
    if feature_list_params.ragged_keys:
      feature_list_output = _construct_tensors_for_composite_features(
          sequence_features, feature_list_output)

    return context_output, feature_list_output


def _parse_single_sequence_example_raw(serialized,
                                       context,
                                       feature_list,
                                       debug_name,
                                       name=None):
  """Parses a single `SequenceExample` proto.

  Args:
    serialized: A scalar (0-D Tensor) of type string, a single binary serialized
      `SequenceExample` proto.
    context: A `ParseOpParams` containing the parameters for the parse op for
      the context features.
    feature_list: A `ParseOpParams` containing the parameters for the parse op
      for the feature_list features.
    debug_name: A scalar (0-D Tensor) of strings (optional), the name of the
      serialized proto.
    name: A name for this operation (optional).

  Returns:
    A tuple of two `dict`s, each mapping keys to `Tensor`s and `SparseTensor`s.
    The first dict contains the context key/values.
    The second dict contains the feature_list key/values.

  Raises:
    TypeError: if feature_list.dense_defaults is not either None or a dict.
  """
  with ops.name_scope(name, "ParseSingleExample", [serialized, debug_name]):
    serialized = ops.convert_to_tensor(serialized, name="serialized")
    serialized = _assert_scalar(serialized, "serialized")
  return _parse_sequence_example_raw(serialized, debug_name, context,
                                     feature_list, name)[:2]


@tf_export("io.decode_raw", v1=[])
def decode_raw(input_bytes,
               out_type,
               little_endian=True,
               fixed_length=None,
               name=None):
  """Convert raw byte strings into tensors.

  Args:
    input_bytes:
      Each element of the input Tensor is converted to an array of bytes.
    out_type:
      `DType` of the output. Acceptable types are `half`, `float`, `double`,
      `int32`, `uint16`, `uint8`, `int16`, `int8`, `int64`.
    little_endian:
      Whether the `input_bytes` data is in little-endian format. Data will be
      converted into host byte order if necessary.
    fixed_length:
      If set, the first `fixed_length` bytes of each element will be converted.
      Data will be zero-padded or truncated to the specified length.

      `fixed_length` must be a multiple of the size of `out_type`.
      `fixed_length` must be specified if the elements of `input_bytes` are of
      variable length.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` object storing the decoded bytes.

  """
  if fixed_length is not None:
    return gen_parsing_ops.decode_padded_raw(
        input_bytes,
        fixed_length=fixed_length,
        out_type=out_type,
        little_endian=little_endian,
        name=name)
  else:
    return gen_parsing_ops.decode_raw(
        input_bytes, out_type, little_endian=little_endian, name=name)


@tf_export(v1=["decode_raw", "io.decode_raw"])
@deprecation.deprecated_args(None,
                             "bytes is deprecated, use input_bytes instead",
                             "bytes")
def decode_raw_v1(
    input_bytes=None,
    out_type=None,
    little_endian=True,
    name=None,
    bytes=None  # pylint: disable=redefined-builtin
):
  """Convert raw byte strings into tensors.

  Args:
    input_bytes:
      Each element of the input Tensor is converted to an array of bytes.
    out_type:
      `DType` of the output. Acceptable types are `half`, `float`, `double`,
      `int32`, `uint16`, `uint8`, `int16`, `int8`, `int64`.
    little_endian:
      Whether the `input_bytes` data is in little-endian format. Data will be
      converted into host byte order if necessary.
    name: A name for the operation (optional).
    bytes: Deprecated parameter. Use `input_bytes` instead.

  Returns:
    A `Tensor` object storing the decoded bytes.
  """
  input_bytes = deprecation.deprecated_argument_lookup("input_bytes",
                                                       input_bytes, "bytes",
                                                       bytes)

  # out_type is a required positional argument in the original API, and had to
  # be changed to a keyword argument in order to facilitate the transition from
  # the reserved named `bytes` to `input_bytes`. Ensure it's still set.
  if out_type is None:
    raise ValueError(
        "decode_raw_v1() missing 1 positional argument: 'out_type'")

  return gen_parsing_ops.decode_raw(
      input_bytes, out_type, little_endian=little_endian, name=name)


# Swap `name` and `na_value` for backward compatibility.
@tf_export(v1=["io.decode_csv", "decode_csv"])
@deprecation.deprecated_endpoints("decode_csv")
def decode_csv(records,
               record_defaults,
               field_delim=",",
               use_quote_delim=True,
               name=None,
               na_value="",
               select_cols=None):
  """Convert CSV records to tensors. Each column maps to one tensor.

  RFC 4180 format is expected for the CSV records.
  (https://tools.ietf.org/html/rfc4180)
  Note that we allow leading and trailing spaces with int or float field.

  Args:
    records: A `Tensor` of type `string`.
      Each string is a record/row in the csv and all records should have
      the same format.
    record_defaults: A list of `Tensor` objects with specific types.
      Acceptable types are `float32`, `float64`, `int32`, `int64`, `string`.
      One tensor per column of the input record, with either a
      scalar default value for that column or an empty vector if the column is
      required.
    field_delim: An optional `string`. Defaults to `","`.
      char delimiter to separate fields in a record.
    use_quote_delim: An optional `bool`. Defaults to `True`.
      If false, treats double quotation marks as regular
      characters inside of the string fields (ignoring RFC 4180, Section 2,
      Bullet 5).
    name: A name for the operation (optional).
    na_value: Additional string to recognize as NA/NaN.
    select_cols: Optional sorted list of column indices to select. If specified,
      only this subset of columns will be parsed and returned.

  Returns:
    A list of `Tensor` objects. Has the same type as `record_defaults`.
    Each tensor will have the same shape as records.

  Raises:
    ValueError: If any of the arguments is malformed.
  """
  return decode_csv_v2(
      records, record_defaults,
      field_delim, use_quote_delim,
      na_value, select_cols, name
      )


@tf_export("io.decode_csv", v1=[])
def decode_csv_v2(records,
                  record_defaults,
                  field_delim=",",
                  use_quote_delim=True,
                  na_value="",
                  select_cols=None,
                  name=None):
  """Convert CSV records to tensors. Each column maps to one tensor.

  RFC 4180 format is expected for the CSV records.
  (https://tools.ietf.org/html/rfc4180)
  Note that we allow leading and trailing spaces with int or float field.

  Args:
    records: A `Tensor` of type `string`.
      Each string is a record/row in the csv and all records should have
      the same format.
    record_defaults: A list of `Tensor` objects with specific types.
      Acceptable types are `float32`, `float64`, `int32`, `int64`, `string`.
      One tensor per column of the input record, with either a
      scalar default value for that column or an empty vector if the column is
      required.
    field_delim: An optional `string`. Defaults to `","`.
      char delimiter to separate fields in a record.
    use_quote_delim: An optional `bool`. Defaults to `True`.
      If false, treats double quotation marks as regular
      characters inside of the string fields (ignoring RFC 4180, Section 2,
      Bullet 5).
    na_value: Additional string to recognize as NA/NaN.
    select_cols: Optional sorted list of column indices to select. If specified,
      only this subset of columns will be parsed and returned.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects. Has the same type as `record_defaults`.
    Each tensor will have the same shape as records.

  Raises:
    ValueError: If any of the arguments is malformed.
  """
  if select_cols is not None and any(select_cols[i] >= select_cols[i + 1]
                                     for i in range(len(select_cols) - 1)):
    raise ValueError("select_cols is not strictly increasing.")
  if select_cols is not None and select_cols[0] < 0:
    raise ValueError("select_cols contains negative values.")
  if select_cols is not None and len(select_cols) != len(record_defaults):
    raise ValueError("Length of select_cols and record_defaults do not match.")
  return gen_parsing_ops.decode_csv(
      records=records,
      record_defaults=record_defaults,
      field_delim=field_delim,
      use_quote_delim=use_quote_delim,
      na_value=na_value,
      name=name,
      select_cols=select_cols,
  )


def _assert_scalar(value, name):
  """Asserts that `value` is scalar, and returns `value`."""
  value_rank = value.shape.rank
  if value_rank is None:
    check = control_flow_ops.Assert(
        math_ops.equal(array_ops.rank(value), 0),
        ["Input %s must be a scalar" % name],
        name="%sIsScalar" % name.capitalize())
    result = control_flow_ops.with_dependencies([check],
                                                value,
                                                name="%sDependencies" % name)
    result.set_shape([])
    return result
  elif value_rank == 0:
    return value
  else:
    raise ValueError("Input %s must be a scalar" % name)
=======


  Example input, format, and output: Dense Inputs (with defaults)
  ===============================================================

  Given two brain.Example input protos:

  serialized:  // serialized versions of the protos below
    [features: {
      feature: { key: "weight" value: { float_list: { value: [ 1.0 ] } } }
     },
     features: {
      feature: { key: "label" value: { float_list: { value: [ -1.0, 0.0 ] } } }
    }]
  names: ["input0", "input1"],
  dense_keys: np.array(["label", "weight"])
  dense_defaults: {
    "label": [1.0, 2.0],  # float (default: vector)
    "weight": 5.0         # float (default: scalar, 5.0)
  }
  dense_shapes: [(2,), (1,)]  # age, gender, label, weight

  Then the expected output is a dictionary:
  {
    "label": [[1.0, 2.0], [-1.0, 0.0]],
    "weight": [[1.0], [5.0]],
  }
  """
  names = [] if names is None else names
  dense_defaults = {} if dense_defaults is None else dense_defaults
  sparse_keys = [] if sparse_keys is None else sparse_keys
  sparse_types = [] if sparse_types is None else sparse_types
  dense_keys = [] if dense_keys is None else dense_keys
  dense_types = [] if dense_types is None else dense_types
  dense_shapes = [
      []] * len(dense_keys) if dense_shapes is None else dense_shapes

  num_dense = len(dense_keys)
  num_sparse = len(sparse_keys)

  if len(dense_shapes) != num_dense:
    raise ValueError("len(dense_shapes) != len(dense_keys): %d vs. %d"
                     % (len(dense_shapes), num_dense))
  if len(dense_types) != num_dense:
    raise ValueError("len(dense_types) != len(num_dense): %d vs. %d"
                     % (len(dense_types), num_dense))
  if len(sparse_types) != num_sparse:
    raise ValueError("len(sparse_types) != len(sparse_keys): %d vs. %d"
                     % (len(sparse_types), num_sparse))
  if num_dense + num_sparse == 0:
    raise ValueError("Must provide at least one sparse key or dense key")
  if not set(dense_keys).isdisjoint(set(sparse_keys)):
    raise ValueError(
        "Dense and sparse keys must not intersect; intersection: %s" %
        set(dense_keys).intersection(set(sparse_keys)))

  dense_defaults_vec = []
  for i, key in enumerate(dense_keys):
    default_value = dense_defaults.get(key)
    if default_value is None:
      default_value = constant_op.constant([], dtype=dense_types[i])
    elif not isinstance(default_value, ops.Tensor):
      default_value = ops.convert_to_tensor(
          default_value, dtype=dense_types[i], name=key)
      default_value = array_ops.reshape(default_value, dense_shapes[i])

    dense_defaults_vec.append(default_value)

  dense_shapes = [tensor_util.MakeTensorShapeProto(shape)
                  if isinstance(shape, (list, tuple)) else shape
                  for shape in dense_shapes]

  outputs = gen_parsing_ops._parse_example(
      serialized=serialized,
      names=names,
      dense_defaults=dense_defaults_vec,
      sparse_keys=sparse_keys,
      sparse_types=sparse_types,
      dense_keys=dense_keys,
      dense_shapes=dense_shapes,
      name=name)

  (sparse_indices, sparse_values, sparse_shapes, dense_values) = outputs

  sparse_tensors = [ops.SparseTensor(ix, val, shape) for (ix, val, shape)
                    in zip(sparse_indices, sparse_values, sparse_shapes)]

  return dict(
      zip(sparse_keys + dense_keys, sparse_tensors + dense_values))


def parse_single_example(serialized,  # pylint: disable=invalid-name
                         names=None,
                         sparse_keys=None,
                         sparse_types=None,
                         dense_keys=None,
                         dense_types=None,
                         dense_defaults=None,
                         dense_shapes=None,
                         name="ParseSingleExample"):
  """Identical to parse_example but for scalar serialized and names.

  Args:
    serialized: A scalar string, a single serialized Example.
      See parse_example documentation for more details.
    names: (Optional) A scalar string, the associated name.
      See parse_example documentation for more details.
    sparse_keys: See parse_example documentation for more details.
    sparse_types: See parse_example documentation for more details.
    dense_keys: See parse_example documentation for more details.
    dense_types: See parse_example documentation for more details.
    dense_defaults: See parse_example documentation for more details.
    dense_shapes: See parse_example documentation for more details.
    name: Optional op name.

  Returns:
    A dictionary mapping keys to Tensors and SparseTensors.

    For dense tensors, the Tensor is identical to the output of parse_example,
    except it is one less dimension (the first, batch, dimension is removed).

    For SparseTensors:
      The first (batch) column of the indices matrix is removed
        (it is now a column vector).
      The values vector is unchanged.
      The first (batch_size) entry of the shape vector is removed
        (it is now a single element vector).

  Raises:
    ValueError: if "scalar" or "names" have known shapes, and are not scalars.
  """
  with ops.op_scope([serialized], name, "parse_single_example"):
    serialized = ops.convert_to_tensor(serialized)
    serialized_shape = serialized.get_shape()
    if serialized_shape.ndims is not None:
      if serialized_shape.ndims != 0:
        raise ValueError("Input serialized must be a scalar")
    else:
      serialized = control_flow_ops.with_dependencies(
          [logging_ops.Assert(
              math_ops.equal(array_ops.rank(serialized), 0),
              ["Input serialized must be a scalar"],
              name="SerializedIsScalar")],
          serialized,
          name="SerializedDependencies")
    serialized = array_ops.expand_dims(serialized, 0)
    if names is not None:
      names = ops.convert_to_tensor(names)
      names_shape = names.get_shape()
      if names_shape.ndims is not None:
        if names_shape.ndims != 0:
          raise ValueError("Input names must be a scalar")
      else:
        names = control_flow_ops.with_dependencies(
            [logging_ops.Assert(
                math_ops.equal(array_ops.rank(names), 0),
                ["Input names must be a scalar"],
                name="NamesIsScalar")],
            names,
            name="NamesDependencies")
      names = array_ops.expand_dims(names, 0)

    outputs = parse_example(serialized,
                            names=names,
                            sparse_keys=sparse_keys,
                            sparse_types=sparse_types,
                            dense_keys=dense_keys,
                            dense_types=dense_types,
                            dense_defaults=dense_defaults,
                            dense_shapes=dense_shapes,
                            name=name)
    if dense_keys is not None:
      for d in dense_keys:
        outputs[d] = array_ops.squeeze(outputs[d], [0], name="Squeeze_%s" % d)
    if sparse_keys is not None:
      for s in sparse_keys:
        outputs[s] = ops.SparseTensor(
            array_ops.slice(outputs[s].indices,
                            [0, 1], [-1, -1], name="Slice_Indices_%s" % s),
            outputs[s].values,
            array_ops.slice(outputs[s].shape,
                            [1], [-1], name="Squeeze_Shape_%s" % s))
    return outputs


@ops.RegisterShape("ParseExample")
def _ParseExampleShape(op):
  """Shape function for the ParseExample op."""
  input_shape = op.inputs[0].get_shape().with_rank(1)
  num_sparse = op.get_attr("Nsparse")
  num_dense = op.get_attr("Ndense")
  dense_shapes = op.get_attr("dense_shapes")
  sparse_index_shapes = [
      tensor_shape.matrix(None, 2) for _ in range(num_sparse)]
  sparse_value_shapes = [tensor_shape.vector(None) for _ in range(num_sparse)]
  sparse_shape_shapes = [tensor_shape.vector(2) for _ in range(num_sparse)]
  assert num_dense == len(dense_shapes)
  dense_shapes = [
      input_shape.concatenate((d.size for d in dense_shape.dim))
      for dense_shape in dense_shapes]
  return (sparse_index_shapes + sparse_value_shapes + sparse_shape_shapes +
          dense_shapes)


ops.RegisterShape("StringToNumber")(
    common_shapes.unchanged_shape)


@ops.RegisterShape("DecodeRaw")
def _DecodeRawShape(op):
  """Shape function for the DecodeRaw op."""
  # NOTE(mrry): Last dimension is data-dependent.
  return [op.inputs[0].get_shape().concatenate([None])]


@ops.RegisterShape("DecodeCSV")
def _DecodeCSVShape(op):
  """Shape function for the DecodeCSV op."""
  input_shape = op.inputs[0].get_shape()
  # Optionally check that all of other inputs are scalar or empty.
  for default_input in op.inputs[1:]:
    default_input_shape = default_input.get_shape().with_rank(1)
    if default_input_shape[0] > 1:
      raise ValueError(
          "Shape of a default must be a length-0 or length-1 vector.")
  return [input_shape] * len(op.outputs)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
