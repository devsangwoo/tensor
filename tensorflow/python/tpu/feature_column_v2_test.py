# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ===================================================================
"""Tests for python.tpu.feature_column."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.client import session
from tensorflow.python.feature_column import feature_column_lib as fc_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.tpu import feature_column_v2 as tpu_fc


def _initialized_session():
  sess = session.Session()
  sess.run(variables_lib.global_variables_initializer())
  sess.run(lookup_ops.tables_initializer())
  return sess


class EmbeddingColumnTestV2(test.TestCase):

  def test_defaults(self):
    categorical_column = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_dimension = 2
    embedding_column = tpu_fc.embedding_column_v2(
        categorical_column, dimension=embedding_dimension)
    # Can't test default initializer as it's a random function.
    self.assertIs(categorical_column, embedding_column.categorical_column)
    self.assertEqual(embedding_dimension, embedding_column.dimension)
    self.assertEqual('mean', embedding_column.combiner)
    self.assertEqual('aaa_embedding', embedding_column.name)
    self.assertEqual((embedding_dimension,), embedding_column.variable_shape)

  def test_all_constructor_args(self):
    categorical_column = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_dimension = 2
    embedding_column = tpu_fc.embedding_column_v2(
        categorical_column,
        dimension=embedding_dimension,
        combiner='my_combiner',
        initializer=lambda: 'my_initializer')
    self.assertIs(categorical_column, embedding_column.categorical_column)
    self.assertEqual(embedding_dimension, embedding_column.dimension)
    self.assertEqual('my_combiner', embedding_column.combiner)
    self.assertEqual('my_initializer', embedding_column.initializer())
    self.assertEqual('aaa_embedding', embedding_column.name)
    self.assertEqual((embedding_dimension,), embedding_column.variable_shape)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int64)
    }, embedding_column._parse_example_spec)

  @test_util.deprecated_graph_mode_only
  def test_feature_layer_cpu(self):
    # Inputs.
    vocabulary_size = 3
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        # example 2, ids []
        # example 3, ids [1]
        indices=((0, 0), (1, 0), (1, 1), (3, 0)),
        values=(2, 0, 1, 1),
        dense_shape=(4, 2))

    # Embedding variable.
    embedding_dimension = 2
    embedding_values = (
        (1., 2.),  # id 0
        (3., 5.),  # id 1
        (7., 11.)  # id 2
    )

    def _initializer(shape, dtype, partition_info=None):
      self.assertAllEqual((vocabulary_size, embedding_dimension), shape)
      self.assertEqual(dtypes.float32, dtype)
      self.assertIsNone(partition_info)
      return embedding_values

    # Expected lookup result, using combiner='mean'.
    expected_lookups = (
        # example 0, ids [2], embedding = [7, 11]
        (7., 11.),
        # example 1, ids [0, 1], embedding = mean([1, 2] + [3, 5]) = [2, 3.5]
        (2., 3.5),
        # example 2, ids [], embedding = [0, 0]
        (0., 0.),
        # example 3, ids [1], embedding = [3, 5]
        (3., 5.),
    )
    expected_lookups_sequence = (
        # example 0, ids [2], embedding = [[7, 11], [0, 0]]
        ((7., 11.), (0., 0.),),
        # example 1, ids [0, 1], embedding = [[1, 2], [3. 5]]
        ((1., 2.), (3., 5.),),
        # example 2, ids [], embedding = [0, 0]
        ((0., 0.), (0., 0.),),
        # example 3, ids [1], embedding = [3, 5]
        ((3., 5.), (0., 0.),),
    )

    # Build columns.
    categorical_column = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    sequence_categorical_column = (
        fc_lib.sequence_categorical_column_with_identity(
            key='bbb', num_buckets=vocabulary_size))
    embedding_column = tpu_fc.embedding_column_v2(
        categorical_column,
        dimension=embedding_dimension,
        initializer=_initializer)
    sequence_embedding_column = tpu_fc.embedding_column_v2(
        sequence_categorical_column,
        dimension=embedding_dimension,
        initializer=_initializer,
        max_sequence_length=2)

    # Provide sparse input and get dense result.
    features = {'aaa': sparse_input, 'bbb': sparse_input}
    dense_features = fc_lib.DenseFeatures([embedding_column])
    sequence_features = fc_lib.SequenceFeatures([sequence_embedding_column])
    embedding_lookup = dense_features(features)
    sequence_embedding_lookup = sequence_features(features)

    # Assert expected embedding variable and lookups.
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(
        ('dense_features/aaa_embedding/embedding_weights:0',
         'sequence_features/bbb_embedding/embedding_weights:0',),
        tuple([v.name for v in global_vars]))
    with _initialized_session():
      self.assertAllEqual(embedding_values, global_vars[0].eval())
      self.assertAllEqual(expected_lookups, embedding_lookup.eval())
      self.assertAllEqual(expected_lookups_sequence,
                          sequence_embedding_lookup[0].eval())


class SharedEmbeddingColumnTestV2(test.TestCase):

  @test_util.deprecated_graph_mode_only
  def test_defaults(self):
    vocabulary_size = 3
    categorical_column_a = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    categorical_column_b = fc_lib.categorical_column_with_identity(
        key='bbb', num_buckets=vocabulary_size)
    embedding_dimension = 2
    embedding_column_b, embedding_column_a = tpu_fc.shared_embedding_columns_v2(
        [categorical_column_b, categorical_column_a],
        dimension=embedding_dimension)
    self.assertIs(categorical_column_a, embedding_column_a.categorical_column)
    self.assertIs(categorical_column_b, embedding_column_b.categorical_column)
    self.assertEqual((vocabulary_size, embedding_dimension),
                     embedding_column_a.get_embedding_table_size())
    self.assertEqual((vocabulary_size, embedding_dimension),
                     embedding_column_a.get_embedding_table_size())
    self.assertEqual('mean', embedding_column_a.combiner)
    self.assertEqual('mean', embedding_column_b.combiner)
    self.assertIsNotNone(embedding_column_a.get_initializer())
    self.assertIsNotNone(embedding_column_b.get_initializer())
    self.assertEqual('aaa_bbb_shared_embedding',
                     embedding_column_a.get_embedding_var_name())
    self.assertEqual('aaa_bbb_shared_embedding',
                     embedding_column_b.get_embedding_var_name())
    self.assertEqual('aaa_shared_embedding', embedding_column_a.name)
    self.assertEqual('bbb_shared_embedding', embedding_column_b.name)
    self.assertEqual((embedding_dimension,), embedding_column_a.variable_shape)
    self.assertEqual((embedding_dimension,), embedding_column_b.variable_shape)

  @test_util.deprecated_graph_mode_only
  def test_all_constructor_args(self):
    vocabulary_size = 3
    categorical_column_a = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    categorical_column_b = fc_lib.categorical_column_with_identity(
        key='bbb', num_buckets=vocabulary_size)
    embedding_dimension = 2
    embedding_column_a, embedding_column_b = tpu_fc.shared_embedding_columns_v2(
        [categorical_column_a, categorical_column_b],
        dimension=embedding_dimension,
        combiner='my_combiner',
        initializer=lambda: 'my_initializer',
        shared_embedding_collection_name='var_scope_name')
    self.assertIs(categorical_column_a, embedding_column_a.categorical_column)
    self.assertIs(categorical_column_b, embedding_column_b.categorical_column)
    self.assertEqual((vocabulary_size, embedding_dimension),
                     embedding_column_a.get_embedding_table_size())
    self.assertEqual((vocabulary_size, embedding_dimension),
                     embedding_column_a.get_embedding_table_size())
    self.assertEqual('my_combiner', embedding_column_a.combiner)
    self.assertEqual('my_combiner', embedding_column_b.combiner)
    self.assertEqual('my_initializer', embedding_column_a.get_initializer()())
    self.assertEqual('my_initializer', embedding_column_b.get_initializer()())
    self.assertEqual('var_scope_name',
                     embedding_column_a.get_embedding_var_name())
    self.assertEqual('var_scope_name',
                     embedding_column_b.get_embedding_var_name())
    self.assertEqual('aaa_shared_embedding', embedding_column_a.name)
    self.assertEqual('bbb_shared_embedding', embedding_column_b.name)
    self.assertEqual((embedding_dimension,), embedding_column_a.variable_shape)
    self.assertEqual((embedding_dimension,), embedding_column_b.variable_shape)

  @test_util.deprecated_graph_mode_only
  def test_feature_layer_cpu(self):
    # Inputs.
    vocabulary_size = 3
    input_a = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        indices=((0, 0), (1, 0), (1, 1)),
        values=(2, 0, 1),
        dense_shape=(2, 2))
    input_b = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        # example 2, ids []
        indices=((0, 0), (1, 0), (1, 1)),
        values=(2, 0, 1),
        dense_shape=(3, 2))
    input_features = {'aaa': input_a, 'bbb': input_b}

    # Embedding variable.
    embedding_dimension = 2
    embedding_values = (
        (1., 2.),  # id 0
        (3., 5.),  # id 1
        (7., 11.)  # id 2
    )

    def _initializer(shape, dtype, partition_info=None):
      self.assertAllEqual((vocabulary_size, embedding_dimension), shape)
      self.assertEqual(dtypes.float32, dtype)
      self.assertIsNone(partition_info)
      return embedding_values

    # Expected lookup result, using combiner='mean'.
    expected_lookups_a = (
        # example 0:
        (7., 11.),  # ids [2], embedding = [7, 11]
        # example 1:
        (2., 3.5),  # ids [0, 1], embedding = mean([1, 2] + [3, 5]) = [2, 3.5]
    )
    expected_lookups_b = (
        # example 0:
        ((7., 11.), (0., 0.),),  # ids [2], embedding = [[7, 11], [0, 0]]
        # example 1:
        ((1., 2.), (3., 5.),),  # ids [0, 1], embedding = [[1, 2], [3, 5]]
        # example 2:
        ((0., 0.), (0., 0.),),  # ids [], embedding = [[0, 0], [0, 0]]
    )

    # Build columns.
    categorical_column_a = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    categorical_column_b = fc_lib.sequence_categorical_column_with_identity(
        key='bbb', num_buckets=vocabulary_size)
    embedding_column_a, embedding_column_b = tpu_fc.shared_embedding_columns_v2(
        [categorical_column_a, categorical_column_b],
        dimension=embedding_dimension,
        initializer=_initializer,
        max_sequence_lengths=[0, 2])

    # Provide sparse input and get dense result.
    dense_features = fc_lib.DenseFeatures([embedding_column_a])
    sequence_features = fc_lib.SequenceFeatures([embedding_column_b])
    embedding_lookup_a = dense_features(input_features)
    embedding_lookup_b = sequence_features(input_features)

    # Assert expected embedding variable and lookups.
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(
        ('aaa_bbb_shared_embedding:0',),
        tuple([v.name for v in global_vars]))
    embedding_var = global_vars[0]
    with _initialized_session():
      self.assertAllEqual(embedding_values, embedding_var.eval())
      self.assertAllEqual(expected_lookups_a, embedding_lookup_a.eval())
      self.assertAllEqual(expected_lookups_b,
                          embedding_lookup_b[0].eval())


if __name__ == '__main__':
  test.main()
