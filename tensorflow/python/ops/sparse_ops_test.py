<<<<<<< HEAD
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for sparse ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
# Need array_grad to register gradient for Identity.
from tensorflow.python.ops import array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import gradient_checker_v2 as gradient_checker
from tensorflow.python.ops import math_ops
# Need sparse_grad to register gradient for SparseToDense.
from tensorflow.python.ops import sparse_grad  # pylint: disable=unused-import
=======
"""Tests for Python ops defined in sparse_ops."""

import tensorflow.python.platform

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import types
from tensorflow.python.ops import constant_op
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import googletest


<<<<<<< HEAD
@test_util.run_all_in_graph_and_eager_modes
class SparseOpsTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def testSparseEye(self):
    def test_one(n, m, as_tensors):
      expected = np.eye(n, m)
      if as_tensors:
        m = constant_op.constant(m)
        n = constant_op.constant(n)
      s = sparse_ops.sparse_eye(n, m)
      d = sparse_ops.sparse_to_dense(s.indices, s.dense_shape, s.values)
      self.assertAllEqual(self.evaluate(d), expected)

    for n in range(2, 10, 2):
      for m in range(2, 10, 2):
        # Test with n and m as both constants and tensors.
        test_one(n, m, True)
        test_one(n, m, False)

  def testDenseFromConstantToSparse(self):
    expected_constant = np.reshape(np.arange(24, dtype=np.int64), (3, 4, 2))
    tensor = constant_op.constant(expected_constant)
    sparse = sparse_ops.from_dense(tensor)
    dense = sparse_ops.sparse_to_dense(sparse.indices, sparse.dense_shape,
                                       sparse.values)
    constant = self.evaluate(dense)
    self.assertAllEqual(expected_constant, constant)

  def testTransposePreservesShape(self):
    with ops.Graph().as_default():
      t = sparse_tensor.SparseTensor(indices=[[0, 0]],
                                     values=[0.],
                                     dense_shape=[3, 4])
      self.assertTrue(t.shape.is_fully_defined)
      transposed = sparse_ops.sparse_transpose(t)
      self.assertAllEqual(transposed.shape, [4, 3])

  def testSparseExpandDims(self):
    for rank in range(1, 4):
      # Create a dummy input. When rank=3, shape=[2, 4, 6].
      shape = np.arange(1, rank + 1) * 2
      before = np.arange(np.prod(shape)).reshape(shape)

      # Make entries sparse.
      before *= np.random.binomial(1, .2, before.shape)
      dense_shape = before.shape
      indices = np.array(np.where(before)).T
      values = before[before != 0]

      # Try every possible valid value of axis.
      for axis in range(-rank - 1, rank):
        expected_after = np.expand_dims(before, axis)

        for axis_as_tensor in [False, True]:
          dense_shape_t = constant_op.constant(dense_shape, dtype=dtypes.int64)
          indices_t = constant_op.constant(indices)
          values_t = constant_op.constant(values)
          before_t = sparse_tensor.SparseTensor(
              indices=indices_t, values=values_t, dense_shape=dense_shape_t)

          if axis_as_tensor:
            axis = constant_op.constant(axis)

          s = sparse_ops.sparse_expand_dims(before_t, axis)
          d = sparse_ops.sparse_to_dense(s.indices, s.dense_shape, s.values)
          self.assertAllEqual(self.evaluate(d), expected_after)

  @parameterized.parameters([
      (math_ops.abs, [1.0, -1.0, 3.0, -4.0], [1.0, 1.0, 3.0, 4.0]),
      (math_ops.negative, [1.0, -1.0, 3.0, -4.0], [-1.0, 1.0, -3.0, 4.0]),
      (math_ops.sign, [3.0, -2.0, 0.0, -4.0], [1.0, -1.0, 0.0, -1.0]),
      (math_ops.square, [1.0, -1.0, 3.0, -4.0], [1.0, 1.0, 9.0, 16.0]),
  ])
  def testUnarySparseDispatch(self, op, values, expected):
    st = sparse_tensor.SparseTensor(
        indices=[[0, 0], [0, 1], [2, 0], [2, 4]],
        values=values,
        dense_shape=[3, 6])
    result = op(st)
    result_value = self.evaluate(result)
    self.assertAllEqual(result_value.indices, st.indices)
    self.assertAllEqual(result_value.values, expected)
    self.assertAllEqual(result_value.dense_shape, st.dense_shape)

  def testSparseToDenseGradient(self):

    def f(sparse_values, default_value):
      st = sparse_tensor.SparseTensor(
          indices=[[0, 3, 6], [1, 4, 7], [2, 5, 8]],
          values=sparse_values,
          dense_shape=[3, 6, 9])
      return sparse_ops.sparse_tensor_to_dense(st, default_value)

    grads = gradient_checker.compute_gradient(
        f, [constant_op.constant([1.0, 2.0, 3.0]),
            constant_op.constant(0.0)])
    epsilon = 1e-4
    self.assertLess(gradient_checker.max_error(*grads), epsilon)

  def testSparseTensorToDenseString(self):
    sp = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]], values=['a', 'b'], dense_shape=[2, 3])
    dense = sparse_ops.sparse_tensor_to_dense(sp)
    expected_dense = [[b'a', b'', b''], [b'', b'', b'b']]
    result_dense = self.evaluate(dense)
    self.assertAllEqual(expected_dense, result_dense)


if __name__ == '__main__':
=======
class SparseToIndicatorTest(test_util.TensorFlowTestCase):

  def _SparseTensor_5x6(self, dtype):
    ind = np.array([
        [0, 0],
        [1, 0], [1, 3], [1, 4],
        [3, 2], [3, 3]])
    val = np.array([0, 10, 13, 14, 32, 33])
    shape = np.array([5, 6])
    return ops.SparseTensor(
        constant_op.constant(ind, types.int64),
        constant_op.constant(val, dtype),
        constant_op.constant(shape, types.int64))

  def _SparseTensor_2x3x4(self, dtype):
    ind = np.array([
        [0, 0, 1],
        [0, 1, 0], [0, 1, 2],
        [1, 0, 3],
        [1, 1, 1], [1, 1, 3],
        [1, 2, 2]])
    val = np.array([1, 10, 12, 103, 111, 113, 122])
    shape = np.array([2, 3, 4])
    return ops.SparseTensor(
        constant_op.constant(ind, types.int64),
        constant_op.constant(val, dtype),
        constant_op.constant(shape, types.int64))

  def testInt32(self):
    with self.test_session(use_gpu=False):
      sp_input = self._SparseTensor_5x6(types.int32)
      output = sparse_ops.sparse_to_indicator(sp_input, 50).eval()

      expected_output = np.zeros((5, 50), dtype=np.bool)
      expected_trues = ((0, 0), (1, 10), (1, 13), (1, 14), (3, 32), (3, 33))
      for expected_true in expected_trues:
        expected_output[expected_true] = True

      self.assertAllEqual(output, expected_output)

  def testInt64(self):
    with self.test_session(use_gpu=False):
      sp_input = self._SparseTensor_5x6(types.int64)
      output = sparse_ops.sparse_to_indicator(sp_input, 50).eval()

      expected_output = np.zeros((5, 50), dtype=np.bool)
      expected_trues = [(0, 0), (1, 10), (1, 13), (1, 14), (3, 32), (3, 33)]
      for expected_true in expected_trues:
        expected_output[expected_true] = True

      self.assertAllEqual(output, expected_output)

  def testHigherRank(self):
    with self.test_session(use_gpu=False):
      sp_input = self._SparseTensor_2x3x4(types.int64)
      output = sparse_ops.sparse_to_indicator(sp_input, 200).eval()

      expected_output = np.zeros((2, 3, 200), dtype=np.bool)
      expected_trues = [(0, 0, 1), (0, 1, 10), (0, 1, 12),
                        (1, 0, 103), (1, 1, 111), (1, 1, 113), (1, 2, 122)]
      for expected_true in expected_trues:
        expected_output[expected_true] = True

      self.assertAllEqual(output, expected_output)


class SparseRetainTest(test_util.TensorFlowTestCase):

  def _SparseTensor_5x6(self):
    ind = np.array([
        [0, 0],
        [1, 0], [1, 3], [1, 4],
        [3, 2], [3, 3]])
    val = np.array([0, 10, 13, 14, 32, 33])
    shape = np.array([5, 6])
    return ops.SparseTensor(
        constant_op.constant(ind, types.int64),
        constant_op.constant(val, types.int32),
        constant_op.constant(shape, types.int64))

  def testBasic(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = self._SparseTensor_5x6()
      to_retain = np.array([1, 0, 0, 1, 1, 0], dtype=np.bool)
      sp_output = sparse_ops.sparse_retain(sp_input, to_retain)

      output = sess.run(sp_output)

      self.assertAllEqual(output.indices, [[0, 0], [1, 4], [3, 2]])
      self.assertAllEqual(output.values, [0, 14, 32])
      self.assertAllEqual(output.shape, [5, 6])

  def testRetainNone(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = self._SparseTensor_5x6()
      to_retain = np.zeros((6,), dtype=np.bool)
      sp_output = sparse_ops.sparse_retain(sp_input, to_retain)

      output = sess.run(sp_output)

      self.assertAllEqual(output.indices, np.array([]).reshape((0, 2)))
      self.assertAllEqual(output.values, [])
      self.assertAllEqual(output.shape, [5, 6])

  def testMismatchedRetainShape(self):
    with self.test_session(use_gpu=False):
      sp_input = self._SparseTensor_5x6()
      to_retain = np.array([1, 0, 0, 1, 0], dtype=np.bool)
      with self.assertRaises(ValueError):
        sparse_ops.sparse_retain(sp_input, to_retain)


class SparseFillEmptyRowsTest(test_util.TensorFlowTestCase):

  def _SparseTensor_5x6(self):
    ind = np.array([
        [0, 0],
        [1, 0], [1, 3], [1, 4],
        [3, 2], [3, 3]])
    val = np.array([0, 10, 13, 14, 32, 33])
    shape = np.array([5, 6])
    return ops.SparseTensor(
        constant_op.constant(ind, types.int64),
        constant_op.constant(val, types.int32),
        constant_op.constant(shape, types.int64))

  def _SparseTensor_String5x6(self):
    ind = np.array([
        [0, 0],
        [1, 0], [1, 3], [1, 4],
        [3, 2], [3, 3]])
    val = np.array(["a", "b", "c", "d", "e", "f"])
    shape = np.array([5, 6])
    return ops.SparseTensor(
        constant_op.constant(ind, types.int64),
        constant_op.constant(val, types.string),
        constant_op.constant(shape, types.int64))

  def _SparseTensor_2x6(self):
    ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4]])
    val = np.array([0, 10, 13, 14])
    shape = np.array([2, 6])
    return ops.SparseTensor(
        constant_op.constant(ind, types.int64),
        constant_op.constant(val, types.int32),
        constant_op.constant(shape, types.int64))

  def testFillNumber(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = self._SparseTensor_5x6()
      sp_output, empty_row_indicator = (
          sparse_ops.sparse_fill_empty_rows(sp_input, -1))

      output, empty_row_indicator_out = sess.run(
          [sp_output, empty_row_indicator])

      self.assertAllEqual(
          output.indices,
          [[0, 0], [1, 0], [1, 3], [1, 4], [2, 0], [3, 2], [3, 3], [4, 0]])
      self.assertAllEqual(output.values, [0, 10, 13, 14, -1, 32, 33, -1])
      self.assertAllEqual(output.shape, [5, 6])
      self.assertAllEqual(empty_row_indicator_out,
                          np.array([0, 0, 1, 0, 1]).astype(np.bool))

  def testFillString(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = self._SparseTensor_String5x6()
      sp_output, empty_row_indicator = (
          sparse_ops.sparse_fill_empty_rows(sp_input, ""))

      output, empty_row_indicator_out = sess.run(
          [sp_output, empty_row_indicator])

      self.assertAllEqual(
          output.indices,
          [[0, 0], [1, 0], [1, 3], [1, 4], [2, 0], [3, 2], [3, 3], [4, 0]])
      self.assertAllEqual(output.values, ["a", "b", "c", "d", "", "e", "f", ""])
      self.assertAllEqual(output.shape, [5, 6])
      self.assertAllEqual(empty_row_indicator_out,
                          np.array([0, 0, 1, 0, 1]).astype(np.bool))

  def testNoEmptyRows(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = self._SparseTensor_2x6()
      sp_output, empty_row_indicator = (
          sparse_ops.sparse_fill_empty_rows(sp_input, -1))

      output, empty_row_indicator_out = sess.run(
          [sp_output, empty_row_indicator])

      self.assertAllEqual(output.indices, [[0, 0], [1, 0], [1, 3], [1, 4]])
      self.assertAllEqual(output.values, [0, 10, 13, 14])
      self.assertAllEqual(output.shape, [2, 6])
      self.assertAllEqual(empty_row_indicator_out, np.zeros(2).astype(np.bool))


if __name__ == "__main__":
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  googletest.main()
