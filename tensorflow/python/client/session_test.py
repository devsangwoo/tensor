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

"""Tests for tensorflow.python.client.session.Session."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import threading
import time

import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.core.lib.core import error_codes_pb2
from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat


# NOTE(mrry): Dummy shape registration for ops used in the tests, since they
# don't have C++ op registrations on which to attach C++ shape fns.
ops.RegisterShape('ConstructionFails')(common_shapes.unknown_shape)


class SessionTest(test_util.TensorFlowTestCase):

  def testUseExistingGraph(self):
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      a = constant_op.constant(6.0, shape=[1, 1])
      b = constant_op.constant(7.0, shape=[1, 1])
      c = math_ops.matmul(a, b, name='matmul')
    with session.Session(graph=g):
      result = c.eval()
      self.assertAllEqual(result, [[42.0]])

  def testUseDefaultGraph(self):
    with ops.Graph().as_default(), ops.device('/cpu:0'):
      a = constant_op.constant(6.0, shape=[1, 1])
      b = constant_op.constant(7.0, shape=[1, 1])
      c = math_ops.matmul(a, b, name='matmul')
      with session.Session():
        result = c.eval()
        self.assertAllEqual(result, [[42.0]])

  def testCreate(self):
    with session.Session():
      inp = constant_op.constant(10.0, shape=[2, 3], name='W1')
      copy = array_ops.identity(inp)
      # Test with feed.
      # TODO(mrry): Investigate why order='F' didn't work.
      arr = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.float32, order='C')
      copy_val = copy.eval({'W1:0': arr})
      self.assertAllEqual(arr, copy_val)
      # Test without feed.
      copy_val = copy.eval()
      self.assertAllEqual(np.asarray([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]],
                                     dtype=np.float32), copy_val)

  def testManyCPUs(self):
    # TODO(keveman): Implement ListDevices and test for the number of
    # devices returned by ListDevices.
    with session.Session(
        config=config_pb2.ConfigProto(device_count={'CPU': 2})):
      inp = constant_op.constant(10.0, name='W1')
      self.assertAllEqual(inp.eval(), 10.0)

  def testPerSessionThreads(self):
    # TODO(keveman): Implement ListDevices and test for the number of
    # devices returned by ListDevices.
    with session.Session(
        config=config_pb2.ConfigProto(use_per_session_threads=True)):
      inp = constant_op.constant(10.0, name='W1')
      self.assertAllEqual(inp.eval(), 10.0)

  def testSessionInterOpThreadPool(self):
    config = config_pb2.ConfigProto()
    pool = config.session_inter_op_thread_pool.add()
    with session.Session(config=config) as s:
      inp = constant_op.constant(10.0, name='W1')
      results = s.run([inp])
      self.assertAllEqual([10.0], results)

    pool = config.session_inter_op_thread_pool.add()
    pool.num_threads = 1
    with session.Session(config=config) as s:
      inp = constant_op.constant(20.0, name='W2')
      results = s.run([inp])
      self.assertAllEqual([20.0], results)

  def testErrorsReported(self):
    with session.Session() as s:
      constant_op.constant(10.0, name='W1')
      with self.assertRaises(ValueError):
        s.run('foo:0')

  def testErrorPayload(self):
    with session.Session():
      a = array_ops.placeholder(dtypes.float32)
      with self.assertRaisesOpError(lambda e: e.op == a.op):
        a.eval()

  def testErrorCodeWithNoNodeDef(self):
    with session.Session() as s:
      a = array_ops.placeholder(dtypes.float32, shape=[])
      b = array_ops.placeholder(dtypes.float32, shape=[])
      r1 = math_ops.add(a, b)

      def exc_predicate(e):
        return (e.op is None and e.node_def is None and
                e.error_code == error_codes_pb2.INVALID_ARGUMENT)
      with self.assertRaisesOpError(exc_predicate):
        # Run with a bogus handle.
        s.partial_run('foo', r1, feed_dict={a: 1, b: 2})

  def testOpConstructionErrorPayload(self):
    with session.Session():
      failing_op = ops.get_default_graph().create_op(
          'ConstructionFails', [], [], name='f')

      def exc_predicate(e):
        return (e.op == failing_op
                and e.error_code == error_codes_pb2.INVALID_ARGUMENT)
      with self.assertRaisesOpError(exc_predicate):
        failing_op.run()

  def testErrorBasedOn(self):
    with session.Session() as sess:
      a = constant_op.constant(0.0, shape=[2, 3])
      # NOTE(mrry): The original_op is nonsense, but used here to test that the
      #   errors are reported correctly.
      # pylint: disable=protected-access
      with sess.graph._original_op(a.op):
        b = array_ops.identity(a, name='id')
      with sess.graph._original_op(b.op):
        c = array_ops.placeholder(dtypes.float32)
      # pylint: enable=protected-access

      def exc_predicate(e):
        return (e.op == c.op
                and e.op._original_op == b.op
                and e.op._original_op._original_op == a.op)
      with self.assertRaisesOpError(exc_predicate):
        c.eval()

  def testFetchNone(self):
    with session.Session() as s:
      a = constant_op.constant(1.0)
      with self.assertRaises(TypeError):
        s.run(None)
      with self.assertRaises(TypeError):
        s.run([None])
      with self.assertRaises(TypeError):
        s.run({'b': None})
      with self.assertRaises(TypeError):
        s.run({'a': a, 'b': None})

  def testFetchSingleton(self):
    with session.Session() as sess:
      a = constant_op.constant(42.0)
      res = sess.run(a)
      self.assertEqual(42.0, res)
      res = sess.run(a.op)  # An op, not a tensor.
      self.assertEqual(None, res)
      tensor_runner = sess.make_callable(a)
      res = tensor_runner()
      self.assertEqual(42.0, res)
      op_runner = sess.make_callable(a.op)
      res = op_runner()
      self.assertEqual(None, res)

  def testFetchSingletonByName(self):
    with session.Session() as sess:
      a = constant_op.constant(42.0)
      res = sess.run(a.name)
      self.assertEqual(42.0, res)
      res = sess.run(a.op)  # An op, not a tensor.
      self.assertEqual(None, res)

  def testFetchList(self):
    with session.Session() as sess:
      a = constant_op.constant(42.0)
      b = control_flow_ops.no_op()  # An op, not a tensor.
      c = constant_op.constant(44.0)
      v = variables.Variable([54.0])
      assign = v.assign([63.0])
      res = sess.run([a, b, c, a.name, assign.op])
      self.assertTrue(isinstance(res, list))
      self.assertEqual([42.0, None, 44.0, 42.0, None], res)
      list_runner = sess.make_callable([a, b, c, a.name, assign.op])
      res = list_runner()
      self.assertTrue(isinstance(res, list))
      self.assertEqual([42.0, None, 44.0, 42.0, None], res)

  def testFetchTuple(self):
    with session.Session() as sess:
      a = constant_op.constant(42.0)
      b = control_flow_ops.no_op()  # An op, not a tensor.
      c = constant_op.constant(44.0)
      res = sess.run((a, b, c, a.name))
      self.assertTrue(isinstance(res, tuple))
      self.assertEqual((42.0, None, 44.0, 42.0), res)
      tuple_runner = sess.make_callable((a, b, c, a.name))
      res = tuple_runner()
      self.assertTrue(isinstance(res, tuple))
      self.assertEqual((42.0, None, 44.0, 42.0), res)

  def testFetchNamedTuple(self):
    # pylint: disable=invalid-name
    ABC = collections.namedtuple('ABC', ['a', 'b', 'c'])
    # pylint: enable=invalid-name
    with session.Session() as sess:
      a = constant_op.constant(42.0)
      b = control_flow_ops.no_op()  # An op, not a tensor.
      c = constant_op.constant(44.0)
      res = sess.run(ABC(a, b, c))
      self.assertTrue(isinstance(res, ABC))
      self.assertEqual(42.0, res.a)
      self.assertEqual(None, res.b)
      self.assertEqual(44.0, res.c)
      namedtuple_runner = sess.make_callable(ABC(a, b, c))
      res = namedtuple_runner()
      self.assertTrue(isinstance(res, ABC))
      self.assertEqual(42.0, res.a)
      self.assertEqual(None, res.b)
      self.assertEqual(44.0, res.c)

  def testFetchDict(self):
    with session.Session() as sess:
      a = constant_op.constant(42.0)
      b = control_flow_ops.no_op()  # An op, not a tensor.
      c = constant_op.constant(44.0)
      res = sess.run({'a': a, 'b': b, 'c': c})
      self.assertTrue(isinstance(res, dict))
      self.assertEqual(42.0, res['a'])
      self.assertEqual(None, res['b'])
      self.assertEqual(44.0, res['c'])

  def testFetchOrderedDict(self):
    with session.Session() as sess:
      a = constant_op.constant(42.0)
      b = control_flow_ops.no_op()  # An op, not a tensor.
      c = constant_op.constant(44.0)
      res = sess.run(collections.OrderedDict([(3, a), (2, b), (1, c)]))
      self.assertTrue(isinstance(res, collections.OrderedDict))
      self.assertEqual([3, 2, 1], list(res.keys()))
      self.assertEqual(42.0, res[3])
      self.assertEqual(None, res[2])
      self.assertEqual(44.0, res[1])

  def testFetchNestingEmptyOneLevel(self):
    with session.Session() as sess:
      a_val = 11.0
      a = constant_op.constant(a_val)

      res = sess.run([[], tuple(), {}])
      self.assertTrue(isinstance(res, list))
      self.assertEquals(3, len(res))
      self.assertTrue(isinstance(res[0], list))
      self.assertEqual(0, len(res[0]))
      self.assertTrue(isinstance(res[1], tuple))
      self.assertEqual(0, len(res[1]))
      self.assertTrue(isinstance(res[2], dict))
      self.assertEqual(0, len(res[2]))

      res = sess.run([[], tuple(), {}, a])
      self.assertTrue(isinstance(res, list))
      self.assertEquals(4, len(res))
      self.assertTrue(isinstance(res[0], list))
      self.assertEqual(0, len(res[0]))
      self.assertTrue(isinstance(res[1], tuple))
      self.assertEqual(0, len(res[1]))
      self.assertTrue(isinstance(res[2], dict))
      self.assertEqual(0, len(res[2]))
      self.assertEqual(a_val, res[3])

  def testFetchNestingOneLevel(self):
    with session.Session() as sess:
      # pylint: disable=invalid-name
      ABC = collections.namedtuple('ABC', ['a', 'b', 'c'])
      DEFG = collections.namedtuple('DEFG', ['d', 'e', 'f', 'g'])
      # pylint: enable=invalid-name
      a_val = 42.0
      b_val = None
      c_val = 44.0
      a = constant_op.constant(a_val)
      b = control_flow_ops.no_op()  # An op, not a tensor.
      c = constant_op.constant(c_val)
      # List of lists, tuples, namedtuple, and  dict
      res = sess.run([[a, b, c], (a, b, c), ABC(a=a, b=b, c=c),
                      {'a': a.name, 'c': c, 'b': b}])
      self.assertTrue(isinstance(res, list))
      self.assertEqual(4, len(res))
      self.assertTrue(isinstance(res[0], list))
      self.assertEqual(3, len(res[0]))
      self.assertEqual(a_val, res[0][0])
      self.assertEqual(b_val, res[0][1])
      self.assertEqual(c_val, res[0][2])
      self.assertTrue(isinstance(res[1], tuple))
      self.assertEqual(3, len(res[1]))
      self.assertEqual(a_val, res[1][0])
      self.assertEqual(b_val, res[1][1])
      self.assertEqual(c_val, res[1][2])
      self.assertTrue(isinstance(res[2], ABC))
      self.assertEqual(a_val, res[2].a)
      self.assertEqual(b_val, res[2].b)
      self.assertEqual(c_val, res[2].c)
      self.assertTrue(isinstance(res[3], dict))
      self.assertEqual(3, len(res[3]))
      self.assertEqual(a_val, res[3]['a'])
      self.assertEqual(b_val, res[3]['b'])
      self.assertEqual(c_val, res[3]['c'])
      # Tuple of lists, tuples, namedtuple, and  dict
      res = sess.run(([a, b, c], (a.name, b, c), ABC(a=a, b=b, c=c),
                      {'a': a, 'c': c, 'b': b}))
      self.assertTrue(isinstance(res, tuple))
      self.assertEqual(4, len(res))
      self.assertTrue(isinstance(res[0], list))
      self.assertEqual(3, len(res[0]))
      self.assertEqual(a_val, res[0][0])
      self.assertEqual(b_val, res[0][1])
      self.assertEqual(c_val, res[0][2])
      self.assertTrue(isinstance(res[1], tuple))
      self.assertEqual(3, len(res[1]))
      self.assertEqual(a_val, res[1][0])
      self.assertEqual(b_val, res[1][1])
      self.assertEqual(c_val, res[1][2])
      self.assertTrue(isinstance(res[2], ABC))
      self.assertEqual(a_val, res[2].a)
      self.assertEqual(b_val, res[2].b)
      self.assertEqual(c_val, res[2].c)
      self.assertTrue(isinstance(res[3], dict))
      self.assertEqual(3, len(res[3]))
      self.assertEqual(a_val, res[3]['a'])
      self.assertEqual(b_val, res[3]['b'])
      self.assertEqual(c_val, res[3]['c'])
      # Namedtuple of lists, tuples, namedtuples, and dict
      res = sess.run(DEFG(d=[a, b, c],
                          e=(a, b, c),
                          f=ABC(a=a.name, b=b, c=c),
                          g={'a': a, 'c': c, 'b': b}))
      self.assertTrue(isinstance(res, DEFG))
      self.assertTrue(isinstance(res.d, list))
      self.assertEqual(3, len(res.d))
      self.assertEqual(a_val, res.d[0])
      self.assertEqual(b_val, res.d[1])
      self.assertEqual(c_val, res.d[2])
      self.assertTrue(isinstance(res.e, tuple))
      self.assertEqual(3, len(res.e))
      self.assertEqual(a_val, res.e[0])
      self.assertEqual(b_val, res.e[1])
      self.assertEqual(c_val, res.e[2])
      self.assertTrue(isinstance(res.f, ABC))
      self.assertEqual(a_val, res.f.a)
      self.assertEqual(b_val, res.f.b)
      self.assertEqual(c_val, res.f.c)
      self.assertTrue(isinstance(res.g, dict))
      self.assertEqual(3, len(res.g))
      self.assertEqual(a_val, res.g['a'])
      self.assertEqual(b_val, res.g['b'])
      self.assertEqual(c_val, res.g['c'])
      # Dict of lists, tuples, namedtuples, and dict
      res = sess.run({'d': [a, b, c],
                      'e': (a, b, c),
                      'f': ABC(a=a, b=b, c=c),
                      'g': {'a': a.name, 'c': c, 'b': b}})
      self.assertTrue(isinstance(res, dict))
      self.assertEqual(4, len(res))
      self.assertTrue(isinstance(res['d'], list))
      self.assertEqual(3, len(res['d']))
      self.assertEqual(a_val, res['d'][0])
      self.assertEqual(b_val, res['d'][1])
      self.assertEqual(c_val, res['d'][2])
      self.assertTrue(isinstance(res['e'], tuple))
      self.assertEqual(3, len(res['e']))
      self.assertEqual(a_val, res['e'][0])
      self.assertEqual(b_val, res['e'][1])
      self.assertEqual(c_val, res['e'][2])
      self.assertTrue(isinstance(res['f'], ABC))
      self.assertEqual(a_val, res['f'].a)
      self.assertEqual(b_val, res['f'].b)
      self.assertEqual(c_val, res['f'].c)
      self.assertTrue(isinstance(res['g'], dict))
      self.assertEqual(3, len(res['g']))
      self.assertEqual(a_val, res['g']['a'])
      self.assertEqual(b_val, res['g']['b'])
      self.assertEqual(c_val, res['g']['c'])

  def testFetchTensorObject(self):
    with session.Session() as s:
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[2, 3])
      c = math_ops.matmul(a, b)
      results_with_list = s.run([c])
      self.assertAllEqual([[4.0, 4.0, 4.0]], results_with_list[0])
      results_with_single = s.run(c)
      self.assertAllEqual([[4.0, 4.0, 4.0]], results_with_single)
      results_with_get = c.eval()
      self.assertAllEqual([[4.0, 4.0, 4.0]], results_with_get)
      a_val, b_val = s.run([a, b])  # Test multiple fetches.
      self.assertAllEqual([[1.0, 1.0]], a_val)
      self.assertAllEqual([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], b_val)
      results_with_dict = s.run({'a': [a], 'b': b, 'z': [a, b]})
      self.assertAllEqual([[1.0, 1.0]], results_with_dict['a'][0])
      self.assertAllEqual([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                          results_with_dict['b'])
      self.assertAllEqual(results_with_dict['a'][0], results_with_dict['z'][0])
      self.assertAllEqual(results_with_dict['b'], results_with_dict['z'][1])

      # Test nested structures
      results_with_nested_list = s.run([[[a, b], b], a, [a, b]])
      self.assertAllEqual([[1.0, 1.0]], results_with_nested_list[0][0][0])
      self.assertAllEqual([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                          results_with_nested_list[0][0][1])
      self.assertAllEqual(results_with_nested_list[0][0][0],
                          results_with_nested_list[1])
      self.assertAllEqual(results_with_nested_list[1],
                          results_with_nested_list[2][0])
      self.assertAllEqual(results_with_nested_list[0][0][1],
                          results_with_nested_list[0][1])
      self.assertAllEqual(results_with_nested_list[0][1],
                          results_with_nested_list[2][1])

  def testFetchScalar(self):
    with session.Session() as s:
      for scalar in np.int32, np.int64, np.float16, np.float32, np.float64:
        x = scalar(7)
        y = scalar(8)
        tf_x = constant_op.constant(x, shape=[])
        tf_y = constant_op.constant(y)
        tf_xy = math_ops.add(tf_x, tf_y)
        # Single fetch
        xy = s.run(tf_xy)
        self.assertEqual(scalar, type(xy))
        self.assertEqual(x + y, xy)
        # List fetch
        xy, = s.run([tf_xy])
        self.assertEqual(scalar, type(xy))
        self.assertEqual(x + y, xy)
        # Dict fetch
        xy = s.run({'xy': tf_xy})['xy']
        self.assertEqual(scalar, type(xy))
        self.assertEqual(x + y, xy)
        # Nested list fetch
        xy = s.run([[[tf_xy]], tf_xy, [tf_xy]])
        self.assertAllEqual(xy, [[[x + y]], x + y, [x + y]])
        self.assertEqual(scalar, type(xy[0][0][0]))
        self.assertEqual(scalar, type(xy[1]))
        self.assertEqual(scalar, type(xy[2][0]))

  def testFetchOperationObject(self):
    with session.Session() as s:
      a = constant_op.constant(1.0, shape=[1, 2])
      v = variables.Variable(a, name='testFetchOperationObject_v')
      s.run(v.initializer)
      v_val = s.run(v)
      self.assertAllEqual([[1.0, 1.0]], v_val)

  def testFetchSparseTensor(self):
    with session.Session() as s:
      indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
      values = np.array([1.0, 2.0]).astype(np.float32)
      shape = np.array([7, 9, 2]).astype(np.int64)
      sp = sparse_tensor.SparseTensor(
          constant_op.constant(indices),
          constant_op.constant(values),
          constant_op.constant(shape))
      # Single fetch, use as tuple
      sp_out = s.run(sp)
      indices_out, values_out, shape_out = sp_out
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # Single fetch, use as SparseTensorValue
      sp_out = s.run(sp)
      self.assertAllEqual(sp_out.indices, indices)
      self.assertAllEqual(sp_out.values, values)
      self.assertAllEqual(sp_out.dense_shape, shape)
      # Tuple fetch, use as tuple
      indices_out, values_out, shape_out = s.run(sp)
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # List fetch, use as tuple
      (indices_out, values_out, shape_out), = s.run([sp])
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # List fetch, use as SparseTensorValue
      sp_out, = s.run([sp])
      self.assertAllEqual(sp_out.indices, indices)
      self.assertAllEqual(sp_out.values, values)
      self.assertAllEqual(sp_out.dense_shape, shape)
      # Dict fetch (single value), use as tuple
      indices_out, values_out, shape_out = s.run({'sp': sp})['sp']
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # Dict fetch (list value), use as tuple
      (indices_out, values_out, shape_out), = s.run({'sp': [sp]})['sp']
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # Dict fetch, use as SparseTensorValue
      sp_out = s.run({'sp': sp})['sp']
      self.assertAllEqual(sp_out.indices, indices)
      self.assertAllEqual(sp_out.values, values)
      self.assertAllEqual(sp_out.dense_shape, shape)
      # Nested list fetch use as tuple
      sp_out = s.run([[[sp]], sp])
      indices_out, values_out, shape_out = sp_out[0][0][0]
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      indices_out, values_out, shape_out = sp_out[1]
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # Nested list fetch, use as SparseTensorValue
      sp_out = s.run([[[sp]], sp])
      self.assertAllEqual(sp_out[0][0][0].indices, indices)
      self.assertAllEqual(sp_out[0][0][0].values, values)
      self.assertAllEqual(sp_out[0][0][0].dense_shape, shape)
      self.assertAllEqual(sp_out[1].indices, indices)
      self.assertAllEqual(sp_out[1].values, values)
      self.assertAllEqual(sp_out[1].dense_shape, shape)

  def testFeedSparseTensor(self):
    with session.Session() as s:
      indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
      values = np.array([1.0, 2.0]).astype(np.float32)
      shape = np.array([7, 9, 2]).astype(np.int64)
      sp = sparse_tensor.SparseTensor(
          array_ops.placeholder(dtype=np.int64, shape=(2, 3)),
          array_ops.placeholder(dtype=np.float32, shape=(2,)),
          array_ops.placeholder(dtype=np.int64, shape=(3,)),)
      sp_indices = array_ops.identity(sp.indices)
      sp_values = array_ops.identity(sp.values)
      sp_shape = array_ops.identity(sp.dense_shape)
      sp2 = sparse_tensor.SparseTensor(sp_indices, sp_values, sp_shape)
      # Feed with tuple
      indices_out, values_out, shape_out = s.run(
          [sp_indices, sp_values, sp_shape], {sp: (indices, values, shape)})
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # Feed with tuple, fetch sp directly
      sp_out = s.run(sp, {sp: (indices, values, shape)})
      self.assertAllEqual(sp_out.indices, indices)
      self.assertAllEqual(sp_out.values, values)
      self.assertAllEqual(sp_out.dense_shape, shape)
      # Feed with SparseTensorValue
      indices_out, values_out, shape_out = s.run(
          [sp_indices, sp_values, sp_shape],
          {sp: sparse_tensor.SparseTensorValue(indices, values, shape)})
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # Feed with SparseTensorValue, fetch SparseTensorValue
      sp2_out = s.run(
          sp2, {sp: sparse_tensor.SparseTensorValue(indices, values, shape)})
      self.assertAllEqual(sp2_out.indices, indices)
      self.assertAllEqual(sp2_out.values, values)
      self.assertAllEqual(sp2_out.dense_shape, shape)
      # Feed SparseTensorValue and fetch sp directly.
      sp_out = s.run(
          sp, {sp: sparse_tensor.SparseTensorValue(indices, values, shape)})
      self.assertAllEqual(sp_out.indices, indices)
      self.assertAllEqual(sp_out.values, values)
      self.assertAllEqual(sp_out.dense_shape, shape)

  def testFeedSparsePlaceholder(self):
    with session.Session() as s:
      indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
      values = np.array([1.0, 2.0]).astype(np.float32)
      shape = np.array([7, 9, 2]).astype(np.int64)
      sp = array_ops.sparse_placeholder(dtype=np.float32, name='placeholder1')
      sp_indices = array_ops.identity(sp.indices)
      sp_values = array_ops.identity(sp.values)
      sp_shape = array_ops.identity(sp.dense_shape)
      sp2 = sparse_tensor.SparseTensor(sp_indices, sp_values, sp_shape)
      # Feed with tuple
      indices_out, values_out, shape_out = s.run(
          [sp_indices, sp_values, sp_shape], {sp: (indices, values, shape)})
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # Feed with SparseTensorValue
      indices_out, values_out, shape_out = s.run(
          [sp_indices, sp_values, sp_shape],
          {sp: sparse_tensor.SparseTensorValue(indices, values, shape)})
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # Feed with SparseTensorValue, fetch SparseTensorValue
      sp2_out = s.run(
          sp2, {sp: sparse_tensor.SparseTensorValue(indices, values, shape)})
      self.assertAllEqual(sp2_out.indices, indices)
      self.assertAllEqual(sp2_out.values, values)
      self.assertAllEqual(sp2_out.dense_shape, shape)

  def testFeedSparsePlaceholderPartialShape(self):
    with session.Session() as s:
      indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
      values = np.array([1.0, 2.0]).astype(np.float32)
      shape = np.array([7, 9, 2]).astype(np.int64)
      sp = array_ops.sparse_placeholder(
          shape=[None, 9, 2], dtype=np.float32, name='placeholder1')
      sp_indices = array_ops.identity(sp.indices)
      sp_values = array_ops.identity(sp.values)
      sp_shape = array_ops.identity(sp.dense_shape)
      sp2 = sparse_tensor.SparseTensor(sp_indices, sp_values, sp_shape)
      # Feed with tuple
      indices_out, values_out, shape_out = s.run(
          [sp_indices, sp_values, sp_shape], {sp: (indices, values, shape)})
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # Feed with SparseTensorValue
      indices_out, values_out, shape_out = s.run(
          [sp_indices, sp_values, sp_shape],
          {sp: sparse_tensor.SparseTensorValue(indices, values, shape)})
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # Feed with SparseTensorValue, fetch SparseTensorValue
      sp2_out = s.run(
          sp2, {sp: sparse_tensor.SparseTensorValue(indices, values, shape)})
      self.assertAllEqual(sp2_out.indices, indices)
      self.assertAllEqual(sp2_out.values, values)
      self.assertAllEqual(sp2_out.dense_shape, shape)

  def testFeedSparsePlaceholderConstantShape(self):
    with session.Session() as s:
      indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
      values = np.array([1.0, 2.0]).astype(np.float32)
      shape = np.array([7, 9, 2]).astype(np.int64)
      sp = array_ops.sparse_placeholder(dtype=np.float32,
                                        shape=shape,
                                        name='placeholder1')
      self.assertAllEqual(sp.dense_shape.eval(session=s), shape)
      self.assertAllEqual(tensor_util.constant_value(sp.dense_shape), shape)
      sp_indices = array_ops.identity(sp.indices)
      sp_values = array_ops.identity(sp.values)
      sp_shape = array_ops.identity(sp.dense_shape)
      # Feed with tuple
      indices_out, values_out, shape_out = s.run(
          [sp_indices, sp_values, sp_shape], {sp: (indices, values)})
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)

  def testFetchIndexedSlices(self):
    with session.Session() as s:
      indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
      values = np.array([1.0, 2.0]).astype(np.float32)
      dense_shape = np.array([7, 9, 2]).astype(np.int64)
      ind = ops.IndexedSlices(
          constant_op.constant(values), constant_op.constant(indices),
          constant_op.constant(dense_shape))
      # Single fetch, use as tuple
      ind_out = s.run(ind)
      values_out, indices_out, dense_shape_out = ind_out
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(dense_shape_out, dense_shape)
      # Single fetch, use as IndexedSlicesValue
      ind_out = s.run(ind)
      self.assertAllEqual(ind_out.values, values)
      self.assertAllEqual(ind_out.indices, indices)
      self.assertAllEqual(ind_out.dense_shape, dense_shape)
      # Tuple fetch, use as tuple
      values_out, indices_out, dense_shape_out = s.run(ind)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(dense_shape_out, dense_shape)
      # List fetch, use as tuple
      (values_out, indices_out, dense_shape_out), = s.run([ind])
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(dense_shape_out, dense_shape)
      # List fetch, use as IndexedSlicesValue
      ind_out, = s.run([ind])
      self.assertAllEqual(ind_out.values, values)
      self.assertAllEqual(ind_out.indices, indices)
      self.assertAllEqual(ind_out.dense_shape, dense_shape)

  def testFeedIndexedSlices(self):
    with session.Session() as s:
      values = np.array([1.0, 2.0]).astype(np.float32)
      indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
      dense_shape = np.array([7, 9, 2]).astype(np.int64)
      ind = ops.IndexedSlices(
          array_ops.placeholder(dtype=np.float32,
                                shape=(2,)),
          array_ops.placeholder(dtype=np.int64,
                                shape=(2, 3)),
          array_ops.placeholder(dtype=np.int64,
                                shape=(3,)),)
      ind_values = array_ops.identity(ind.values)
      ind_indices = array_ops.identity(ind.indices)
      ind_dense_shape = array_ops.identity(ind.dense_shape)
      ind2 = ops.IndexedSlices(ind_values, ind_indices, ind_dense_shape)
      # Feed with tuple
      values_out, indices_out, dense_shape_out = s.run(
          [ind_values, ind_indices, ind_dense_shape],
          {ind: (values, indices, dense_shape)})
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(dense_shape_out, dense_shape)
      # Feed with IndexedSlicesValue
      values_out, indices_out, dense_shape_out = s.run(
          [ind_values, ind_indices, ind_dense_shape],
          {ind: ops.IndexedSlicesValue(values, indices, dense_shape)})
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(dense_shape_out, dense_shape)
      # Feed with IndexedSlicesValue, fetch IndexedSlicesValue
      ind2_out = s.run(ind2, {ind: ops.IndexedSlicesValue(values, indices,
                                                          dense_shape)})
      self.assertAllEqual(ind2_out.values, values)
      self.assertAllEqual(ind2_out.indices, indices)
      self.assertAllEqual(ind2_out.dense_shape, dense_shape)

  def testFetchIndexedSlicesWithoutDenseShape(self):
    with session.Session() as s:
      indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
      values = np.array([1.0, 2.0]).astype(np.float32)
      dense_shape = None
      ind = ops.IndexedSlices(
          constant_op.constant(values), constant_op.constant(indices), None)
      # Single fetch, use as tuple
      ind_out = s.run(ind)
      values_out, indices_out, dense_shape_out = ind_out
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(dense_shape_out, dense_shape)
      # Single fetch, use as IndexedSlicesValue
      ind_out = s.run(ind)
      self.assertAllEqual(ind_out.values, values)
      self.assertAllEqual(ind_out.indices, indices)
      self.assertAllEqual(ind_out.dense_shape, dense_shape)
      # Tuple fetch, use as tuple
      values_out, indices_out, dense_shape_out = s.run(ind)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(dense_shape_out, dense_shape)
      # List fetch, use as tuple
      (values_out, indices_out, dense_shape_out), = s.run([ind])
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(dense_shape_out, dense_shape)
      # List fetch, use as IndexedSlicesValue
      ind_out, = s.run([ind])
      self.assertAllEqual(ind_out.values, values)
      self.assertAllEqual(ind_out.indices, indices)
      self.assertAllEqual(ind_out.dense_shape, dense_shape)

  def testFeedIndexedSlicesWithoutDenseShape(self):
    with session.Session() as s:
      values = np.array([1.0, 2.0]).astype(np.float32)
      indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
      dense_shape = None
      ind = ops.IndexedSlices(
          array_ops.placeholder(dtype=np.float32,
                                shape=(2,)),
          array_ops.placeholder(dtype=np.int64,
                                shape=(2, 3)),
          None)
      ind_values = array_ops.identity(ind.values)
      ind_indices = array_ops.identity(ind.indices)
      ind2 = ops.IndexedSlices(ind_values, ind_indices)
      # Feed with tuple
      values_out, indices_out = s.run(
          [ind_values, ind_indices], {ind: (values, indices)})
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(indices_out, indices)
      # Feed with IndexedSlicesValue
      values_out, indices_out = s.run(
          [ind_values, ind_indices],
          {ind: ops.IndexedSlicesValue(values, indices, dense_shape)})
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(indices_out, indices)
      # Feed with IndexedSlicesValue, fetch IndexedSlicesValue
      ind2_out = s.run(ind2, {ind: ops.IndexedSlicesValue(values, indices,
                                                          dense_shape)})
      self.assertAllEqual(ind2_out.values, values)
      self.assertAllEqual(ind2_out.indices, indices)
      self.assertAllEqual(ind2_out.dense_shape, dense_shape)

  def testExtendWithStatelessOperations(self):
    with session.Session() as s:
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[2, 3])
      c = math_ops.matmul(a, b)
      c_val = s.run(c)
      self.assertAllEqual([[4.0, 4.0, 4.0]], c_val)
      d = constant_op.constant([1.0, 2.0, 3.0], shape=[3, 1])
      e = math_ops.matmul(c, d)
      # Extend will happen here.
      e_val = s.run(e)
      self.assertAllEqual([[24.0]], e_val)

  def testExtendWithStatefulOperations(self):
    with session.Session() as s:
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[2, 3])
      c = math_ops.matmul(a, b)
      v = variables.Variable(c, name='testExtendWithStatefulOperations_v')
      v.initializer.run()
      v_val = v.eval()
      self.assertAllEqual([[4.0, 4.0, 4.0]], v_val)
      d = constant_op.constant(3.0, shape=[2, 3])
      e = math_ops.matmul(a, d)
      assign_e_to_v = state_ops.assign(v, e)
      # Extend will happen here.
      e_val = e.eval()
      self.assertAllEqual([[6.0, 6.0, 6.0]], e_val)
      v_val = v.eval()
      self.assertAllEqual([[4.0, 4.0, 4.0]], v_val)
      s.run(assign_e_to_v)
      v_val = v.eval()
      self.assertAllEqual([[6.0, 6.0, 6.0]], v_val)

  def testExtendWithGroupBy(self):
    with session.Session() as s:
      a = constant_op.constant(1.0, shape=[1, 2])
      p = variables.Variable(a, name='testExtendWithGroupBy_p')
      a_val = a.eval()  # Force an Extend after this op.
      self.assertAllEqual([[1.0, 1.0]], a_val)

      b = constant_op.constant(2.0, shape=[1, 2])
      q = variables.Variable(b, name='testExtendWithGroupBy_q')
      # Extend will happen here.
      init = control_flow_ops.group(p.initializer, q.initializer)
      s.run(init)
      p_val, q_val = s.run([p, q])

      self.assertAllEqual([[1.0, 1.0]], p_val)
      self.assertAllEqual([[2.0, 2.0]], q_val)

  def testTensorGetMethod(self):
    with session.Session():
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[2, 3])
      c = math_ops.matmul(a, b)

      c_val = c.eval()
      self.assertAllEqual([[4.0, 4.0, 4.0]], c_val)

      fed_c_val = c.eval(feed_dict={a.name: [[4.0, 4.0]]})
      self.assertAllEqual([[16.0, 16.0, 16.0]], fed_c_val)

  def testOperationRunMethod(self):
    with session.Session():
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[1, 2], name='b')
      v = variables.Variable(a, a.dtype)
      assign_a_to_v = state_ops.assign(v, a)

      assign_a_to_v.eval()

      v_val = v.eval()
      self.assertAllEqual([[1.0, 1.0]], v_val)

      assign_b_to_v = state_ops.assign(v, b)

      assign_b_to_v.eval()
      v_val = v.eval()
      self.assertAllEqual([[2.0, 2.0]], v_val)

      assign_b_to_v.eval(feed_dict={'b:0': [[3.0, 3.0]]})
      v_val = v.eval()
      self.assertAllEqual([[3.0, 3.0]], v_val)

  def testDefaultGraph(self):
    with session.Session() as s:
      self.assertEqual(ops.get_default_graph(), s.graph)
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[2, 3])
      self.assertEqual(ops.get_default_graph(), a.graph)
      self.assertEqual(ops.get_default_graph(), b.graph)
      c = math_ops.matmul(a, b)
      v = variables.Variable(c, name='testDefaultGraph_v')
      v.initializer.run()
      v_val = v.eval()
      self.assertAllEqual([[4.0, 4.0, 4.0]], v_val)
      d = constant_op.constant(3.0, shape=[2, 3])
      e = math_ops.matmul(a, d)
      assign_e_to_v = state_ops.assign(v, e)
      e_val = e.eval()
      self.assertAllEqual([[6.0, 6.0, 6.0]], e_val)
      v_val = v.eval()
      self.assertAllEqual([[4.0, 4.0, 4.0]], v_val)
      s.run(assign_e_to_v)
      v_val = v.eval()
      self.assertAllEqual([[6.0, 6.0, 6.0]], v_val)
      self.assertEqual(ops.get_default_graph(), s.graph)

  def _testDefaultGraphInThread(self, constructed_event, continue_event, i):
    with session.Session() as s:
      self.assertEqual(ops.get_default_graph(), s.graph)
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[2, 3])
      c = math_ops.matmul(a, b)
      v = variables.Variable(c, name='var_%d' % i)

      # Block here until all threads have constructed their graph.
      constructed_event.set()
      continue_event.wait()

      assign_c_to_v = state_ops.assign(v, c)
      v.initializer.run()
      assign_c_to_v.eval()
      v_val = v.eval()
      self.assertAllEqual([[4.0, 4.0, 4.0]], v_val)
      d = constant_op.constant(3.0, shape=[2, 3])
      e = math_ops.matmul(a, d)
      assign_e_to_v = state_ops.assign(v, e)
      e_val = e.eval()
      self.assertAllEqual([[6.0, 6.0, 6.0]], e_val)
      v_val = v.eval()
      self.assertAllEqual([[4.0, 4.0, 4.0]], v_val)
      s.run(assign_e_to_v)
      v_val = v.eval()
      self.assertAllEqual([[6.0, 6.0, 6.0]], v_val)
      self.assertEqual(ops.get_default_graph(), s.graph)

  def testDefaultGraphWithThreads(self):
    # Fork ten threads that use their thread-local default graph.
    threads = []
    constructed_events = [threading.Event() for _ in range(10)]
    continue_event = threading.Event()
    for i, constructed_event in enumerate(constructed_events):
      t = self.checkedThread(target=self._testDefaultGraphInThread,
                             args=(constructed_event, continue_event, i))
      threads.append(t)
    for t in threads:
      t.start()
    for constructed_event in constructed_events:
      constructed_event.wait()
    continue_event.set()
    for t in threads:
      t.join()

  def testParallelRun(self):
    with session.Session() as sess:
      c = constant_op.constant(5.0)
      ev = threading.Event()

      def run_step():
        ev.wait()
        val = c.eval(session=sess)
        self.assertEqual(val, 5.0)
      threads = [self.checkedThread(target=run_step) for _ in range(100)]
      for t in threads:
        t.start()
      ev.set()
      for t in threads:
        t.join()

  def testRunFeedDict(self):
    with session.Session() as s:
      x = array_ops.zeros([2])

      y = s.run(2 * x, feed_dict={x: np.ones(2).astype(np.float32)})
      self.assertAllEqual(y, 2 * np.ones(2))

      y = s.run(2 * x, feed_dict={x.name: np.ones(2).astype(np.float32)})
      self.assertAllEqual(y, 2 * np.ones(2))

      y = s.run(2 * x, feed_dict={x: [1, 1]})
      assert (y == 2 * np.ones(2)).all()

      # Test nested tuple keys
      z = (((array_ops.zeros([2]),),), array_ops.zeros([2]),
           (array_ops.zeros([2]),))
      result = [z[0][0][0] * 2, z[1] * 2, z[2][0] * 2]
      values = (((np.array([1, 1]),),), np.array([2, 2]), (np.array([3, 3]),))
      result_value = s.run(result, feed_dict={z: values})
      self.assertAllEqual(result_value[0], 2 * np.ones(2))
      self.assertAllEqual(result_value[1], 2 * np.array([2, 2]))
      self.assertAllEqual(result_value[2], 2 * np.array([3, 3]))

  def testGraphDef(self):
    with session.Session() as sess:
      self.assertProtoEquals(
          'versions { producer: %d min_consumer: %d }' % (
              versions.GRAPH_DEF_VERSION,
              versions.GRAPH_DEF_VERSION_MIN_CONSUMER),
          sess.graph_def)
      c = constant_op.constant(5.0, name='c')
      self.assertEquals(len(sess.graph_def.node), 1)
      d = constant_op.constant(6.0, name='d')
      self.assertEquals(len(sess.graph_def.node), 2)
      self.assertAllEqual(c.eval(), 5.0)
      self.assertAllEqual(d.eval(), 6.0)
      e = constant_op.constant(7.0, name='e')
      self.assertEquals(len(sess.graph_def.node), 3)
      self.assertAllEqual(e.eval(), 7.0)

  def testUseAfterClose(self):
    with session.Session() as sess:
      c = constant_op.constant(5.0)
      self.assertAllEqual(sess.run(c), 5.0)
    with self.assertRaisesWithPredicateMatch(
        RuntimeError, lambda e: 'Attempted to use a closed Session.' in str(e)):
      sess.run(c)

  def testUseAfterCloseConcurrent(self):
    with session.Session() as sess:
      c = constant_op.constant(5.0)
      self.assertAllEqual(sess.run(c), 5.0)

      def update_thread():
        with self.assertRaisesWithPredicateMatch(
            RuntimeError,
            lambda e: 'Attempted to use a closed Session.' in str(e)):
          while True:
            sess.run(c)
      t = threading.Thread(target=update_thread)
      t.start()
      time.sleep(0.1)
      sess.close()
      t.join()

  def testUseEmptyGraph(self):
    with session.Session() as sess:
      with self.assertRaisesRegexp(RuntimeError, 'The Session graph is empty.'):
        sess.run([])
      with self.assertRaisesRegexp(RuntimeError, 'The Session graph is empty.'):
        sess.run(())
      with self.assertRaisesRegexp(RuntimeError, 'The Session graph is empty.'):
        sess.run({})

  def testNotEntered(self):
    # pylint: disable=protected-access
    self.assertEqual(ops._default_session_stack.get_default(), None)
    # pylint: enable=protected-access
    with ops.device('/cpu:0'):
      sess = session.Session()
      c_1 = constant_op.constant(5.0)
      with sess.graph.as_default():
        c_2 = constant_op.constant(5.0)
      self.assertEqual(c_1.graph, c_2.graph)
      self.assertEqual(sess.run(c_2), 5.0)
      with self.assertRaisesWithPredicateMatch(
          ValueError, lambda e: 'No default session is registered.' in str(e)):
        c_2.eval()

  def testInteractive(self):
    with ops.device('/cpu:0'):
      sess = session.InteractiveSession()
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[2, 3])
      c = math_ops.matmul(a, b)
      self.assertAllEqual([[4.0, 4.0, 4.0]], c.eval())
      d = constant_op.constant([1.0, 2.0, 3.0], shape=[3, 1])
      e = math_ops.matmul(c, d)
      self.assertAllEqual([[24.0]], e.eval())
      sess.close()

  def testInteractivePlacePrunedGraph(self):
    sess = session.InteractiveSession()

    # Build a graph that has a bad op in it (no kernel).
    #
    # This test currently does not link in any GPU kernels,
    # which is why placing this is invalid.  If at some point
    # GPU kernels are added to this test, some other different
    # op / device combo should be chosen.
    with ops.device('/gpu:0'):
      a = constant_op.constant(1.0, shape=[1, 2])

    b = constant_op.constant(1.0, shape=[1, 2])

    # Only run the valid op, this should work.
    b.eval()

    with self.assertRaises(errors.InvalidArgumentError):
      a.eval()
    sess.close()

  def testDefaultSessionPlacePrunedGraph(self):
    sess = session.Session()

    # Build a graph that has a bad op in it (no kernel).
    #
    # This test currently does not link in any GPU kernels,
    # which is why placing this is invalid.  If at some point
    # GPU kernels are added to this test, some other different
    # op / device combo should be chosen.
    with ops.device('/gpu:0'):
      _ = constant_op.constant(1.0, shape=[1, 2])

    b = constant_op.constant(1.0, shape=[1, 2])

    with self.assertRaises(errors.InvalidArgumentError):
      # Even though we don't run the bad op, we place the entire
      # graph, which should fail with a non-interactive session.
      sess.run(b)

    sess.close()

  def testSharedGraph(self):
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[2, 3])
      c = math_ops.matmul(a, b)

    with session.Session(graph=g) as sess1:
      with session.Session(graph=g) as sess2:
        self.assertAllEqual(sess1.run(c), sess2.run(c))

  def testDuplicatedInputs(self):
    with session.Session() as sess:
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[1, 3])
      a_val, b_val, a2_val = sess.run([a, b, a])
      self.assertAllEqual(a_val, [[1.0, 1.0]])
      self.assertAllEqual(b_val, [[2.0, 2.0, 2.0]])
      self.assertAllEqual(a2_val, [[1.0, 1.0]])

  def testFeedAndFetch(self):
    with session.Session() as sess:
      for dtype in [dtypes.float16,
                    dtypes.float32,
                    dtypes.float64,
                    dtypes.int32,
                    dtypes.uint8,
                    dtypes.int16,
                    dtypes.int8,
                    dtypes.int64,
                    dtypes.bool,
                    dtypes.complex64,
                    dtypes.complex128]:
        for shape in [(32, 4, 128), (37,), (2, 0, 6), (0, 0, 0)]:
          np_dtype = dtype.as_numpy_dtype

          feed_t = array_ops.placeholder(dtype=dtype, shape=shape)
          out_t = array_ops.identity(feed_t)

          np_array = np.random.randint(-10, 10, shape)

          if dtype == dtypes.bool:
            np_array = np_array > 0
          elif dtype == dtypes.complex64:
            np_array = np.sqrt(np_array.astype(np_dtype))
          elif dtype == dtypes.complex64:
            np_array = np.sqrt(np_array.astype(np_dtype))
          else:
            np_array = np_array.astype(np_dtype)

          self.assertAllEqual(np_array,
                              sess.run(out_t, feed_dict={feed_t: np_array}))
          # Check that we can also get the feed back.
          self.assertAllEqual(np_array,
                              sess.run(feed_t, feed_dict={feed_t: np_array}))
          # Also check that we can get both back.
          out_v, feed_v = sess.run([out_t, feed_t],
                                   feed_dict={feed_t: np_array})
          self.assertAllEqual(np_array, out_v)
          self.assertAllEqual(np_array, feed_v)

          feed_fetch_runner = sess.make_callable([out_t, feed_t], [feed_t])
          out_v, feed_v = feed_fetch_runner(np_array)
          self.assertAllEqual(np_array, out_v)
          self.assertAllEqual(np_array, feed_v)

  def testFeedError(self):
    with session.Session() as sess:
      feed_t = array_ops.placeholder(dtype=dtypes.float32)
      out_t = array_ops.identity(feed_t)
      feed_val = constant_op.constant(5.0)
      with self.assertRaisesRegexp(TypeError, 'cannot be a tf.Tensor object'):
        sess.run(out_t, feed_dict={feed_t: feed_val})
      with self.assertRaisesRegexp(TypeError, 'cannot be a tf.Tensor object'):
        out_t.eval(feed_dict={feed_t: feed_val})
      with self.assertRaisesRegexp(TypeError, 'cannot be a tf.Tensor object'):
        out_t.op.run(feed_dict={feed_t: feed_val})

  def testFeedPrecisionLossError(self):
    with session.Session() as sess:
      largest_int64 = np.iinfo(np.int64).max

      feed_int_implicit_int32 = constant_op.constant(1)
      feed_int_explicit_int32 = constant_op.constant(1, dtype=dtypes.int32)

      out_t = constant_op.constant(1.0)

      with self.assertRaisesRegexp(TypeError,
                                   'is not compatible with Tensor type'):
        sess.run(out_t, feed_dict={feed_int_implicit_int32: largest_int64})
      with self.assertRaisesRegexp(TypeError,
                                   'is not compatible with Tensor type'):
        sess.run(out_t, feed_dict={feed_int_explicit_int32: largest_int64})

  def testStringFetch(self):
    with session.Session():
      for shape in [(32, 4, 128), (37,), (2, 0, 6), (0, 0, 0)]:
        size = 1
        for s in shape:
          size *= s
        c_list = np.array([compat.as_bytes(str(i)) for i in xrange(size)],
                          dtype=np.object).reshape(shape) if size > 0 else []
        c = constant_op.constant(c_list)
        self.assertAllEqual(c.eval(), c_list)

  def testStringFeed(self):
    with session.Session() as sess:
      for shape in [(32, 4, 128), (37,), (2, 0, 6), (0, 0, 0)]:
        size = 1
        for s in shape:
          size *= s
        c_list = np.array([compat.as_bytes(str(i)) for i in xrange(size)],
                          dtype=np.object).reshape(shape)
        feed_t = array_ops.placeholder(dtype=dtypes.string, shape=shape)
        c = array_ops.identity(feed_t)
        self.assertAllEqual(sess.run(c, feed_dict={feed_t: c_list}), c_list)
        self.assertAllEqual(sess.run(feed_t, feed_dict={feed_t: c_list}),
                            c_list)
        c_v, feed_v = sess.run([c, feed_t], feed_dict={feed_t: c_list})
        self.assertAllEqual(c_v, c_list)
        self.assertAllEqual(feed_v, c_list)

  def testStringFeedWithNullCharacters(self):
    with session.Session():
      c_list = [b'\n\x01\x00', b'\n\x00\x01']
      feed_t = array_ops.placeholder(dtype=dtypes.string, shape=[2])
      c = array_ops.identity(feed_t)
      out = c.eval(feed_dict={feed_t: c_list})
      self.assertEqual(c_list[0], out[0])
      self.assertEqual(c_list[1], out[1])

  def testStringFeedWithUnicode(self):
    with session.Session():
      c_list = [u'\n\x01\x00', u'\n\x00\x01',
                u'\u26a3 unicode', u'\U0001f60e deal with it']
      feed_t = array_ops.placeholder(dtype=dtypes.string, shape=[len(c_list)])
      c = array_ops.identity(feed_t)

      out = c.eval(feed_dict={feed_t: c_list})
      for i in range(len(c_list)):
        self.assertEqual(c_list[i], out[i].decode('utf-8'))

      out = c.eval(feed_dict={feed_t: np.array(c_list, dtype=np.object)})
      for i in range(len(c_list)):
        self.assertEqual(c_list[i], out[i].decode('utf-8'))

  def testInvalidTargetFails(self):
    with self.assertRaisesRegexp(
        errors.NotFoundError,
        'No session factory registered for the given session options'):
      session.Session('INVALID_TARGET')

  def testFetchByNameDifferentStringTypes(self):
    with session.Session() as sess:
      c = constant_op.constant(42.0, name='c')
      d = constant_op.constant(43.0, name=u'd')
      e = constant_op.constant(44.0, name=b'e')
      f = constant_op.constant(45.0, name=r'f')

      self.assertTrue(isinstance(c.name, six.text_type))
      self.assertTrue(isinstance(d.name, six.text_type))
      self.assertTrue(isinstance(e.name, six.text_type))
      self.assertTrue(isinstance(f.name, six.text_type))

      self.assertEqual(42.0, sess.run('c:0'))
      self.assertEqual(42.0, sess.run(u'c:0'))
      self.assertEqual(42.0, sess.run(b'c:0'))
      self.assertEqual(42.0, sess.run(r'c:0'))

      self.assertEqual(43.0, sess.run('d:0'))
      self.assertEqual(43.0, sess.run(u'd:0'))
      self.assertEqual(43.0, sess.run(b'd:0'))
      self.assertEqual(43.0, sess.run(r'd:0'))

      self.assertEqual(44.0, sess.run('e:0'))
      self.assertEqual(44.0, sess.run(u'e:0'))
      self.assertEqual(44.0, sess.run(b'e:0'))
      self.assertEqual(44.0, sess.run(r'e:0'))

      self.assertEqual(45.0, sess.run('f:0'))
      self.assertEqual(45.0, sess.run(u'f:0'))
      self.assertEqual(45.0, sess.run(b'f:0'))
      self.assertEqual(45.0, sess.run(r'f:0'))

  def testIncorrectGraph(self):
    with ops.Graph().as_default() as g_1:
      c_1 = constant_op.constant(1.0, name='c')

    with ops.Graph().as_default() as g_2:
      c_2 = constant_op.constant(2.0, name='c')

    self.assertEqual('c', c_1.op.name)
    self.assertEqual('c', c_2.op.name)

    with session.Session(graph=g_1) as sess_1:
      self.assertEqual(1.0, sess_1.run(c_1))
      with self.assertRaises(ValueError):
        sess_1.run(c_2)
      with self.assertRaises(ValueError):
        sess_1.run(c_2.op)

    with session.Session(graph=g_2) as sess_2:
      with self.assertRaises(ValueError):
        sess_2.run(c_1)
      with self.assertRaises(ValueError):
        sess_2.run(c_1.op)
      self.assertEqual(2.0, sess_2.run(c_2))

  def runTestPartialRun(self, sess):
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)
    r2 = math_ops.multiply(r1, c)

    h = sess.partial_run_setup([r1, r2], [a, b, c])
    res = sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
    self.assertEqual(3, res)
    temp = res * 17
    res = sess.partial_run(h, r2, feed_dict={c: temp})
    self.assertEqual(153, res)

    # Call again on the same graph.
    h2 = sess.partial_run_setup([r1, r2], [a, b, c])
    res = sess.partial_run(h2, r1, feed_dict={a: 1, b: 2})
    self.assertEqual(3, res)
    temp = res * 18
    res = sess.partial_run(h2, r2, feed_dict={c: temp})
    self.assertEqual(162, res)

  def runTestPartialRunIncomplete(self, sess):
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)
    r2 = math_ops.multiply(r1, c)

    h = sess.partial_run_setup([r1, r2], [a, b, c])
    res = sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
    self.assertEqual(3, res)

  def runTestConcurrentPartialRun(self, sess):
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)
    r2 = math_ops.multiply(r1, c)

    h1 = sess.partial_run_setup([r1], [a, b, c])
    h2 = sess.partial_run_setup([r1, r2], [a, b, c])
    res = sess.partial_run(h1, r1, feed_dict={a: 1, b: 2})
    self.assertEqual(3, res)
    temp = res * 19
    res = sess.partial_run(h2, r1, feed_dict={a: temp, b: 9})
    self.assertEqual(66, res)
    res = sess.partial_run(h2, r2, feed_dict={c: 7})
    self.assertEqual(462, res)

  def runTestManyPartialRun(self, sess):
    steps = 200
    inputs = []
    outputs = []
    a = constant_op.constant(2.0, dtypes.float32)
    for i in xrange(steps):
      inputs.append(array_ops.placeholder(dtypes.float32, shape=[]))
      a = math_ops.multiply(a, inputs[i])
      outputs.append(a)

    h = sess.partial_run_setup(outputs, inputs)
    for i in xrange(steps):
      res = sess.partial_run(h, outputs[i], feed_dict={inputs[i]: 1.0})
    self.assertEqual(2.0, res)

    feed_dict = {}
    for i in xrange(steps):
      feed_dict[inputs[i]] = 1.0
    res = sess.run(outputs, feed_dict)
    self.assertEqual(steps, len(res))
    self.assertEqual(2.0, res[-1])

  def runTestRunAndPartialRun(self, sess):
    a = constant_op.constant(2.0, dtypes.float32)
    b = a * 2
    c = b * 3
    r1 = sess.run([b, c])
    h = sess.partial_run_setup([b, c], [])
    r2 = sess.partial_run(h, [b, c])
    self.assertEqual(r1, r2)

  def runTestPartialRunMissingPlaceholderFeedException(self, sess):
    x = array_ops.placeholder(dtypes.float32, shape=())
    fetches = [x * 2, x * 3]
    handle = sess.partial_run_setup(fetches=fetches, feeds=[])
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 'You must feed a value for placeholder'):
      sess.partial_run(handle, fetches[0])

  def runTestPartialRunUnspecifiedFeed(self, sess):
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)

    h = sess.partial_run_setup([r1], [a, b])
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 'was not specified in partial_run_setup.$'):
      sess.partial_run(h, r1, feed_dict={a: 1, b: 2, c: 3})

  def runTestPartialRunUnspecifiedFetch(self, sess):
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)
    r2 = math_ops.multiply(a, c)

    h = sess.partial_run_setup([r1], [a, b, c])
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 'was not specified in partial_run_setup.$'):
      sess.partial_run(h, r2, feed_dict={a: 1, c: 3})

  def runTestPartialRunAlreadyFed(self, sess):
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)
    r2 = math_ops.multiply(a, c)

    h = sess.partial_run_setup([r1, r2], [a, b, c])
    sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 'has already been fed.$'):
      sess.partial_run(h, r2, feed_dict={a: 1, c: 3})

  def runTestPartialRunAlreadyFetched(self, sess):
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)
    r2 = math_ops.multiply(a, c)

    h = sess.partial_run_setup([r1, r2], [a, b, c])
    sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 'has already been fetched.$'):
      sess.partial_run(h, r1, feed_dict={c: 3})

  def testInvalidPartialRunSetup(self):
    sess = session.Session()
    x = array_ops.placeholder(dtypes.float32, shape=[])
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        'specify at least one target to fetch or execute.'):
      sess.partial_run_setup(fetches=[], feeds=[x])

  def testPartialRunDirect(self):
    self.runTestPartialRun(session.Session())

  def testPartialRunIncompleteDirect(self):
    self.runTestPartialRunIncomplete(session.Session())

  def testConcurrentPartialRunDirect(self):
    self.runTestConcurrentPartialRun(session.Session())

  def testManyPartialRunDirect(self):
    self.runTestManyPartialRun(session.Session())

  def testRunAndPartialRunDirect(self):
    self.runTestRunAndPartialRun(session.Session())

  def testPartialRunMissingPlaceholderFeedExceptionDirect(self):
    self.runTestPartialRunMissingPlaceholderFeedException(session.Session())

  def testPartialRunUnspecifiedFeedDirect(self):
    self.runTestPartialRunUnspecifiedFeed(session.Session())

  def testPartialRunUnspecifiedFetchDirect(self):
    self.runTestPartialRunUnspecifiedFetch(session.Session())

  def testPartialRunAlreadyFedDirect(self):
    self.runTestPartialRunAlreadyFed(session.Session())

  def testPartialRunAlreadyFetchedDirect(self):
    self.runTestPartialRunAlreadyFetched(session.Session())

  def testPartialRunDist(self):
    server = server_lib.Server.create_local_server()
    self.runTestPartialRun(session.Session(server.target))

  def testPartialRunIncompleteDist(self):
    server = server_lib.Server.create_local_server()
    self.runTestPartialRunIncomplete(session.Session(server.target))

  def testConcurrentPartialRunDist(self):
    server = server_lib.Server.create_local_server()
    self.runTestConcurrentPartialRun(session.Session(server.target))

  def testManyPartialRunDist(self):
    server = server_lib.Server.create_local_server()
    self.runTestManyPartialRun(session.Session(server.target))

  def testRunAndPartialRunDist(self):
    server = server_lib.Server.create_local_server()
    self.runTestRunAndPartialRun(session.Session(server.target))

  def testPartialRunMissingPlaceholderFeedExceptionDist(self):
    server = server_lib.Server.create_local_server()
    self.runTestPartialRunMissingPlaceholderFeedException(
        session.Session(server.target))

  def testPartialRunUnspecifiedFeedDist(self):
    server = server_lib.Server.create_local_server()
    self.runTestPartialRunUnspecifiedFeed(session.Session(server.target))

  def testPartialRunUnspecifiedFetchDist(self):
    server = server_lib.Server.create_local_server()
    self.runTestPartialRunUnspecifiedFetch(session.Session(server.target))

  def testPartialRunAlreadyFedDist(self):
    server = server_lib.Server.create_local_server()
    self.runTestPartialRunAlreadyFed(session.Session(server.target))

  def testPartialRunAlreadyFetchedDist(self):
    server = server_lib.Server.create_local_server()
    self.runTestPartialRunAlreadyFetched(session.Session(server.target))

  def testFeedDictKeyException(self):
    with session.Session() as sess:
      a = constant_op.constant(1.0, dtypes.float32, name='a')
      with self.assertRaisesRegexp(TypeError, 'Cannot interpret feed_dict'):
        sess.run(a, feed_dict={'a': [2.0]})

  def testPerStepTrace(self):
    run_options = config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.FULL_TRACE)
    run_metadata = config_pb2.RunMetadata()

    with ops.device('/cpu:0'):
      with session.Session() as sess:
        sess.run(constant_op.constant(1.0))
        self.assertTrue(not run_metadata.HasField('step_stats'))

        sess.run(constant_op.constant(1.0), run_metadata=run_metadata)
        self.assertTrue(not run_metadata.HasField('step_stats'))

        sess.run(constant_op.constant(1.0),
                 options=run_options,
                 run_metadata=run_metadata)

        self.assertTrue(run_metadata.HasField('step_stats'))
        self.assertEquals(len(run_metadata.step_stats.dev_stats), 1)

  def testRunOptionsRunMetadata(self):
    run_options = config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.FULL_TRACE)
    run_metadata = config_pb2.RunMetadata()

    with ops.device('/cpu:0'):
      with session.Session() as sess:
        # all combinations are valid
        sess.run(constant_op.constant(1.0), options=None, run_metadata=None)
        sess.run(constant_op.constant(1.0), options=None,
                 run_metadata=run_metadata)
        self.assertTrue(not run_metadata.HasField('step_stats'))

        sess.run(constant_op.constant(1.0), options=run_options,
                 run_metadata=None)
        self.assertTrue(not run_metadata.HasField('step_stats'))

        sess.run(constant_op.constant(1.0), options=run_options,
                 run_metadata=run_metadata)

        self.assertTrue(run_metadata.HasField('step_stats'))
        self.assertEquals(len(run_metadata.step_stats.dev_stats), 1)

  def testFeedShapeCompatibility(self):
    with session.Session() as sess:
      some_tensor = constant_op.constant([2.0, 2.0, 2.0, 2.0])
      new_shape = constant_op.constant([2, 2])
      reshaped_tensor = array_ops.reshape(some_tensor, new_shape)

      with self.assertRaisesRegexp(ValueError, 'Cannot feed value of shape'):
        sess.run(reshaped_tensor, feed_dict={some_tensor: [1.0, 2.0, 3.0]})

      with self.assertRaisesRegexp(ValueError, 'may not be fed'):
        sess.run(reshaped_tensor, feed_dict={new_shape: [3, 7]})

  def testInferShapesFalse(self):
    with ops.Graph().as_default(), ops.device('/cpu:0'):
      a = constant_op.constant([[1, 2]])
      sess = session.Session()
      self.assertFalse('_output_shapes' in sess.graph_def.node[0].attr)
      # Avoid lint error regarding 'unused' var a.
      self.assertTrue(a == a)

  def testInferShapesTrue(self):
    config = config_pb2.ConfigProto(
        graph_options=config_pb2.GraphOptions(infer_shapes=True))
    with ops.Graph().as_default(), ops.device('/cpu:0'):
      a = constant_op.constant([[1, 2]])
      sess = session.Session(config=config)
      self.assertTrue('_output_shapes' in sess.graph_def.node[0].attr)
      # Avoid lint error regarding 'unused' var a.
      self.assertTrue(a == a)

  def testBuildCostModel(self):
    run_options = config_pb2.RunOptions()
    config = config_pb2.ConfigProto(
        allow_soft_placement=True,
        graph_options=config_pb2.GraphOptions(build_cost_model=100))
    with session.Session(config=config) as sess:
      with ops.device('/gpu:0'):
        a = array_ops.placeholder(dtypes.float32, shape=[])
        b = math_ops.add(a, a)
        c = array_ops.identity(b)
        d = math_ops.multiply(c, c)
      for step in xrange(120):
        run_metadata = config_pb2.RunMetadata()
        sess.run(d, feed_dict={a: 1.0},
                 options=run_options, run_metadata=run_metadata)
        if step == 99:
          self.assertTrue(run_metadata.HasField('cost_graph'))
        else:
          self.assertFalse(run_metadata.HasField('cost_graph'))

  def testNonInteractiveSessionNesting(self):
    sess1 = session.Session()
    sess1_controller = sess1.as_default()
    sess1_controller.__enter__()

    sess2 = session.Session()
    sess2_controller = sess2.as_default()
    sess2_controller.__enter__()

    with self.assertRaisesRegexp(AssertionError, 'Nesting violated'):
      sess1_controller.__exit__(None, None, None)

    ops._default_session_stack.reset()

  def testInteractiveSessionNesting(self):
    sess1 = session.InteractiveSession()
    sess2 = session.InteractiveSession()
    del sess1
    del sess2

  def testAsDefault(self):
    c = constant_op.constant(37)
    sess = session.Session()
    with sess.as_default():
      self.assertEqual(37, c.eval())

    # Ensure that the session remains valid even when it is not captured.
    with session.Session().as_default():
      self.assertEqual(37, c.eval())

  def testReentry(self):
    sess = session.Session()
    with self.assertRaisesRegexp(RuntimeError, 'not re-entrant'):
      with sess:
        with sess:
          pass

  def testInvalidArgument(self):
    with self.assertRaisesRegexp(TypeError, 'target must be a string'):
      session.Session(37)
    with self.assertRaisesRegexp(TypeError, 'config must be a tf.ConfigProto'):
      session.Session(config=37)
    with self.assertRaisesRegexp(TypeError, 'graph must be a tf.Graph'):
      session.Session(graph=37)

  def testTimeoutWithShortOperations(self):
    num_epochs = 5
    q = data_flow_ops.FIFOQueue(
        capacity=50, dtypes=[dtypes.int32], shapes=[()])
    enqueue_op = q.enqueue_many(constant_op.constant([1, 2]))

    # Use a 10-second timeout, which should be longer than any
    # non-blocking enqueue_many op.
    config = config_pb2.ConfigProto(operation_timeout_in_ms=10000)
    with session.Session(config=config) as sess:
      for _ in range(num_epochs):
        sess.run(enqueue_op)
      self.assertEqual(sess.run(q.size()), num_epochs * 2)

  def testRegisterFetchAndFeedConversionFunctions(self):
    class SquaredTensor(object):
      def __init__(self, tensor):
        self.sq = math_ops.square(tensor)

    fetch_fn = lambda squared_tensor: ([squared_tensor.sq], lambda val: val[0])
    feed_fn1 = lambda feed, feed_val: [(feed.sq, feed_val)]
    feed_fn2 = lambda feed: [feed.sq]

    session.register_session_run_conversion_functions(SquaredTensor, fetch_fn,
        feed_fn1, feed_fn2)
    with self.assertRaises(ValueError):
      session.register_session_run_conversion_functions(SquaredTensor,
          fetch_fn, feed_fn1, feed_fn2)
    with self.test_session() as sess:
      np1 = np.array([1.0, 1.5, 2.0, 2.5])
      np2 = np.array([3.0, 3.5, 4.0, 4.5])
      squared_tensor = SquaredTensor(np2)
      squared_eval = sess.run(squared_tensor)
      self.assertAllClose(np2 * np2, squared_eval)
      squared_eval = sess.run(squared_tensor, feed_dict={
        squared_tensor : np1 * np1})
      self.assertAllClose(np1 * np1, squared_eval)
      partial_run = sess.partial_run_setup([squared_tensor], [])
      squared_eval = sess.partial_run(partial_run, squared_tensor)
      self.assertAllClose(np2 * np2, squared_eval)

  def testDefaultLogDevicePlacement(self):
    class CaptureStderr(str):
      """Class to capture stderr from C++ shared library."""

      def __enter__(self):
        self._esc = compat.as_str('\b')
        self._output = compat.as_str('')
        self._stderr = sys.stderr
        self._fd = self._stderr.fileno()
        self._out_pipe, in_pipe = os.pipe()
        # Save the original io stream.
        self._dup_fd = os.dup(self._fd)
        # Replace the original io stream with in pipe.
        os.dup2(in_pipe, self._fd)
        return self

      def __exit__(self, *args):
        self._stderr.write(self._esc)
        self._stderr.flush()
        self.read()
        os.close(self._out_pipe)
        # Restore the original io stream.
        os.dup2(self._dup_fd, self._fd)

      def read(self):
        while True:
          data = os.read(self._out_pipe, 1)
          if not data or compat.as_str(data) == self._esc:
            break
          self._output += compat.as_str(data)

      def __str__(self):
        return self._output

    # Passing the config to the server, but not the session should still result
    # in logging device placement.
    config = config_pb2.ConfigProto(log_device_placement=True)
    server = server_lib.Server.create_local_server(config=config)
    a = constant_op.constant(1)
    b = constant_op.constant(2)
    c = a + b
    with session.Session(server.target) as sess:
      with CaptureStderr() as log:
        sess.run(c)
      # Ensure that we did log device placement.
      self.assertTrue('/job:local/replica:0/task:0/cpu:0' in str(log), str(log))

  def testLocalMasterSessionTimeout(self):
    # Test that the timeout passed in a config to the session works correctly.
    config = config_pb2.ConfigProto(operation_timeout_in_ms=1000)
    server = server_lib.Server.create_local_server()
    q = data_flow_ops.FIFOQueue(1, dtypes.float32)
    dequeued_t = q.dequeue()

    with session.Session(server.target, config=config) as sess:
      # Intentionally do not run any enqueue_ops so that dequeue will block
      # until operation_timeout_in_ms.
      with self.assertRaises(errors.DeadlineExceededError):
        sess.run(dequeued_t)

  def testDefaultServerTimeout(self):
    # Test that the default server config timeout gets used when no Session
    # config is provided.
    config = config_pb2.ConfigProto(operation_timeout_in_ms=1000)
    server = server_lib.Server.create_local_server(config=config)
    q = data_flow_ops.FIFOQueue(1, dtypes.float32)
    dequeued_t = q.dequeue()

    with session.Session(server.target) as sess:
      # Intentionally do not run any enqueue_ops so that dequeue will block
      # until operation_timeout_in_ms.
      with self.assertRaises(errors.DeadlineExceededError):
        sess.run(dequeued_t)

  def runTestBuildGraphError(self, sess):
    # Ensure that errors from building the graph get propagated.
    data = array_ops.placeholder(dtypes.float32, shape=[])
    enter_1 = control_flow_ops.enter(data, 'foo_1', False)
    enter_2 = control_flow_ops.enter(data, 'foo_2', False)
    res = math_ops.add(enter_1, enter_2)
    with self.assertRaisesOpError('has inputs from different frames'):
      sess.run(res, feed_dict={data: 1.0})

  def testBuildGraphErrorDirect(self):
    self.runTestBuildGraphError(session.Session())

  def testBuildGraphErrorDist(self):
    server = server_lib.Server.create_local_server()
    self.runTestBuildGraphError(session.Session(server.target))

  def testClusterSpecPropagationSimple(self):
    server1 = server_lib.Server.create_local_server()
    server2 = server_lib.Server.create_local_server()
    cluster_def = cluster_pb2.ClusterDef()
    job = cluster_def.job.add()
    job.name = 'worker'
    job.tasks[0] = server1.target[len('grpc://'):]
    job.tasks[1] = server2.target[len('grpc://'):]
    config = config_pb2.ConfigProto(cluster_def=cluster_def)

    const = constant_op.constant(17)
    sess = session.Session(server1.target, config=config)
    output = sess.run(const)
    self.assertEqual(17, output)

  def testClusterSpecPropagationWorker2Placement(self):
    server1 = server_lib.Server.create_local_server()
    server2 = server_lib.Server.create_local_server()
    cluster_def = cluster_pb2.ClusterDef()
    job = cluster_def.job.add()
    job.name = 'worker'
    job.tasks[0] = server1.target[len('grpc://'):]
    job.tasks[1] = server2.target[len('grpc://'):]
    config = config_pb2.ConfigProto(cluster_def=cluster_def)

    with ops.Graph().as_default() as g, ops.device('/job:worker/task:1'):
      const = constant_op.constant(17)
    sess = session.Session(server1.target, config=config, graph=g)
    run_options = config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.FULL_TRACE)
    run_metadata = config_pb2.RunMetadata()
    output = sess.run(const, options=run_options, run_metadata=run_metadata)
    self.assertEqual(17, output)
    self.assertEqual(1,
                     len([
                         node_stats
                         for dev_stats in run_metadata.step_stats.dev_stats
                         for node_stats in dev_stats.node_stats
                         if '/job:worker/replica:0/task:1/device:CPU:0' ==
                         dev_stats.device and 'Const' == node_stats.node_name
                     ]))

  def testClusterSpecPropagationWorker1Placement(self):
    server1 = server_lib.Server.create_local_server()
    server2 = server_lib.Server.create_local_server()
    cluster_def = cluster_pb2.ClusterDef()
    job = cluster_def.job.add()
    job.name = 'worker'
    job.tasks[0] = server1.target[len('grpc://'):]
    job.tasks[1] = server2.target[len('grpc://'):]
    config = config_pb2.ConfigProto(cluster_def=cluster_def)

    with ops.Graph().as_default() as g, ops.device('/job:worker/task:0'):
      const = constant_op.constant(17)
    sess = session.Session(server1.target, config=config, graph=g)
    output = sess.run(const)
    self.assertEqual(17, output)

  def testClusterSpecPropagationThreeServers2Graphs(self):
    """Boots 3 servers, creates 2 sessions, ensures appropriate operations.

    We create 2 clusterspecs:
     1. server2 as the master, server1 as a worker
     2. server2 as the master, server3 as a worker

    We ensure that variables on the workers are independent.
    """
    server1 = server_lib.Server.create_local_server()
    server2 = server_lib.Server.create_local_server()
    server3 = server_lib.Server.create_local_server()
    cluster_def1 = cluster_pb2.ClusterDef()
    job1 = cluster_def1.job.add()
    job1.name = 'worker1'
    job1.tasks[0] = server2.target[len('grpc://'):]
    job1.tasks[1] = server1.target[len('grpc://'):]

    cluster_def2 = cluster_pb2.ClusterDef()
    job2 = cluster_def2.job.add()
    job2.name = 'worker2'
    job2.tasks[0] = server2.target[len('grpc://'):]
    job2.tasks[1] = server3.target[len('grpc://'):]

    config1 = config_pb2.ConfigProto(cluster_def=cluster_def1)
    config2 = config_pb2.ConfigProto(cluster_def=cluster_def2)

    with ops.Graph().as_default() as g1:
      with ops.device('/job:worker1/task:1'):
        var1 = variables.Variable(array_ops.zeros([2]), name='var1')
        update_op1 = state_ops.assign_add(
            var1, array_ops.ones([2]), name='var1_assign_add')
        init1 = variables.global_variables_initializer()

    with ops.Graph().as_default() as g2:
      with ops.device('/job:worker2/task:1'):
        var2 = variables.Variable(array_ops.zeros([2]), name='var2')
        update_op2 = state_ops.assign_add(
            var2, array_ops.ones([2]), name='var2_assign_add')
        init2 = variables.global_variables_initializer()

    sess1 = session.Session(server2.target, graph=g1, config=config1)
    sess2 = session.Session(server2.target, graph=g2, config=config2)

    init1.run(session=sess1)
    init2.run(session=sess2)

    expected_zeros = np.zeros([2])
    expected_ones = np.ones([2])

    self.assertAllEqual(expected_zeros, sess1.run(var1))
    self.assertAllEqual(expected_zeros, sess2.run(var2))

    self.assertAllEqual(expected_ones, sess1.run(update_op1))
    self.assertAllEqual(expected_ones, sess1.run(var1))
    self.assertAllEqual(expected_zeros, sess2.run(var2))
    self.assertAllEqual(expected_ones, sess2.run(update_op2))
    self.assertAllEqual(expected_ones + expected_ones, sess1.run(update_op1))
    self.assertAllEqual(expected_ones, sess2.run(var2))
    self.assertAllEqual(expected_ones + expected_ones, sess1.run(var1))

  def testClusterSpecPropagationThreeServers(self):
    """Boots 3 servers, creates 2 sessions, ensures appropriate operations.

    We create 2 clusterspecs:
     1. server2 as the master, server1 as a worker
     2. server2 as the master, server3 as a worker

    We ensure that variables on the workers are independent.
    """
    server1 = server_lib.Server.create_local_server()
    server2 = server_lib.Server.create_local_server()
    server3 = server_lib.Server.create_local_server()
    cluster_def1 = cluster_pb2.ClusterDef()
    job1 = cluster_def1.job.add()
    job1.name = 'worker'
    job1.tasks[0] = server2.target[len('grpc://'):]
    job1.tasks[1] = server1.target[len('grpc://'):]

    cluster_def2 = cluster_pb2.ClusterDef()
    job2 = cluster_def2.job.add()
    job2.name = 'worker'
    job2.tasks[0] = server2.target[len('grpc://'):]
    job2.tasks[1] = server3.target[len('grpc://'):]

    config1 = config_pb2.ConfigProto(cluster_def=cluster_def1)
    config2 = config_pb2.ConfigProto(cluster_def=cluster_def2)

    with ops.device('/job:worker/task:1'):
      var = variables.Variable(array_ops.zeros([2]), name='var')
      feed = array_ops.placeholder(dtypes.float32, shape=(2))
      update_op = var.assign_add(feed)

    sess1 = session.Session(server2.target, config=config1)
    sess2 = session.Session(server2.target, config=config2)

    variables.global_variables_initializer().run(session=sess1)
    variables.global_variables_initializer().run(session=sess2)

    expected_zeros = np.zeros([2])
    expected_ones = np.ones([2])

    self.assertAllEqual(expected_zeros, sess1.run(var))
    self.assertAllEqual(expected_zeros, sess2.run(var))
    self.assertAllEqual(expected_ones,
                        sess1.run(update_op, feed_dict={feed: expected_ones}))
    self.assertAllEqual(expected_ones, sess1.run(var))
    self.assertAllEqual(expected_zeros, sess2.run(var))
    self.assertAllEqual(expected_ones,
                        sess2.run(update_op, feed_dict={feed: expected_ones}))
    self.assertAllEqual(expected_ones + expected_ones,
                        sess1.run(update_op, feed_dict={feed: expected_ones}))
    self.assertAllEqual(expected_ones, sess2.run(var))
    self.assertAllEqual(expected_ones + expected_ones, sess1.run(var))

  def testClusterSpecPropagationThreeServersOneCluster(self):
    """Boots 3 servers, ensures appropriate communication across workers.

    Additionally, in this cluster, we ensure the master is not the 0-th worker.

    Note: this test only uses one session.
    """
    server1 = server_lib.Server.create_local_server()
    server2 = server_lib.Server.create_local_server()
    server3 = server_lib.Server.create_local_server()
    cluster_def = cluster_pb2.ClusterDef()
    job = cluster_def.job.add()
    job.name = 'worker'
    job.tasks[0] = server3.target[len('grpc://'):]
    job.tasks[1] = server2.target[len('grpc://'):]
    job.tasks[2] = server1.target[len('grpc://'):]
    config = config_pb2.ConfigProto(cluster_def=cluster_def)

    # Add ops to the devices in non-linear order.

    with ops.device('/job:worker/task:1'):
      feed1 = array_ops.placeholder(dtypes.float32, shape=(2))
      const1 = constant_op.constant(2.0)
      mul1 = const1 * feed1

    with ops.device('/job:worker/task:2'):
      feed2 = array_ops.placeholder(dtypes.float32, shape=(2))
      const2 = constant_op.constant(2.0)
      mul2 = const2 * feed2

    with ops.device('/job:worker/task:0'):
      feed0 = array_ops.placeholder(dtypes.float32, shape=(2))
      const0 = constant_op.constant(2.0)
      mul0 = const0 * feed0

    sum_op = mul0 + mul1 + mul2

    ones = np.ones([2])
    run_options = config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.FULL_TRACE)
    run_metadata = config_pb2.RunMetadata()

    # Run!
    with session.Session(server1.target, config=config) as sess:
      output = sess.run(
          sum_op,
          options=run_options,
          run_metadata=run_metadata,
          feed_dict={feed1: ones,
                     feed2: ones,
                     feed0: ones})
      self.assertAllEqual(6 * ones, output)

      self.assertEqual(
          3,
          len([
              dev_stats.device
              for dev_stats in run_metadata.step_stats.dev_stats
              for node_stats in dev_stats.node_stats
              if '/job:worker/replica:0/task:' in dev_stats.device and
              node_stats.node_name.startswith('Const')
          ]), run_metadata)

  def testClusterSpecPropagationPartialRun(self):
    """Test successful partial run with ClusterSpec propagation."""
    server1 = server_lib.Server.create_local_server()
    server2 = server_lib.Server.create_local_server()

    cluster_def = cluster_pb2.ClusterDef()
    job = cluster_def.job.add()
    job.name = 'worker'
    job.tasks[0] = server1.target[len('grpc://'):]
    job.tasks[1] = server2.target[len('grpc://'):]
    config = config_pb2.ConfigProto(cluster_def=cluster_def)

    with ops.device('/job:worker/task:0'):
      a = array_ops.placeholder(dtypes.float32, shape=[])
    with ops.device('/job:worker/task:1'):
      b = array_ops.placeholder(dtypes.float32, shape=[])
      c = array_ops.placeholder(dtypes.float32, shape=[])
      r1 = math_ops.add(a, b)
    with ops.device('/job:worker/task:0'):
      r2 = math_ops.multiply(r1, c)

    with session.Session(server1.target, config=config) as sess:
      h = sess.partial_run_setup([r1, r2], [a, b, c])
      res = sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
      self.assertEqual(3, res)
      res = sess.partial_run(h, r2, feed_dict={c: 3})
      self.assertEqual(9, res)


if __name__ == '__main__':
  googletest.main()
