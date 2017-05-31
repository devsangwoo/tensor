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
# WITHOUT WARRANTIES OiR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# pylint: disable=g-long-lambda
"""Tests for tensorflow.ops.control_flow_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import device_lib
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
# pylint: disable=unused-import
import tensorflow.python.ops.tensor_array_grad
# pylint: enable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.training import adam
from tensorflow.python.training import gradient_descent
from tensorflow.python.util import nest


def check_op_order(graph):
  """Sanity check on the ordering of op id."""

  for op in graph.get_operations():
    for v in op.inputs:
      assert v.op._id < op._id or op.type == "Merge", (
          "The id of %s must be less than the id of %s" % (v.op.name, op.name))
  return True


def check_consumers(graph):
  """Sanity check on the consumer list of the tensors."""

  consumer_count = {}
  for op in graph.get_operations():
    for v in op.inputs:
      cnt = consumer_count.get(v, 0)
      consumer_count[v] = cnt + 1
  for k, v in consumer_count.items():
    if len(k.consumers()) != v:
      return False
  return True


def all_fetchables():
  tensor_names = []
  graph = ops.get_default_graph()
  for op in graph.get_operations():
    for t in op.outputs:
      if graph.is_fetchable(t):
        tensor_names.append(t.name)
  return tensor_names


def all_feedables():
  feedable_tensors = []
  graph = ops.get_default_graph()
  for op in graph.get_operations():
    for t in op.inputs:
      if graph.is_feedable(t):
        feedable_tensors.append(t)
  return feedable_tensors


def opt_cfg():
  return config_pb2.ConfigProto(
      allow_soft_placement=True,
      graph_options=config_pb2.GraphOptions(
          optimizer_options=config_pb2.OptimizerOptions(
              opt_level=config_pb2.OptimizerOptions.L1,
              do_function_inlining=True,
              do_constant_folding=True)))


def isum(s):
  i = constant_op.constant(0, name="i")
  c = lambda i, s: math_ops.less(i, 10)
  b = lambda i, s: [math_ops.add(i, 1), math_ops.add(i, s)]
  _, r_s = control_flow_ops.while_loop(c, b, [i, s])
  return r_s


class ControlFlowTest(test.TestCase):

  def testRefIdentity(self):
    with self.test_session():
      v = variables.Variable(7)

      v = control_flow_ops._Identity(v)
      op = state_ops.assign(v, 9)
      v2 = control_flow_ops.with_dependencies([op], v)

      self.assertTrue(check_op_order(v.graph))
      self.assertTrue(isinstance(v2, ops.Tensor))
      variables.global_variables_initializer().run()
      self.assertEqual(9, v2.eval())

  def testRefEnter(self):
    with self.test_session():
      v = variables.Variable(7)

      enter_v = control_flow_ops._Enter(v, "foo_1", is_constant=True)
      nine = constant_op.constant(9)
      enter_nine = control_flow_ops.enter(nine, "foo_1")
      op = state_ops.assign(enter_v, enter_nine)
      v2 = control_flow_ops.with_dependencies([op], enter_v)
      v3 = control_flow_ops.exit(v2)
      variables.global_variables_initializer().run()
      self.assertEqual(9, v3.eval())

  def testRefSwitch(self):
    with self.test_session():
      v = variables.Variable(7)

      p = constant_op.constant(True)
      v1 = control_flow_ops._SwitchRefOrTensor(v._ref(), p)  # pylint: disable=protected-access
      v2 = state_ops.assign(v1[1], 9)
      variables.global_variables_initializer().run()
      self.assertEqual(9, v2.eval())

  def testEnterMulExit(self):
    with self.test_session():
      data = constant_op.constant([1, 2, 3, 4, 5, 6], name="data")
      enter_data = control_flow_ops.enter(data, "foo_1", False)
      five = constant_op.constant(5)
      enter_five = control_flow_ops.enter(five, "foo_1", False)
      mul_op = math_ops.multiply(enter_data, enter_five)
      exit_op = control_flow_ops.exit(mul_op)

      result = exit_op.eval()
    self.assertAllEqual(np.array([x * 5 for x in [1, 2, 3, 4, 5, 6]]), result)

  def testSwitchMergeIndexedSlices(self):
    with self.test_session():
      values = constant_op.constant([1, 2, 3, 4, 5, 6])
      indices = constant_op.constant([0, 2, 4, 6, 8, 10])
      data = ops.IndexedSlices(values, indices)
      pred = ops.convert_to_tensor(True)
      switch_op = control_flow_ops.switch(data, pred)
      merge_op = control_flow_ops.merge(switch_op)[0]

      val = merge_op.values.eval()
      ind = merge_op.indices.eval()
    self.assertAllEqual(np.arange(1, 7), val)
    self.assertAllEqual(np.arange(0, 12, 2), ind)

  def testSwitchDeadBranch(self):
    with self.test_session():
      data = constant_op.constant([1, 2, 3, 4, 5, 6], name="data")
      ports = ops.convert_to_tensor(True, name="ports")
      switch_op = control_flow_ops.switch(data, ports)
      dead_branch = array_ops.identity(switch_op[0])

      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          lambda e: "Retval[0] does not have value" in str(e)):
        dead_branch.eval()

  def testSwitchMergeLess(self):
    with self.test_session():
      data = constant_op.constant([1, 2, 3, 4, 5, 6], name="data")
      zero = ops.convert_to_tensor(0)
      one = ops.convert_to_tensor(1)
      less_op = math_ops.less(zero, one)
      switch_op = control_flow_ops.switch(data, less_op)
      merge_op = control_flow_ops.merge(switch_op)[0]

      result = merge_op.eval()
    self.assertAllEqual(np.arange(1, 7), result)

  def testSwitchMergeAddIdentity(self):
    with self.test_session():
      data = constant_op.constant([1, 2, 3, 4, 5, 6], name="data")
      ports = ops.convert_to_tensor(False, name="ports")
      switch_op = control_flow_ops.switch(data, ports)
      one = constant_op.constant(1)
      add_op = math_ops.add(switch_op[0], one)
      id_op = array_ops.identity(switch_op[1])
      merge_op = control_flow_ops.merge([add_op, id_op])[0]

      result = merge_op.eval()
    self.assertAllEqual(np.array([x + 1 for x in [1, 2, 3, 4, 5, 6]]), result)

  def testSwitchMergeAddMul(self):
    with self.test_session():
      data = constant_op.constant([1, 2, 3, 4, 5, 6], name="data")
      ports = ops.convert_to_tensor(True, name="ports")
      switch_op = control_flow_ops.switch(data, ports)
      one = constant_op.constant(1)
      add_op = math_ops.add(switch_op[0], one)
      five = constant_op.constant(5)
      mul_op = math_ops.multiply(switch_op[1], five)
      merge_op = control_flow_ops.merge([add_op, mul_op])[0]

      result = merge_op.eval()
    self.assertAllEqual(np.array([x * 5 for x in [1, 2, 3, 4, 5, 6]]), result)

  def testLoop_false(self):
    with self.test_session():
      false = ops.convert_to_tensor(False)
      n = constant_op.constant(10)

      enter_false = control_flow_ops.enter(false, "foo_1", False)
      enter_n = control_flow_ops.enter(n, "foo_1", False)

      merge_n = control_flow_ops.merge([enter_n, enter_n], name="merge_n")[0]
      switch_n = control_flow_ops.switch(merge_n, enter_false)
      exit_n = control_flow_ops.exit(switch_n[0])
      next_n = control_flow_ops.next_iteration(switch_n[0])
      merge_n.op._update_input(1, next_n)

      result = exit_n.eval()
    self.assertAllEqual(10, result)

  def testLoop_1(self):
    with self.test_session():
      zero = constant_op.constant(0)
      one = constant_op.constant(1)
      n = constant_op.constant(10)

      enter_i = control_flow_ops.enter(zero, "foo", False)
      enter_one = control_flow_ops.enter(one, "foo", True)
      enter_n = control_flow_ops.enter(n, "foo", True)

      with ops.device(test.gpu_device_name()):
        merge_i = control_flow_ops.merge([enter_i, enter_i])[0]

      less_op = math_ops.less(merge_i, enter_n)
      cond_op = control_flow_ops.loop_cond(less_op)
      switch_i = control_flow_ops.switch(merge_i, cond_op)

      add_i = math_ops.add(switch_i[1], enter_one)

      next_i = control_flow_ops.next_iteration(add_i)
      merge_i.op._update_input(1, next_i)

      exit_i = control_flow_ops.exit(switch_i[0])
      result = exit_i.eval()
    self.assertAllEqual(10, result)

  def testLoop_2(self):
    with self.test_session():
      zero = constant_op.constant(0)
      one = constant_op.constant(1)
      n = constant_op.constant(10)

      enter_i = control_flow_ops.enter(zero, "foo", False)
      enter_one = control_flow_ops.enter(one, "foo", True)
      enter_n = control_flow_ops.enter(n, "foo", True)

      merge_i = control_flow_ops.merge([enter_i, enter_i])[0]

      less_op = math_ops.less(merge_i, enter_n)
      cond_op = control_flow_ops.loop_cond(less_op)
      switch_i = control_flow_ops.switch(merge_i, cond_op)

      add_i = math_ops.add(switch_i[1], enter_one)

      with ops.device(test.gpu_device_name()):
        next_i = control_flow_ops.next_iteration(add_i)
      merge_i.op._update_input(1, next_i)

      exit_i = control_flow_ops.exit(switch_i[0])
      result = exit_i.eval()
    self.assertAllEqual(10, result)

  def testDifferentFrame(self):
    with self.test_session():
      data = array_ops.placeholder(dtypes.float32, shape=[])
      enter_1 = control_flow_ops.enter(data, "foo_1", False)
      enter_2 = control_flow_ops.enter(data, "foo_2", False)
      res = math_ops.add(enter_1, enter_2)
      with self.assertRaisesOpError("has inputs from different frames"):
        res.eval(feed_dict={data: 1.0})

  def testCondBool(self):
    values = constant_op.constant(10)
    fn1 = lambda: math_ops.add(values, 1)
    fn2 = lambda: math_ops.subtract(values, 1)
    with self.assertRaisesRegexp(TypeError, "must not be a Python bool"):
      _ = control_flow_ops.cond(False, fn1, fn2)

  def testCondInt(self):
    p = array_ops.placeholder(dtypes.bool, shape=[])
    v = constant_op.constant(10)
    fn1 = lambda: math_ops.add(v, 1)
    fn2 = lambda: math_ops.subtract(v, 1)
    y = control_flow_ops.cond(p, fn1, fn2)
    grad = gradients_impl.gradients(y, [v])
    self.assertAllEqual([None], grad)

  def testFetchables(self):
    with self.test_session() as sess:
      x = array_ops.placeholder(dtypes.float32)
      control_flow_ops.cond(
          constant_op.constant(True), lambda: x + 2, lambda: x + 0)
      tensor_names = all_fetchables()
      for name in tensor_names:
        sess.run(name, feed_dict={x: 3})

  def testFeedable(self):
    with self.test_session() as sess:
      c = constant_op.constant(2)
      i0 = constant_op.constant(0)
      r = control_flow_ops.while_loop(lambda i: i < 1000,
                                      lambda i: math_ops.square(c) + i, [i0])
      self.assertEqual(1000, r.eval(feed_dict={i0: 0}))
      feedable_tensors = all_feedables()
      for t in feedable_tensors:
        sess.run(r, feed_dict={t: 3})
      graph = ops.get_default_graph()
      for op in graph.get_operations():
        for t in op.inputs:
          if t not in feedable_tensors and t.dtype is dtypes.int32:
            with self.assertRaisesRegexp(ValueError, "may not be fed"):
              sess.run(r, feed_dict={t: 3})

  def testCondIndexedSlices(self):
    with self.test_session():
      values = constant_op.constant(10)
      indices = constant_op.constant(0)
      x = ops.IndexedSlices(values, indices)
      pred = math_ops.less(1, 2)
      fn1 = lambda: ops.IndexedSlices(math_ops.add(x.values, 1), indices)
      fn2 = lambda: ops.IndexedSlices(math_ops.subtract(x.values, 1), indices)
      r = control_flow_ops.cond(pred, fn1, fn2)

      val = r.values.eval()
      ind = r.indices.eval()
    self.assertTrue(check_op_order(x.values.graph))
    self.assertAllEqual(11, val)
    self.assertAllEqual(0, ind)

  def testCondSparseTensor(self):
    with self.test_session():
      values = constant_op.constant([2.0, 4.0], name="values")
      indices = constant_op.constant(
          [[0], [3]], dtype=dtypes.int64, name="indices")
      shape = constant_op.constant([10], dtype=dtypes.int64, name="dense_shape")
      x = sparse_tensor.SparseTensor(indices, values, dense_shape=shape)
      pred = math_ops.less(1, 2)
      fn1 = lambda: sparse_tensor.SparseTensor(
          indices + 1, x.values + 1, dense_shape=shape)
      fn2 = lambda: sparse_tensor.SparseTensor(
          indices, x.values - 1, dense_shape=shape)
      r = control_flow_ops.cond(pred, fn1, fn2)
      self.assertAllEqual([3.0, 5.0], r.values.eval())
      self.assertAllEqual([[1], [4]], r.indices.eval())
      self.assertAllEqual(r.values.get_shape(), (2,))

  def testCondResource(self):
    with self.test_session():
      rv = resource_variable_ops.ResourceVariable(True)
      variables.global_variables_initializer().run()
      t = ops.convert_to_tensor(1.0)

      def case():
        assign = resource_variable_ops.assign_variable_op(rv.handle, False)
        with ops.control_dependencies([assign]):
          return array_ops.identity(t)

      self.assertEqual(1.0, control_flow_ops.cond(rv, case, lambda: t).eval())

  def testCondIndexedSlicesDifferentTypes(self):
    with self.test_session():
      values = constant_op.constant(10)
      i_32 = ops.convert_to_tensor(0, name="one", dtype=dtypes.int32)
      i_64 = ops.convert_to_tensor(0, name="one", dtype=dtypes.int64)
      x = ops.IndexedSlices(values, i_32)
      pred = math_ops.less(1, 2)
      fn1 = lambda: ops.IndexedSlices(math_ops.add(x.values, 1), i_32)
      fn2 = lambda: ops.IndexedSlices(math_ops.subtract(x.values, 1), i_64)
      r = control_flow_ops.cond(pred, fn1, fn2)

      val = r.values.eval()
      ind = r.indices.eval()
    self.assertTrue(check_op_order(x.values.graph))
    self.assertAllEqual(11, val)
    self.assertAllEqual(0, ind)
    self.assertTrue(ind.dtype == np.int64)

  def testCondColocation(self):
    with self.test_session(use_gpu=True):
      with ops.device("/cpu:0"):
        v = variables.Variable(7.0)

      x = constant_op.constant(10.0)
      pred = math_ops.less(1.0, 2.0)
      fn1 = lambda: math_ops.add(v, 1.0)
      fn2 = lambda: math_ops.subtract(x, 1.0)
      r = control_flow_ops.cond(pred, fn1, fn2)

      for op in x.graph.get_operations():
        if op.name == "cond/Add/Switch":
          self.assertDeviceEqual(op.device, "/cpu:0")

  def _testCond_1(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      x = constant_op.constant(10)
      pred = math_ops.less(1, 2)
      fn1 = lambda: math_ops.add(x, 1)
      fn2 = lambda: math_ops.subtract(x, 1)
      r = control_flow_ops.cond(pred, fn1, fn2)

      result = r.eval()
    self.assertTrue(check_op_order(x.graph))
    self.assertAllEqual(11, result)

  def testCond_1(self):
    self._testCond_1(use_gpu=False)
    self._testCond_1(use_gpu=True)

  def testCond_2(self):
    with self.test_session():
      x = constant_op.constant(10)
      r = control_flow_ops.cond(
          math_ops.less(1, 0), lambda: math_ops.add(x, 1),
          lambda: math_ops.subtract(x, 1))
      result = r.eval()
    self.assertTrue(check_op_order(x.graph))
    self.assertAllEqual(9, result)

  def testCond_3(self):
    with self.test_session():
      x = constant_op.constant(10)
      pred = math_ops.less(1, 2)
      fn1 = lambda: math_ops.add(x, 1)
      fn2 = lambda: math_ops.subtract(x, 1)
      fn3 = lambda: math_ops.add(control_flow_ops.cond(pred, fn1, fn2), 1)
      r = control_flow_ops.cond(pred, fn3, fn2)

      result = r.eval()
    self.assertTrue(check_op_order(x.graph))
    self.assertAllEqual(12, result)

  def testCond_4(self):
    with self.test_session():
      v1 = variables.Variable(7)
      v2 = variables.Variable(7)
      v3 = variables.Variable(7)

      age = constant_op.constant(3)
      max_age = constant_op.constant(2)
      pred = math_ops.greater(age, max_age)
      fn1 = lambda: [state_ops.assign(v1, 1).op, state_ops.assign(v2, 2).op]
      fn2 = lambda: [state_ops.assign(v3, 3).op, constant_op.constant(10).op]
      r = control_flow_ops.cond(pred, fn1, fn2)

      variables.global_variables_initializer().run()
      self.assertEqual(len(r), 2)
      result = r[1].eval()
      self.assertTrue(check_op_order(age.graph))
      self.assertAllEqual(True, result)
      self.assertAllEqual(7, v1.eval())
      self.assertAllEqual(2, v2.eval())
      self.assertAllEqual(7, v3.eval())

  def testCond_5(self):
    with self.test_session():
      alive = constant_op.constant(True, name="alive")
      count = constant_op.constant(0, name="count")

      def body(i):
        return control_flow_ops.cond(
            alive, lambda: [math_ops.less(i, 3), math_ops.add(count, 1)],
            lambda: [alive, count])

      for i in range(10):
        alive, count = body(i)
      self.assertAllEqual(4, count.eval())

  def testCond_6(self):
    with self.test_session():
      v1 = variables.Variable([7])

      age = constant_op.constant(3)
      pred = math_ops.greater(age, 4)
      fn1 = lambda: age
      fn2 = lambda: v1
      r = control_flow_ops.cond(pred, fn1, fn2)

      variables.global_variables_initializer().run()
      result = r.eval()
      self.assertAllEqual(np.array([7]), result)

  def testCond_7(self):
    with self.test_session() as sess:
      x = constant_op.constant(10)
      y = constant_op.constant(200)
      pred = math_ops.less(1, 2)
      fn1 = lambda: [math_ops.add(x, 1), math_ops.add(x, 2)]
      fn2 = lambda: [y, y]
      r = control_flow_ops.cond(pred, fn1, fn2)
      self.assertAllEqual([11, 12], sess.run(r))

  def testCondRef(self):
    with self.test_session():
      x = gen_state_ops._variable(
          shape=[1],
          dtype=dtypes.float32,
          name="x",
          container="",
          shared_name="")
      true_fn = lambda: x
      false_fn = lambda: constant_op.constant([2.0])
      r = control_flow_ops.cond(constant_op.constant(False), true_fn, false_fn)
      self.assertAllEqual([2.0], r.eval())

  def testCondWithControl(self):
    with self.test_session() as sess:
      control_holder = array_ops.placeholder(dtypes.float32, shape=())
      a = constant_op.constant(3)

      def true_branch():
        with ops.control_dependencies([control_holder]):
          _ = a + 1
        return a + 2

      r = control_flow_ops.cond(
          constant_op.constant(True), true_branch,
          lambda: constant_op.constant(1))
      self.assertEqual(5, r.eval())

  def testUninitializedRefIdentity(self):
    with self.test_session() as sess:
      v = gen_state_ops._variable(
          shape=[1],
          dtype=dtypes.float32,
          name="v",
          container="",
          shared_name="")
      inited = state_ops.is_variable_initialized(v)
      v_f, v_t = control_flow_ops.ref_switch(v, inited)
      # Both v_f and v_t are uninitialized references. However, an actual use
      # of the reference in the 'true' branch in the 'tf.identity' op will
      # not 'fire' when v is uninitialized, so this is a valid construction.
      # This test tests that _ref_identity allows uninitialized ref as input
      # so that this construction is allowed.
      v_f_op = gen_array_ops._ref_identity(v_f)
      v_t_op = gen_array_ops._ref_identity(v_t)
      with ops.control_dependencies([v_f_op]):
        assign_v = state_ops.assign(v, [1.0])
      with ops.control_dependencies([v_t_op]):
        orig_v = array_ops.identity(v)
      merged_op = control_flow_ops.merge([assign_v, orig_v])
      self.assertAllEqual([1.0], sess.run(merged_op.output))

  def testCondSwitchIdentity(self):
    # Make sure the recv identity is not removed by optimization.
    with session.Session(config=opt_cfg()) as sess:
      pred = constant_op.constant(True)

      def fn1():
        return control_flow_ops.no_op()

      def fn2():
        return control_flow_ops.Assert(False, ["Wrong branch!!!"])

      r = control_flow_ops.cond(pred, fn1, fn2)
      sess.run(r)

  def testCondRecvIdentity(self):
    # Make sure the switch identity is not removed by optimization.
    with session.Session(config=opt_cfg()) as sess:
      with ops.device(test.gpu_device_name()):
        pred = constant_op.constant(True)

      def fn1():
        return control_flow_ops.no_op()

      def fn2():
        with ops.device("/cpu:0"):
          return control_flow_ops.Assert(False, ["Wrong branch!!!"])

      r = control_flow_ops.cond(pred, fn1, fn2)
      sess.run(r)

  def testCondGrad_1(self):
    with self.test_session():
      x = constant_op.constant(10.0, name="x")
      pred = math_ops.less(1, 2)
      fn1 = lambda: array_ops.identity(x)
      fn2 = lambda: array_ops.identity(x)
      r = control_flow_ops.cond(pred, fn1, fn2)

      grad = gradients_impl.gradients(r, [x])[0]
      result = grad.eval()
    self.assertAllEqual(1.0, result)

  def testCondGrad_2(self):
    with self.test_session():
      c = array_ops.placeholder(dtypes.int32, shape=[])
      x = constant_op.constant(10.0)
      pred = math_ops.less(c, 2)
      fn1 = lambda: math_ops.multiply(x, 42.0)
      fn2 = lambda: math_ops.multiply(x, 3.0)
      r = control_flow_ops.cond(pred, fn1, fn2)

      grad = gradients_impl.gradients(r, [x])[0]
      self.assertAllEqual(42.0, grad.eval(feed_dict={c: 1}))
      self.assertAllEqual(3.0, grad.eval(feed_dict={c: 3}))

  def testNestedCond_Simple(self):
    with self.test_session():
      x = constant_op.constant(0., name="X")
      y = control_flow_ops.cond(
          constant_op.constant(True), lambda: x,
          lambda: control_flow_ops.cond(x < 1., lambda: x, lambda: x))
      result = gradients_impl.gradients(y, x)[0]
      self.assertEqual(1.0, result.eval())

      z = control_flow_ops.cond(
          constant_op.constant(False), lambda: x,
          lambda: control_flow_ops.cond(x < 1., lambda: x, lambda: x))
      result = gradients_impl.gradients(z, x)[0]
      self.assertEqual(1.0, result.eval())

  def testCondGrad_Gather(self):
    with self.test_session() as sess:
      v1 = variables.Variable([1.0, 42.0])
      c = array_ops.placeholder(dtypes.int32, shape=[])
      pred = math_ops.less(c, 2)
      fn1 = lambda: array_ops.identity(v1)
      fn2 = lambda: array_ops.gather(v1, [1, 1])
      r = control_flow_ops.cond(pred, fn1, fn2)
      grad = gradients_impl.gradients(r, [v1])[0]
      variables.global_variables_initializer().run()
      # Should just be [1, 1], but possibly a sparse representation
      gv, gi = sess.run([grad.values, grad.indices], feed_dict={c: 1})
      dense_gv = [
          sum([y for (x, y) in zip(gi, gv) if x == i]) for i in range(2)
      ]
      self.assertAllEqual(dense_gv, [1.0, 1.0])
      # Should be [0, 2], as the else forwards v1[1] twice
      gv, gi = sess.run([grad.values, grad.indices], feed_dict={c: 3})
      dense_gv = [
          sum([y for (x, y) in zip(gi, gv) if x == i]) for i in range(2)
      ]
      self.assertAllEqual(dense_gv, [0.0, 2.0])

  # Microbenchmark: 256,000 iterations/s.
  def testWhile_1(self):
    with self.test_session():
      n = constant_op.constant(0)
      c = lambda x: math_ops.less(x, 10000)
      b = lambda x: math_ops.add(x, 1)
      r = control_flow_ops.while_loop(c, b, [n], parallel_iterations=20)
      self.assertEqual(10000, r.eval())

  def testWhileWithRefs_1(self):
    with self.test_session() as sess:
      x = variables.Variable(0)._ref()  # pylint: disable=protected-access
      i = constant_op.constant(0)
      c = lambda i, x: math_ops.less(i, 100)

      self.assertEqual(x.dtype, dtypes.int32_ref)

      def b(i, x):
        self.assertEqual(x.dtype, dtypes.int32_ref)
        return (i + 1, gen_array_ops._ref_identity(x))

      r = control_flow_ops.while_loop(c, b, [i, x], parallel_iterations=5)

      variables.global_variables_initializer().run()

      self.assertEqual(r[0].dtype, dtypes.int32)
      self.assertEqual(r[1].dtype, dtypes.int32_ref)

      value_i, value_x = sess.run(r)

    self.assertEqual(100, value_i)
    self.assertEqual(0, value_x)

  def testWhile_2(self):
    with self.test_session():
      s = constant_op.constant(0)
      r = isum(s)
      self.assertAllEqual(45, r.eval())

  # Have more than 10 parallel iterations and hence exercise k-bound
  # most of the time.
  def testWhile_3(self):
    with self.test_session():

      def compute(i, m, c, o):
        m, c = [math_ops.add(m, 1), math_ops.add(c, 1)]
        o = math_ops.add(o, m)
        o = math_ops.add(o, c)
        i = math_ops.add(i, 1)
        return [i, m, c, o]

      i = ops.convert_to_tensor(0)
      m = ops.convert_to_tensor(0)
      c = ops.convert_to_tensor(0)
      o = ops.convert_to_tensor(0)
      d = ops.convert_to_tensor(100)
      r = control_flow_ops.while_loop(lambda i, m, c, o: math_ops.less(i, d),
                                      compute, [i, m, c, o])
      result = r[3].eval()
    self.assertTrue(check_op_order(i.graph))
    self.assertAllEqual(10100, result)

  def testWhile_4(self):
    with self.test_session():

      def compute(i, m, c, o):
        m, c = [array_ops.gather(x, i), array_ops.gather(x, i)]
        o = math_ops.add(o, m)
        o = math_ops.add(o, c)
        i = math_ops.add(i, 1)
        return [i, m, c, o]

      i = ops.convert_to_tensor(0)
      m = ops.convert_to_tensor(0)
      c = ops.convert_to_tensor(0)
      o = ops.convert_to_tensor(0)
      x = ops.convert_to_tensor([1, 2, 3, 4, 5, 6])
      s = array_ops.size(x)
      r = control_flow_ops.while_loop(lambda i, m, c, o: math_ops.less(i, s),
                                      compute, [i, m, c, o])
      result = r[3].eval()
    self.assertTrue(check_op_order(i.graph))
    self.assertAllEqual(42, result)

  def testWhile_5(self):
    with self.test_session():

      def compute(i, c, o):
        c = array_ops.strided_slice(x,
                                    array_ops.expand_dims(i, 0),
                                    [1] + array_ops.expand_dims(i, 0))
        o = array_ops.concat([o, c], 0)
        i = math_ops.add(i, 1)
        return [i, c, o]

      i = ops.convert_to_tensor(0)
      c = ops.convert_to_tensor([0])
      o = ops.convert_to_tensor([0])
      x = ops.convert_to_tensor([1, 2, 3, 4, 5, 6])
      s = array_ops.size(x)
      r = control_flow_ops.while_loop(
          lambda i, c, o: math_ops.less(i, s), compute, [i, c, o], [
              i.get_shape(), tensor_shape.unknown_shape(),
              tensor_shape.unknown_shape()
          ])
      result = r[2].eval()
    self.assertTrue(check_op_order(i.graph))
    self.assertAllEqual(np.array([0, 1, 2, 3, 4, 5, 6]), result)

  def testBufferForwarding(self):
    run_options = config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.FULL_TRACE)
    run_metadata = config_pb2.RunMetadata()

    with self.test_session() as sess:
      with ops.device("/cpu:0"):
        c = constant_op.constant(2)
        i0 = constant_op.constant(0)
        r = control_flow_ops.while_loop(lambda i: i < 1000,
                                        lambda i: math_ops.square(c) + i, [i0])
      r_val = sess.run(r, options=run_options, run_metadata=run_metadata)
      self.assertEqual(1000, r_val)
      self.assertTrue(run_metadata.HasField("step_stats"))
      unique_allocs = set()
      for node_stat in run_metadata.step_stats.dev_stats[0].node_stats:
        for output in node_stat.output:
          unique_allocs.add(
              output.tensor_description.allocation_description.ptr)
      # Prior to cl/147536680, the number of unique allocations was about 1005.
      self.assertLess(len(unique_allocs), 756)

  def _testWhile_Gpu_1(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      n = constant_op.constant(1.0)
      c = lambda x: math_ops.less(x, 10.0)
      b = lambda x: math_ops.add(x, 1.0)
      r = control_flow_ops.while_loop(c, b, [n])
      self.assertAllClose(10.0, r.eval())

  def testWhile_Gpu_1(self):
    self._testWhile_Gpu_1(use_gpu=False)
    self._testWhile_Gpu_1(use_gpu=True)

  def _testWhile_Gpu_2(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      n = constant_op.constant(1.0)
      c = lambda x: math_ops.less(x, 10.0)

      def b(x):
        with ops.device("/cpu:0"):
          return math_ops.add(x, 1.0)

      r = control_flow_ops.while_loop(c, b, [n])
      self.assertAllClose(10.0, r.eval())

  def testWhile_Gpu_2(self):
    self._testWhile_Gpu_1(use_gpu=False)
    self._testWhile_Gpu_1(use_gpu=True)

  def testWhileShape(self):
    with self.test_session():
      i = constant_op.constant(0)
      m = array_ops.ones([2, 2])
      c = lambda i, j: math_ops.less(i, 2)

      def _b(i, j):
        new_i = math_ops.add(i, 1)
        new_j = array_ops.tile(j, [2, 2])
        return [new_i, new_j]

      r = control_flow_ops.while_loop(
          c, _b, [i, m], [i.get_shape(), tensor_shape.unknown_shape()])
      r = r[1] * array_ops.ones([8, 8])
      self.assertAllEqual(np.ones((8, 8)), r.eval())

  def testWhileWithNonTensorInput_Scalar(self):
    with self.test_session():
      n = 0
      c = lambda x: x < 10000
      b = lambda x: x + 1
      r = control_flow_ops.while_loop(c, b, [n], parallel_iterations=20)
      self.assertEqual(10000, r.eval())

  def testWhileWithNonTensorInput_Vector(self):
    with self.test_session():
      n = np.array([0])  # Note, [0] would not work here; that is a list
      c = lambda x: x[0] < 10000
      b = lambda x: array_ops.stack([x[0] + 1])
      r = control_flow_ops.while_loop(c, b, [n], parallel_iterations=20)
      self.assertEqual([10000], r.eval())

  def testWhileShapeInference(self):
    with self.test_session():
      i = constant_op.constant(0)
      m = array_ops.ones([2, 2])
      c = lambda i, j: math_ops.less(i, 2)

      def b(i, j):
        new_i = math_ops.add(i, 1)
        new_j = array_ops.concat([j, j], 0)
        return [new_i, new_j]

      r = control_flow_ops.while_loop(
          c, b, [i, m], [i.get_shape(), tensor_shape.TensorShape([None, 2])])
      self.assertTrue(r[1].get_shape()[0].value is None)
      self.assertEqual(r[1].get_shape()[1], tensor_shape.Dimension(2))

      with self.assertRaisesRegexp(ValueError, "not an invariant for"):
        r = control_flow_ops.while_loop(c, b, [i, m])

  def testWhileShapeInferenceSparseTensor(self):
    with self.test_session():
      values = constant_op.constant([2.0, 4.0], name="values")
      indices = constant_op.constant(
          [[0], [3]], dtype=dtypes.int64, name="indices")
      shape = constant_op.constant([10], dtype=dtypes.int64, name="dense_shape")
      i = constant_op.constant(0)
      x = sparse_tensor.SparseTensor(indices, values, dense_shape=shape)

      def c(i, _):
        return i < 10

      def b(i, x):
        return [
            i + 1, sparse_tensor.SparseTensor(x.indices, x.values * 2.0,
                                              x.dense_shape)
        ]

      _, r = control_flow_ops.while_loop(c, b, [i, x])
      self.assertEqual(r.dense_shape.get_shape()[0].value, 1)

      _, r = control_flow_ops.while_loop(
          c, b, [i, x], [i.get_shape(), tensor_shape.TensorShape([None])])
      self.assertTrue(r.dense_shape.get_shape()[0].value is None)

      with self.assertRaisesRegexp(ValueError, "is not compatible with"):
        _, r = control_flow_ops.while_loop(
            c, b, [i, x], [i.get_shape(), tensor_shape.TensorShape([5])])

  def testWhileShapeInferenceIndexedSlices(self):
    with self.test_session():
      values = constant_op.constant([[2.0, 4.0], [3.0, 5.0]], name="values")
      indices = constant_op.constant([0, 3], name="indices")
      shape = constant_op.constant([10, 2], name="dense_shape")
      i = constant_op.constant(0)
      x = ops.IndexedSlices(values, indices, dense_shape=shape)

      def c(i, _):
        return i < 10

      def b(i, x):
        return [
            i + 1, ops.IndexedSlices(x.values * 2.0, x.indices, x.dense_shape)
        ]

      _, r = control_flow_ops.while_loop(c, b, [i, x])
      self.assertEqual(r.dense_shape.get_shape()[0].value, 2)
      self.assertEqual(r.values.get_shape(), tensor_shape.TensorShape([2, 2]))

      _, r = control_flow_ops.while_loop(
          c, b, [i, x], [i.get_shape(), tensor_shape.TensorShape([None, 2])])
      self.assertEqual(r.dense_shape.get_shape()[0].value, 2)
      self.assertTrue(r.values.get_shape()[0].value is None)
      self.assertEqual(r.values.get_shape()[1].value, 2)

      with self.assertRaisesRegexp(ValueError, "is not compatible with"):
        _, r = control_flow_ops.while_loop(
            c, b, [i, x], [i.get_shape(), tensor_shape.TensorShape([None, 5])])

  def _testNestedWhile_1(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      n = constant_op.constant(0)

      def cpu_sum(s):
        c = lambda i, s: math_ops.less(i, 10)

        def b(i, s):
          i1 = math_ops.add(i, 1)
          with ops.device("/cpu:0"):
            s1 = math_ops.add(i, s)
          return i1, s1

        _, r_s = control_flow_ops.while_loop(c, b, [n, s])
        return r_s

      c = lambda x: math_ops.less(x, 200)
      b = lambda x: math_ops.add(x, cpu_sum(n))
      r = control_flow_ops.while_loop(c, b, [n])
      self.assertEqual(225, r.eval())

  def testNestedWhile_1(self):
    self._testNestedWhile_1(use_gpu=False)
    self._testNestedWhile_1(use_gpu=True)

  def _testNestedWhile_2(self, use_gpu):
    # Test the cases that A -> Enter and Exit -> A are partitioned.
    with self.test_session(use_gpu=use_gpu):
      s0 = constant_op.constant(2.0)

      def inner_loop(s):
        c = lambda s: math_ops.less(s, 20.0)

        def b(s):
          s1 = math_ops.add(s, s)
          return s1

        r_s = control_flow_ops.while_loop(c, b, [s], parallel_iterations=1)
        return r_s

      outer_c = lambda x: math_ops.less(x, 3000.0)

      def outer_b(x):
        x = logging_ops.Print(x, [x])  # Edge "Print -> Enter" is partitioned
        x = inner_loop(x)
        with ops.device("/cpu:0"):
          x = math_ops.square(x)  # Edge "Exit -> Square" is partitioned
        return x

      r = control_flow_ops.while_loop(
          outer_c, outer_b, [s0], parallel_iterations=1)
      self.assertEqual(1048576.0, r.eval())

  def testNestedWhile_2(self):
    self._testNestedWhile_2(use_gpu=False)
    self._testNestedWhile_2(use_gpu=True)

  def testWhileWithControl_1(self):
    with self.test_session():
      n = constant_op.constant(0)
      r = constant_op.constant(0)
      condition = lambda n_, r_: math_ops.less(n_, 10)

      def body(n_, r_):
        n_ = math_ops.add(n_, 1)
        with r_.graph.control_dependencies([r_]):
          r_ = constant_op.constant(12)
        return [n_, r_]

      res = control_flow_ops.while_loop(
          condition, body, [n, r], parallel_iterations=1)
      self.assertAllEqual(12, res[1].eval())

  def testWhileWithControl_2(self):
    with self.test_session():
      r = constant_op.constant(0)
      condition = lambda r_: math_ops.less(r_, 10)

      def body(r_):
        with r_.graph.control_dependencies([r_]):
          r_ = constant_op.constant(12)
        return [r_]

      res = control_flow_ops.while_loop(
          condition, body, [r], parallel_iterations=1)
      self.assertAllEqual(12, res.eval())

  def testWhileWithControl_3(self):
    with self.test_session() as sess:
      b = array_ops.placeholder(dtypes.bool)
      c = constant_op.constant(1)
      x0 = constant_op.constant(0)
      with ops.control_dependencies([b]):
        r = control_flow_ops.while_loop(lambda x: x < 10, lambda x: x + c, [x0])
      self.assertEqual(10, sess.run(r, {b: True}))

  def testWhileWithControl_4(self):
    with self.test_session() as sess:
      b = array_ops.placeholder(dtypes.bool)
      c = constant_op.constant(1)
      x0 = constant_op.constant(0)
      with ops.control_dependencies([b]):
        r = control_flow_ops.while_loop(
            lambda x: x < 10, lambda x: x + array_ops.identity(c), [x0])
      self.assertEqual(10, sess.run(r, {b: True}))

  def testWhileWithControl_5(self):
    with self.test_session() as sess:
      b = array_ops.placeholder(dtypes.bool)
      c = constant_op.constant(1)
      x0 = constant_op.constant(0)

      def body(x):
        with ops.control_dependencies([b]):
          return x + c

      r = control_flow_ops.while_loop(lambda x: x < 10, body, [x0])
      self.assertEqual(10, sess.run(r, {b: True}))

  def testWhileCondWithControl(self):
    # Ensure that no control edges by an outer control dependency context are
    # added to nodes inside cond/while contexts.
    with self.test_session() as sess:
      const_true = lambda: constant_op.constant(True)
      const_false = lambda: constant_op.constant(False)
      cond = lambda i: control_flow_ops.cond(i > 0, const_true, const_false)
      body = lambda i: control_flow_ops.cond(i > 0, lambda: i - 1, lambda: i)

      with ops.control_dependencies([control_flow_ops.no_op()]):
        loop = control_flow_ops.while_loop(cond, body,
                                           (constant_op.constant(5),))
      self.assertEqual(0, sess.run(loop))

  def testWhileCondExitControl(self):
    with self.test_session():
      v = variables.Variable(1)

      def false_branch():
        cond = lambda i: i < 100

        def body(i):
          x = state_ops.assign(v, i)
          return x + 1

        loop = control_flow_ops.while_loop(cond, body, [0])
        # Make sure to handle correctly control edge from Exit to a node.
        with ops.control_dependencies([loop]):
          return constant_op.constant(6.0)

      r = control_flow_ops.cond(
          constant_op.constant(False), lambda: constant_op.constant(1.0),
          false_branch)
      variables.global_variables_initializer().run()
      self.assertEqual(6.0, r.eval())
      self.assertEqual(99, v.eval())

  def testCondWhile_1(self):
    with self.test_session():
      n = ops.convert_to_tensor(0, name="n")
      c = lambda x: math_ops.less(x, 10)
      b = lambda x: math_ops.add(x, 1)
      r = control_flow_ops.cond(
          math_ops.less(0, 1), lambda: control_flow_ops.while_loop(c, b, [n]),
          lambda: n)
      self.assertAllEqual(10, r.eval())

  def testCondWhile_2(self):
    with self.test_session():
      n = ops.convert_to_tensor(0)
      c = lambda x: math_ops.less(x, 10)
      b = lambda x: math_ops.add(x, 1)
      r = control_flow_ops.cond(
          math_ops.less(1, 0), lambda: math_ops.add(n, 1),
          lambda: control_flow_ops.while_loop(c, b, [n]))
      self.assertAllEqual(10, r.eval())

  def _testCondWhile_3(self, use_gpu):
    with self.test_session(use_gpu=use_gpu) as sess:
      p = array_ops.placeholder(dtypes.bool)
      n = constant_op.constant(0.0)

      def c(x):
        return math_ops.less(x, 10.0)

      def b(x):
        with ops.device("/cpu:0"):
          x1 = math_ops.add(x, 1.0)
        return x1

      r = control_flow_ops.cond(p,
                                lambda: control_flow_ops.while_loop(c, b, [n]),
                                lambda: math_ops.multiply(n, 2.0))
      r1 = gradients_impl.gradients(r, [n])
      self.assertEqual(10, sess.run(r, {p: True}))
      self.assertEqual([1.0], sess.run(r1, {p: True}))
      self.assertEqual(0.0, sess.run(r, {p: False}))
      self.assertEqual([2.0], sess.run(r1, {p: False}))

  def testCondWhile_3(self):
    self._testCondWhile_3(use_gpu=False)
    self._testCondWhile_3(use_gpu=True)

  def testWhileCond_1(self):
    with self.test_session():
      i = ops.convert_to_tensor(0, name="i")
      n = ops.convert_to_tensor(10, name="n")
      one = ops.convert_to_tensor(1, name="one")
      c = lambda x: math_ops.less(x, n)
      # pylint: disable=undefined-variable
      # for OSS build
      b = lambda x: control_flow_ops.cond(
          constant_op.constant(True),
          lambda: math_ops.add(x, one), lambda: math_ops.subtract(x, one))
      # pylint: enable=undefined-variable
      r = control_flow_ops.while_loop(c, b, [i])
      self.assertAllEqual(10, r.eval())

  def testWhileCond_2(self):
    with self.test_session():
      n = ops.convert_to_tensor(0, name="n")
      c = lambda x: math_ops.less(x, 10)
      b = lambda x: control_flow_ops.cond(constant_op.constant(True), lambda: math_ops.add(x, 1), lambda: n)
      r = control_flow_ops.while_loop(c, b, [n])
      self.assertAllEqual(10, r.eval())

  def testWhileCond_3(self):
    with self.test_session():
      n = ops.convert_to_tensor(0)
      c = lambda x: math_ops.less(x, 10)
      # pylint: disable=undefined-variable
      # for OSS build
      b = lambda x: control_flow_ops.cond(math_ops.less(0, 1),
                                          lambda: math_ops.add(x, 1),
                                          lambda: math_ops.subtract(x, 1))
      # pylint: enable=undefined-variable
      r = control_flow_ops.while_loop(c, b, [n])
      self.assertAllEqual(10, r.eval())

  # NOTE: It is ok to have parallel_iterations > 1
  def testWhileUpdateVariable_1(self):
    with self.test_session():
      select = variables.Variable([3.0, 4.0, 5.0])
      n = constant_op.constant(0)

      def loop_iterator(j):
        return math_ops.less(j, 3)

      def loop_body(j):
        ns = state_ops.scatter_update(select, j, 10.0)
        nj = math_ops.add(j, 1)
        op = control_flow_ops.group(ns)
        nj = control_flow_ops.with_dependencies([op], nj)
        return [nj]

      r = control_flow_ops.while_loop(
          loop_iterator, loop_body, [n], parallel_iterations=1)
      self.assertTrue(check_op_order(n.graph))
      variables.global_variables_initializer().run()
      self.assertEqual(3, r.eval())
      result = select.eval()
      self.assertAllClose(np.array([10.0, 10.0, 10.0]), result)

  def testWhileUpdateVariable_2(self):
    with self.test_session():
      select1 = variables.Variable([3.0, 4.0, 5.0])
      select2 = variables.Variable([3.0, 4.0, 5.0])
      n = constant_op.constant(0)

      def loop_iterator(j):
        return math_ops.less(j, 3)

      def loop_body(j):
        ns1 = state_ops.scatter_update(select1, j, 10.0)
        ns2 = state_ops.scatter_update(select2, j, 10.0)
        nj = math_ops.add(j, 1)
        op = control_flow_ops.group(ns1, ns2)
        nj = control_flow_ops.with_dependencies([op], nj)
        return [nj]

      r = control_flow_ops.while_loop(
          loop_iterator, loop_body, [n], parallel_iterations=1)
      self.assertTrue(check_op_order(n.graph))
      variables.global_variables_initializer().run()
      self.assertEqual(3, r.eval())
      result1 = select1.eval()
      self.assertAllClose(np.array([10.0, 10.0, 10.0]), result1)
      result2 = select2.eval()
      self.assertAllClose(np.array([10.0, 10.0, 10.0]), result2)

  def testWhileUpdateVariable_3(self):
    with self.test_session():
      select = variables.Variable([3.0, 4.0, 5.0])
      n = constant_op.constant(0)

      def loop_iterator(j, _):
        return math_ops.less(j, 3)

      def loop_body(j, _):
        ns = state_ops.scatter_update(select, j, 10.0)
        nj = math_ops.add(j, 1)
        return [nj, ns]

      r = control_flow_ops.while_loop(
          loop_iterator,
          loop_body, [n, array_ops.identity(select)],
          parallel_iterations=1)
      variables.global_variables_initializer().run()
      result = r[1].eval()
    self.assertTrue(check_op_order(n.graph))
    self.assertAllClose(np.array([10.0, 10.0, 10.0]), result)

  # b/24814703
  def testWhileUpdateVariable_4(self):
    with self.test_session():
      var_a = variables.Variable(0, name="a")
      var_b = variables.Variable(0, name="b")
      variables.global_variables_initializer().run()

      c = constant_op.constant(0, name="c")
      asn1 = state_ops.assign_add(var_a, 1, name="a_add")

      # Loop condition
      def pred(i):
        return math_ops.less(i, 10)

      # Loop body
      def loop_body(i):
        asn2 = state_ops.assign_add(var_b, asn1, name="b_add")
        with ops.control_dependencies([asn2]):
          ni = math_ops.add(i, 1, name="i_add")
        return ni

      lpa = control_flow_ops.while_loop(
          pred, loop_body, [c], parallel_iterations=1)

      self.assertEqual(0, var_b.eval())
      lpa.eval()  # Run the loop
      self.assertEqual(10, var_b.eval())

  # b/24736492
  def testWhileUpdateVariable_5(self):
    with self.test_session():
      # Create some variables.
      var_a = variables.Variable(0, name="a")
      var_b = variables.Variable(0, name="b")
      variables.global_variables_initializer().run()

      # Change condition to check var_b
      def pred(_):
        return math_ops.less(var_b, 10)

      # Change body to increment var_b
      def loop_body(i):
        asn1 = state_ops.assign_add(
            var_a, constant_op.constant(1), name="a_add")
        asn2 = state_ops.assign_add(
            var_b, constant_op.constant(1), name="b_add")
        with ops.control_dependencies([asn1, asn2]):
          inc_b = array_ops.identity(var_b)
        return inc_b

      lpa = control_flow_ops.while_loop(
          pred, loop_body, [var_b], parallel_iterations=1, name="loop")

      self.assertEqual(0, var_b.eval())
      lpa.eval()  # Run the loop
      self.assertEqual(10, var_a.eval())
      self.assertEqual(10, var_b.eval())

  # b/24814668
  def testWhileUpdateVariable_6(self):
    with self.test_session():
      # Create some variables.
      var_a = variables.Variable(0, name="a")
      var_b = variables.Variable(0, name="b")
      c = constant_op.constant(0)
      variables.global_variables_initializer().run()

      # Loop condition
      def pred(i):
        return math_ops.less(i, 10)

      # Loop body
      def loop_body(i):
        asn1 = state_ops.assign_add(var_a, 1, name="a_add")
        with ops.control_dependencies([asn1]):
          asn2 = state_ops.assign_add(var_b, var_a, name="b_add")
        with ops.control_dependencies([asn2]):
          ni = math_ops.add(i, 1, name="i_add")
          return ni

      lpa = control_flow_ops.while_loop(
          pred, loop_body, [c], parallel_iterations=1, name="loop")

      self.assertEqual(0, var_b.eval())
      lpa.eval()  # Run the loop
      self.assertEqual(55, var_b.eval())
      self.assertEqual(10, var_a.eval())

  def testWhileQueue_1(self):
    with self.test_session():
      q = data_flow_ops.FIFOQueue(-1, dtypes.int32)
      i = constant_op.constant(0)

      def c(i):
        return math_ops.less(i, 10)

      def b(i):
        ni = math_ops.add(i, 1)
        ni = control_flow_ops.with_dependencies([q.enqueue((i,))], ni)
        return ni

      r = control_flow_ops.while_loop(c, b, [i], parallel_iterations=1)
      self.assertEqual([10], r.eval())
      for i in xrange(10):
        self.assertEqual([i], q.dequeue().eval())

  def testWhileStack_1(self):
    with self.test_session():
      s = gen_data_flow_ops._stack(dtypes.int32, stack_name="foo")
      i = constant_op.constant(0)

      def c(i):
        return math_ops.less(i, 10)

      def b(i):
        ni = math_ops.add(i, 1)
        ni = control_flow_ops.with_dependencies(
            [gen_data_flow_ops._stack_push(s, i)], ni)
        return ni

      r = control_flow_ops.while_loop(c, b, [i], parallel_iterations=1)

      x = constant_op.constant(0)

      def c1(i, _):
        return math_ops.greater(i, 0)

      def b1(i, x):
        ni = math_ops.subtract(i, 1)
        nx = x + gen_data_flow_ops._stack_pop(s, dtypes.int32)
        return [ni, nx]

      _, rx = control_flow_ops.while_loop(
          c1,
          b1, [r, x], [r.get_shape(), tensor_shape.unknown_shape()],
          parallel_iterations=1)
      self.assertEqual(45, rx.eval())

  def _testWhileGrad_ColocateGradients(self, colocate):
    gpu_dev_name = test.gpu_device_name().lower() if test.is_gpu_available(
    ) else "/gpu:0"
    gpu_short_name = gpu_dev_name.split("/")[-1]

    with self.test_session(graph=ops.Graph()) as sess:
      v = constant_op.constant(2.0, name="v")
      c = lambda v: math_ops.less(v, 100.0)

      def b(x):
        with ops.device(gpu_dev_name):
          return math_ops.square(x)

      loop = control_flow_ops.while_loop(c, b, [v], parallel_iterations=1)
      r = gradients_impl.gradients(
          loop, v, colocate_gradients_with_ops=colocate)[0]
    r_ops = r.graph.get_operations()
    r_devices = [(op.name, op.device.lower()) for op in r_ops]

    self.assertTrue(any("Square" in op.name for op in r_ops))

    for (name, dev) in r_devices:
      if not colocate and name.endswith("Square"):
        # Only forward graph contain gpu in Square device
        self.assertTrue(gpu_short_name in dev)
      elif colocate and "Square" in name:
        # Forward and backward graphs contain gpu in Square/Square_grad devices
        self.assertTrue(gpu_short_name in dev)
      else:
        self.assertFalse(gpu_short_name in dev)
    self.assertAllClose(1024.0, sess.run(r))

  def testWhileGrad_ColocateGradients(self):
    self._testWhileGrad_ColocateGradients(colocate=False)
    self._testWhileGrad_ColocateGradients(colocate=True)

  def testWhileGrad_Square(self):
    with self.test_session():
      v = constant_op.constant(2.0, name="v")
      c = lambda v: math_ops.less(v, 100.0)
      b = math_ops.square
      r = control_flow_ops.while_loop(c, b, [v], parallel_iterations=1)
      r = control_flow_ops.cond(math_ops.less(1, 2), lambda: r, lambda: v)

      r = gradients_impl.gradients(r, v)[0]
      self.assertAllClose(1024.0, r.eval())

  def testWhileGrad_Shape(self):
    with self.test_session():
      x = array_ops.placeholder(dtypes.float32, shape=[None])
      v = constant_op.constant([2.0], name="v")
      n = constant_op.constant(0, name="n")
      c = lambda i, v: math_ops.less(i, 5)
      b = lambda i, v: [i + 1, math_ops.multiply(x, v)]
      r = control_flow_ops.while_loop(
          c,
          b, [n, v], [n.get_shape(), tensor_shape.unknown_shape()],
          parallel_iterations=1)

      r = gradients_impl.gradients(r[1], x)[0]
      self.assertEqual([None], r.get_shape().as_list())
      self.assertAllClose([810.0, 2560.0], r.eval(feed_dict={x: [3.0, 4.0]}))

  def testWhileGrad_BaseShape(self):
    with self.test_session() as sess:
      x = array_ops.placeholder(dtypes.float32, [None])
      v0 = constant_op.constant([2.0, 2.0], name="v")
      c = lambda v: constant_op.constant(False)
      b = lambda v: math_ops.multiply(v, x)
      r = control_flow_ops.while_loop(c, b, [v0])
      y = math_ops.square(x)

      r = gradients_impl.gradients([r, y], x)[0]
      self.assertAllClose([2.0, 4.0], sess.run(r, feed_dict={x: [1.0, 2.0]}))

  def testWhileGrad_MultipleUses(self):
    with self.test_session():
      v = constant_op.constant(2.0, name="v")
      c = lambda v: math_ops.less(v, 100.0)
      b = math_ops.square
      r = control_flow_ops.while_loop(c, b, [v], parallel_iterations=1)
      r = math_ops.multiply(r, r)

      r = gradients_impl.gradients(r, v)[0]
      self.assertEqual(524288.0, r.eval())

  def testWhileGrad_LoopAdd(self):
    with self.test_session():
      v = constant_op.constant(2.0, name="v")
      c = lambda v: math_ops.less(v, 100.0)
      b = math_ops.square
      r = control_flow_ops.while_loop(c, b, [v], parallel_iterations=1)
      r = math_ops.add(r, r)

      r = gradients_impl.gradients(r, v)[0]
      self.assertAllClose(2048.0, r.eval())

  def _testWhileGrad_Mul(self, use_gpu, p_iters):
    with self.test_session(use_gpu=use_gpu) as sess:
      a = constant_op.constant(3.0, name="a")
      v = constant_op.constant(2.0, name="v")
      c = lambda v: math_ops.less(v, 100.0)
      b = lambda v: math_ops.multiply(v, a)
      r = control_flow_ops.while_loop(c, b, [v], parallel_iterations=p_iters)

      grad_a, grad_v = gradients_impl.gradients(r, [a, v])
      grad_a_val, grad_v_val = sess.run([grad_a, grad_v])
      self.assertAllClose(216.0, grad_a_val)
      self.assertAllClose(81.0, grad_v_val)

  def testWhileGrad_Mul(self):
    self._testWhileGrad_Mul(use_gpu=False, p_iters=1)
    self._testWhileGrad_Mul(use_gpu=False, p_iters=10)
    self._testWhileGrad_Mul(use_gpu=True, p_iters=1)
    self._testWhileGrad_Mul(use_gpu=True, p_iters=10)

  def _testNestedWhileCondWhileGrad(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      v = constant_op.constant(1.0)

      def inner_loop(s):
        z = constant_op.constant(0)
        c = lambda i, x: math_ops.less(i, 4)
        b = lambda i, x: [math_ops.add(i, 1), math_ops.multiply(x, 2.0)]
        return control_flow_ops.while_loop(c, b, [z, s])

      c = lambda x: math_ops.less(x, 128.0)

      def b(x):
        return control_flow_ops.cond(
            constant_op.constant(True),
            lambda: math_ops.square(inner_loop(x)[1]),
            lambda: math_ops.multiply(x, 2.0))

      r = control_flow_ops.while_loop(c, b, [v])
      r = gradients_impl.gradients(r, v)[0]
      self.assertAllClose(512.0, r.eval())

  def testNestedWhileCondWhileGrad(self):
    self._testNestedWhileCondWhileGrad(use_gpu=False)
    self._testNestedWhileCondWhileGrad(use_gpu=True)

  def testWhileGrad_Variable(self):
    with self.test_session():
      a = variables.Variable(3.0)
      v = constant_op.constant(2.0, name="v")
      c = lambda v: math_ops.less(v, 100.0)
      b = lambda v: math_ops.multiply(v, a)
      r = control_flow_ops.while_loop(c, b, [v], parallel_iterations=1)

      r = gradients_impl.gradients(r, a)
      variables.global_variables_initializer().run()
      self.assertAllClose(216.0, r[0].eval())

  def testWhileGradInCond(self):
    with self.test_session():
      n = ops.convert_to_tensor(1.0, name="n")
      x = array_ops.placeholder(dtypes.float32, shape=None)
      c = lambda n: math_ops.less(n, 10.0)
      b = lambda n: math_ops.add(n, x)

      def fn1():
        r = control_flow_ops.while_loop(c, b, [n],
                                        [tensor_shape.unknown_shape()])
        return gradients_impl.gradients(r, x)

      r = control_flow_ops.cond(math_ops.less(1, 2), fn1, lambda: x)
      self.assertAllClose(9.0, r.eval(feed_dict={x: 1.0}))

  def testWhileGradInWhile(self):
    with self.test_session():
      n = ops.convert_to_tensor(1.0, name="n")
      x = array_ops.placeholder(dtypes.float32, shape=None)
      c = lambda n: math_ops.less(n, 10.0)
      b = lambda n: math_ops.add(n, x)

      def b1(n):
        r = control_flow_ops.while_loop(c, b, [n],
                                        [tensor_shape.unknown_shape()])
        return gradients_impl.gradients(r, x)

      r = control_flow_ops.while_loop(lambda n: n < 6.0, b1, [n],
                                      [tensor_shape.unknown_shape()])
      self.assertAllClose(9.0, r.eval(feed_dict={x: 1.0}))

  def testWhile_NestedInput(self):
    with self.test_session() as sess:
      named = collections.namedtuple("named", ("a", "b"))
      loop_vars = [
          named(a=constant_op.constant(0.0), b=constant_op.constant(1.0)),
          (constant_op.constant(2.0),
           constant_op.constant(3.0)), constant_op.constant(4.0)
      ]
      c = lambda lv0, _1, _2: lv0.a < 100.0

      def b(lv0, lv1, lv2):
        lv0 = named(a=lv0.a + 1, b=lv0.b)
        lv1 = (lv1[0] + 1, lv1[1])
        lv2 += 2
        return [lv0, lv1, lv2]

      r = control_flow_ops.while_loop(c, b, loop_vars)

      self.assertTrue(isinstance(r, list))
      self.assertTrue(isinstance(r[0], named))
      self.assertTrue(isinstance(r[1], tuple))
      self.assertTrue(isinstance(r[2], ops.Tensor))

      r_flattened = nest.flatten(r)
      self.assertEqual([100.0, 1.0, 102.0, 3.0, 4.0 + 100 * 2.0],
                       sess.run(r_flattened))

  def testWhile_NestedBadArityFails(self):
    with self.test_session():
      named = collections.namedtuple("named", ("a", "b"))
      loop_vars = [
          named(a=constant_op.constant(0.0), b=constant_op.constant(1.0)),
          (constant_op.constant(2.0),
           constant_op.constant(3.0)), constant_op.constant(4.0)
      ]
      c = lambda lv0, _1, _2: lv0.a < 100.0

      def b(lv0, lv1, _):
        return [lv0, lv1]

      with self.assertRaisesRegexp(ValueError, "the same number of elements"):
        control_flow_ops.while_loop(c, b, loop_vars)

  def testWhileGrad_ys_xs(self):
    with self.test_session():
      x = constant_op.constant(3.0, name="x")
      y = constant_op.constant(2.0, name="y")

      c = lambda x, y: math_ops.less(x, 100.0)

      def b(x, y):
        y1 = math_ops.add(x, y)
        x1 = math_ops.multiply(x, y1)
        return x1, y1

      rx, ry = control_flow_ops.while_loop(c, b, [x, y], parallel_iterations=1)

      r = gradients_impl.gradients([rx, ry], x)
      self.assertAllClose(304.0, r[0].eval())
      r = gradients_impl.gradients([rx, ry], y)
      self.assertAllClose(124.0, r[0].eval())
      r = gradients_impl.gradients([rx], x)
      self.assertAllClose(295.0, r[0].eval())
      r = gradients_impl.gradients([rx], y)
      self.assertAllClose(120.0, r[0].eval())

  def testWhileGrad_Dependency(self):
    with self.test_session():
      i = constant_op.constant(0, name="i")
      x = constant_op.constant(2.0, name="x")

      c = lambda i, x: math_ops.less(i, 10)

      def b(i, x):
        x = math_ops.multiply(x, 2.0)
        i = math_ops.add(i, 1)
        return i, x

      ri, rx = control_flow_ops.while_loop(c, b, [i, x], parallel_iterations=1)

      r = gradients_impl.gradients([ri, rx], x)
      self.assertAllClose(1024.0, r[0].eval())
      r = gradients_impl.gradients([rx], x)
      self.assertAllClose(1024.0, r[0].eval())

  def testWhileGrad_NoGradient(self):
    with self.test_session():
      v = constant_op.constant(2.0, name="v")
      c = lambda v: math_ops.less(v, 100.0)
      b = math_ops.square
      r = control_flow_ops.while_loop(c, b, [v], back_prop=False)
      r = math_ops.add(r, v)
      r = gradients_impl.gradients(r, v)
      self.assertAllClose(1.0, r[0].eval())

  def testWhileGrad_NoDependency(self):
    with self.test_session() as sess:
      variable = variables.Variable(array_ops.ones([2, 3]))
      time = array_ops.zeros([], dtype=dtypes.int32)

      def cond(time, tensor, _):
        return time < 10

      def body(time, tensor, _):
        return (time + 1, tensor, tensor)

      loop_vars = [time, variable, variable]
      tensors = control_flow_ops.while_loop(
          cond=cond, body=body, loop_vars=loop_vars)
      cost = math_ops.reduce_sum(tensors[2])
      grad = gradients_impl.gradients(cost, [variable])
      variables.global_variables_initializer().run()
      self.assertAllClose(np.ones([2, 3]), sess.run(grad[0]))

  def testWhileGrad_Const(self):
    with self.test_session() as sess:
      c0 = constant_op.constant(0.0, name="c0")
      c1 = constant_op.constant(1.0, name="c1")
      time = constant_op.constant(0, name="t")

      def cond(time, _):
        return time < 1

      def body(time, tensor):
        return time + 1, c1

      loop_vars = [time, c0]
      tensors = control_flow_ops.while_loop(
          cond=cond, body=body, loop_vars=loop_vars)
      cost = math_ops.reduce_sum(tensors[1])
      grad = gradients_impl.gradients(cost, [c0])
      self.assertAllClose(0.0, sess.run(grad[0]))

  def testWhileGrad_SerialTwoLoops(self):
    with self.test_session():
      i = constant_op.constant(0, name="i")
      x = constant_op.constant(2.0, name="x")

      c = lambda i, x: math_ops.less(i, 5)

      def b(i, x):
        x = math_ops.multiply(x, 2.0)
        i = math_ops.add(i, 1)
        return i, x

      _, rx = control_flow_ops.while_loop(c, b, [i, x], parallel_iterations=1)
      _, rx = control_flow_ops.while_loop(c, b, [i, rx], parallel_iterations=1)

      r = gradients_impl.gradients([rx], x)
      self.assertAllClose(1024.0, r[0].eval())

  def testWhileGrad_ParallelTwoLoops(self):
    with self.test_session():
      i = constant_op.constant(0, name="i")
      x = constant_op.constant(2.0, name="x")

      c = lambda i, x: math_ops.less(i, 5)

      def b(i, x):
        x = math_ops.multiply(x, 2.0)
        i = math_ops.add(i, 1)
        return i, x

      _, r1 = control_flow_ops.while_loop(c, b, [i, x], parallel_iterations=1)
      _, r2 = control_flow_ops.while_loop(c, b, [i, x], parallel_iterations=1)
      rx = math_ops.add(r1, r2)

      r = gradients_impl.gradients([rx], x)
      self.assertAllClose(64.0, r[0].eval())

  def testWhileGrad_OneOutputWithControlDependencyOnSecond(self):
    with self.test_session():
      i = constant_op.constant(0, name="i")
      x = constant_op.constant(1.0, name="x")
      y = constant_op.constant(1.0, name="y")
      c = lambda i, *_: math_ops.less(i, 1, name="cond_less")

      def b(i, xi, yi):
        # return (i + 1, xi, xi + yi)
        return (math_ops.add(i, 1, name="inc"), array_ops.identity(
            xi, name="xi"), math_ops.add(xi, yi, name="xi_plus_yi"))

      _, x_f, y_f = control_flow_ops.while_loop(c, b, [i, x, y])
      with ops.control_dependencies([x_f]):
        y_f_d = array_ops.identity(y_f, name="y_f_d")

      self.assertAllClose(2.0, y_f_d.eval())  # y_f_d = 1.0 + 1.0
      g = gradients_impl.gradients([y_f_d], [x])[0]
      self.assertTrue(g is not None)
      self.assertAllClose(1.0, g.eval())  # y_f_d = x + 1.0, dy_f_d/dx = 1.0

  def _testNestedWhileGrad_Simple(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      v = constant_op.constant(1.0)

      def inner_loop(s):
        c = lambda x: math_ops.less(x, 4.0)
        b = lambda x: math_ops.multiply(x, 2.0)
        return control_flow_ops.while_loop(c, b, [s])

      c = lambda x: math_ops.less(x, 2.0)
      b = lambda x: math_ops.multiply(inner_loop(x), 2.0)
      r = control_flow_ops.while_loop(c, b, [v])

      r = gradients_impl.gradients(r, v)[0]
      self.assertAllClose(8.0, r.eval())

  def testNestedWhileGrad_Simple(self):
    self._testNestedWhileGrad_Simple(use_gpu=False)
    self._testNestedWhileGrad_Simple(use_gpu=True)

  def testNestedWhileGrad_SerialInner(self):
    with self.test_session():
      v = constant_op.constant(1.0)

      def inner_loop1(s):
        z = constant_op.constant(0)
        c = lambda i, x: math_ops.less(i, 4)
        b = lambda i, x: [math_ops.add(i, 1), math_ops.multiply(x, 2.0)]
        return control_flow_ops.while_loop(c, b, [z, s])

      def inner_loop2(s):
        z = constant_op.constant(0)
        c = lambda i, x: math_ops.less(i, 4)
        b = lambda i, x: [math_ops.add(i, 1), math_ops.multiply(x, 2.0)]
        return control_flow_ops.while_loop(c, b, [z, s])

      c = lambda x: math_ops.less(x, 128.0)
      b = lambda x: inner_loop2(inner_loop1(x)[1])[1]
      r = control_flow_ops.while_loop(c, b, [v])

      r = gradients_impl.gradients(r, v)[0]
      self.assertAllClose(256.0, r.eval())

  def testNestedWhileGrad_ParallelInner(self):
    with self.test_session():
      v = constant_op.constant(1.0)

      def inner_loop1(s):
        z = constant_op.constant(0)
        c = lambda i, x: math_ops.less(i, 4)
        b = lambda i, x: [math_ops.add(i, 1), math_ops.multiply(x, 2.0)]
        return control_flow_ops.while_loop(c, b, [z, s])

      def inner_loop2(s):
        z = constant_op.constant(0)
        c = lambda i, x: math_ops.less(i, 4)
        b = lambda i, x: [math_ops.add(i, 1), math_ops.multiply(x, 2.0)]
        return control_flow_ops.while_loop(c, b, [z, s])

      c = lambda x: math_ops.less(x, 128.0)
      b = lambda x: math_ops.multiply(inner_loop1(x)[1], inner_loop2(x)[1])
      r = control_flow_ops.while_loop(c, b, [v])

      r = gradients_impl.gradients(r, v)[0]
      self.assertAllClose(512.0, r.eval())

  def testNestedWhileGrad_ParallelIterations(self):
    # Make sure the stack pushes and pops of an inner loop are executed in
    # the sequential order of the iterations of its outer loop.
    with self.test_session() as sess:

      def inner_loop(t):
        fn = lambda n: n + math_ops.square(var)
        return functional_ops.map_fn(fn=fn, elems=t, parallel_iterations=10)

      def outer_loop(inp):
        return functional_ops.map_fn(
            fn=inner_loop, elems=inp, parallel_iterations=10)

      var = variables.Variable(constant_op.constant(3.0))
      inp = constant_op.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
      res = outer_loop(inp)
      optimizer = adam.AdamOptimizer(learning_rate=0.001)
      train_op = optimizer.minimize(math_ops.reduce_mean(math_ops.square(res)))
      sess.run(variables.global_variables_initializer())
      sess.run(train_op)
      self.assertAllClose(2.999, var.eval())

  def _testWhileCondGrad_Simple(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      v = ops.convert_to_tensor(2.0, name="v")
      n = ops.convert_to_tensor(100.0, name="n")
      one = ops.convert_to_tensor(1.0, name="one")
      c = lambda x: math_ops.less(x, n)
      # pylint: disable=undefined-variable
      # for OSS build
      b = lambda x: control_flow_ops.cond(constant_op.constant(True),
                                          lambda: math_ops.square(x),
                                          lambda: math_ops.subtract(x, one))
      # pylint: enable=undefined-variable
      r = control_flow_ops.while_loop(c, b, [v])
      r = gradients_impl.gradients(r, v)[0]
      self.assertAllClose(1024.0, r.eval())

  def testWhileCondGrad_Simple(self):
    self._testWhileCondGrad_Simple(use_gpu=False)
    self._testWhileCondGrad_Simple(use_gpu=True)

  def testWhileCondGrad_UnknownShape(self):
    with self.test_session() as sess:
      v = array_ops.placeholder(dtypes.float32)
      n = ops.convert_to_tensor(100.0, name="n")
      one = ops.convert_to_tensor(1.0, name="one")
      c = lambda x: math_ops.less(x, n)
      # pylint: disable=undefined-variable
      # for OSS build
      b = lambda x: control_flow_ops.cond(constant_op.constant(True),
                                          lambda: math_ops.square(x),
                                          lambda: math_ops.subtract(x, one))
      # pylint: enable=undefined-variable
      r = control_flow_ops.while_loop(c, b, [v])
      r = gradients_impl.gradients(r, v)[0]
      r = sess.run(r, feed_dict={v: 2.0})
      self.assertAllClose(1024.0, r)

  def testWhileGrad_Concat(self):
    with self.test_session() as sess:
      x = variable_scope.get_variable("x", initializer=[[1., 2.]])
      i0 = constant_op.constant(0)
      h0 = array_ops.zeros([0, 2])

      def condition(i, _):
        return i < 2

      def body(i, h):
        return i + 1, array_ops.concat([h, x], 0)

      _, h = control_flow_ops.while_loop(
          condition, body, [i0, h0],
          [i0.get_shape(), tensor_shape.TensorShape([None, 2])])
      s = math_ops.reduce_sum(h)

      sess.run(variables.global_variables_initializer())
      optimizer = gradient_descent.GradientDescentOptimizer(0.01)
      op = optimizer.minimize(s)
      sess.run(op)
      self.assertAllClose([[0.98000002, 1.98000002]], sess.run(x))

  def testWhileWithRefsWithGradients_1(self):
    with self.test_session() as sess:
      x = variables.Variable(0)._ref()  # pylint: disable=protected-access
      i = constant_op.constant(0)
      c = lambda i, x: math_ops.less(i, 10)

      self.assertEqual(x.dtype, dtypes.int32_ref)

      # pylint: disable=protected-access
      def body(i, x):
        self.assertEqual(x.dtype, dtypes.int32_ref)
        return [i + 1, gen_array_ops._ref_identity(x)]

      # pylint: enable=protected-access

      r = control_flow_ops.while_loop(c, body, [i, x], parallel_iterations=5)

      grad_ys = [variables.Variable(73)._ref()]  # pylint: disable=protected-access
      grad = gradients_impl.gradients([r[1]], [x], grad_ys=grad_ys)

      variables.global_variables_initializer().run()

      self.assertEqual(r[0].dtype, dtypes.int32)
      self.assertEqual(r[1].dtype, dtypes.int32_ref)

      value_i, value_x, value_x_grad = sess.run(r + grad)

    self.assertEqual(10, value_i)
    self.assertEqual(0, value_x)
    self.assertEqual(73, value_x_grad)

  def testWhileGrad_IndexedSlices(self):
    with self.test_session():
      values = constant_op.constant([2.0, 4.0], name="values")
      indices = constant_op.constant([0, 3], name="indices")
      shape = constant_op.constant([10], name="dense_shape")
      i = constant_op.constant(0)
      x = ops.IndexedSlices(values, indices, dense_shape=shape)

      def c(i, _):
        return i < 10

      def b(i, x):
        return [
            i + 1, ops.IndexedSlices(x.values * 2.0, x.indices, x.dense_shape)
        ]

      _, r = control_flow_ops.while_loop(c, b, [i, x])
      r = gradients_impl.gradients(r.values, values)[0]
      self.assertAllClose(np.array([1024.0, 1024.0]), r.eval())

  def testWhileGrad_SparseTensor(self):
    with self.test_session():
      values = constant_op.constant([2.0, 4.0], name="values")
      indices = constant_op.constant(
          [[0], [3]], dtype=dtypes.int64, name="indices")
      shape = constant_op.constant([10], dtype=dtypes.int64, name="dense_shape")
      i = constant_op.constant(0)
      x = sparse_tensor.SparseTensor(indices, values, dense_shape=shape)

      def c(i, _):
        return i < 10

      def b(i, x):
        return [
            i + 1, sparse_tensor.SparseTensor(x.indices, x.values * 2.0,
                                              x.dense_shape)
        ]

      _, r = control_flow_ops.while_loop(c, b, [i, x])
      r = gradients_impl.gradients(r.values, values)[0]
      self.assertAllClose(np.array([1024.0, 1024.0]), r.eval())

  def testCallGradInLoop(self):
    with self.test_session() as sess:
      i0 = constant_op.constant(0)
      params = constant_op.constant(5.0)
      params_1 = math_ops.square(params)

      def c(i, _):
        return i < 10

      def b(i, x):
        data = constant_op.constant([1.0, 2.0, 3.0])
        data = math_ops.multiply(data, params_1)
        x1 = x + gradients_impl.gradients(data, params)[0]
        return i + 1, x1

      output_grad = control_flow_ops.while_loop(c, b,
                                                [i0, constant_op.constant(0.0)])
      self.assertAllClose(600.0, sess.run(output_grad)[1])

  def testWhileAndTensorArray(self):
    with self.test_session() as sess:
      param = constant_op.constant(2.0)
      n0 = constant_op.constant(0)
      y0 = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="elems")

      def c(i, _):
        return i < 10

      def b(i, y):
        return [
            i + 1,
            functional_ops.map_fn(lambda x: math_ops.multiply(x, param), y)
        ]

      r = control_flow_ops.while_loop(c, b, [n0, y0], parallel_iterations=1)
      r = gradients_impl.gradients(r, param)[0]
      self.assertAllClose(107520.0, sess.run(r))

  def testWhileGrad_StopGrad(self):
    with self.test_session():
      x = constant_op.constant(3.0, name="x")
      y = constant_op.constant(2.0, name="y")

      c = lambda x, y: math_ops.less(x, 100.0)

      def b(x, y):
        y1 = math_ops.square(y)
        x1 = math_ops.add(math_ops.square(x), y1)
        return x1, y1

      rx, ry = control_flow_ops.while_loop(c, b, [x, y])

      r = gradients_impl.gradients(rx, y)[0]
      self.assertEqual(136.0, r.eval())
      r = gradients_impl.gradients(ry, y)[0]
      self.assertEqual(32.0, r.eval())

      r = gradients_impl.gradients(array_ops.stop_gradient(rx), y)[0]
      self.assertEqual(r, None)
      r = gradients_impl.gradients(array_ops.stop_gradient(ry), y)[0]
      self.assertEqual(r, None)

      r = gradients_impl.gradients(
          array_ops.stop_gradient(math_ops.square(rx)), y)[0]
      self.assertEqual(r, None)
      r = gradients_impl.gradients(
          array_ops.stop_gradient(math_ops.add(rx, ry)), x)[0]
      self.assertEqual(r, None)
      r = gradients_impl.gradients(
          array_ops.stop_gradient(math_ops.add(rx, ry)), y)[0]
      self.assertEqual(r, None)

      r = gradients_impl.gradients(math_ops.add(rx, ry), y)[0]
      self.assertEqual(168.0, r.eval())
      r = gradients_impl.gradients(
          math_ops.add(rx, array_ops.stop_gradient(ry)), y)[0]
      self.assertEqual(136.0, r.eval())
      r = gradients_impl.gradients(
          math_ops.add(array_ops.stop_gradient(rx), ry), y)[0]
      self.assertEqual(32.0, r.eval())

  def testWhileGrad_StopGradInside(self):
    with self.test_session():
      x = constant_op.constant(3.0, name="x")
      y = constant_op.constant(2.0, name="y")

      c = lambda x, y: math_ops.less(x, 100.0)

      def b(x, y):
        y1 = array_ops.stop_gradient(math_ops.square(y))
        x1 = math_ops.add(math_ops.square(x), y1)
        return x1, y1

      rx, _ = control_flow_ops.while_loop(c, b, [x, y])

      r = gradients_impl.gradients(rx, y)[0]
      self.assertAllClose(0.0, r.eval())
      r = gradients_impl.gradients(rx, x)[0]
      self.assertAllClose(156.0, r.eval())

  def testWhileGrad_StopGradInsideNoShape(self):
    with self.test_session() as sess:
      x = array_ops.placeholder(dtypes.float32)
      y = array_ops.placeholder(dtypes.float32)

      c = lambda x, y: math_ops.less(math_ops.reduce_sum(x), 100.0)

      def b(x, y):
        y1 = array_ops.stop_gradient(math_ops.square(y, name="stopped"))
        x1 = math_ops.add(math_ops.square(x), y1)
        return x1, y1

      rx, _ = control_flow_ops.while_loop(c, b, [x, y])

      r = gradients_impl.gradients(rx, y)[0]
      feed_dict = {x: [3.0, 4.0], y: [2.0, 3.0]}
      self.assertAllClose([0.0, 0.0], sess.run(r, feed_dict=feed_dict))
      r = gradients_impl.gradients(rx, x)[0]
      self.assertAllClose([156.0, 400.0], sess.run(r, feed_dict=feed_dict))
      name = "gradients/while/stopped_grad"
      all_ops = x.graph.get_operations()
      self.assertFalse(any([name in op.name for op in all_ops]))

  def testWhileGradGradFail(self):
    theta = variables.Variable(initial_value=1.)

    def fn(prev, x):
      return prev + x * theta

    result = functional_ops.scan(fn, np.array([1., 2., 3.], dtype=np.float32))
    grad_theta = gradients_impl.gradients(result, theta)
    with self.assertRaisesRegexp(TypeError, "Second-order gradient"):
      gradients_impl.gradients(grad_theta, theta)
    grad_theta_stopped = array_ops.stop_gradient(grad_theta)
    gradients_impl.gradients(grad_theta_stopped, theta)

  def testStopGradOnWhileGrad(self):
    with self.test_session():
      x = constant_op.constant(2.0, name="x")
      y = constant_op.constant(2.0, name="y")

      c = lambda x: math_ops.less(x, 100.0)
      b = lambda x: math_ops.multiply(x, y)
      rx = control_flow_ops.while_loop(c, b, [x])

      rg = gradients_impl.gradients(rx, y)[0]
      rg = array_ops.stop_gradient(rg)
      r = math_ops.add(math_ops.square(y), rx)
      r = math_ops.add(r, rg)
      r = gradients_impl.gradients(r, y)[0]
      self.assertEqual(388.0, r.eval())

  def testStopGradMultiFlows(self):
    with self.test_session():
      def body(i, y, r):
        x = variable_scope.get_variable(
            "x", shape=(), dtype=dtypes.float32,
            initializer=init_ops.ones_initializer())
        y *= x
        return [i + 1, y, r + math_ops.reduce_sum(y)]

      i0 = constant_op.constant(0)
      y0 = array_ops.ones(5)
      r0 = constant_op.constant(0.0)
      cond = lambda i, y, r: i < 1
      _, _, r = control_flow_ops.while_loop(
          cond, body, [i0, y0, r0], back_prop=True)

      vars_ = variables.global_variables()
      grads = linalg_ops.norm(gradients_impl.gradients(r, vars_)[0])
      z = math_ops.add(r, array_ops.stop_gradient(math_ops.reduce_sum(grads)))
      result = gradients_impl.gradients(z, vars_)[0]
      variables.global_variables_initializer().run()
      self.assertEqual(5.0, result.eval())

  def testOneValueCond(self):
    with self.test_session():
      c = array_ops.placeholder(dtypes.int32, shape=[])
      one = ops.convert_to_tensor(1, name="one")
      two = ops.convert_to_tensor(2, name="two")
      p = math_ops.greater_equal(c, 1)
      i = control_flow_ops.cond(p, lambda: one, lambda: two)
      self.assertTrue(isinstance(i, ops.Tensor))

      # True case: c = 2 is >= 1
      self.assertEqual([1], i.eval(feed_dict={c: 2}))

      # False case: c = 0 is not >= 1
      self.assertEqual([2], i.eval(feed_dict={c: 0}))

  def testExampleCond(self):
    with self.test_session():
      x = ops.convert_to_tensor([-2.0, 2.0], name="x")
      d = array_ops.placeholder(dtypes.int32, shape=[])

      def l2():
        return math_ops.sqrt(math_ops.reduce_sum(math_ops.square(x)))

      def l1():
        return math_ops.reduce_sum(math_ops.abs(x))

      i = control_flow_ops.cond(math_ops.equal(d, 2), l2, l1)
      self.assertAllClose(4.0, i.eval(feed_dict={d: 1}))
      self.assertAllClose(2.0 * math.sqrt(2), i.eval(feed_dict={d: 2}))

  def testCase(self):
    with self.test_session():
      x = constant_op.constant(1)
      y = constant_op.constant(2)
      z = constant_op.constant(3)
      f1 = lambda: constant_op.constant(17)
      f2 = lambda: constant_op.constant(23)
      f3 = lambda: constant_op.constant(-1)

      r1 = control_flow_ops.case(
          {
              x < y: f1,
              x > z: f2
          }, default=f3, exclusive=True)
      self.assertAllEqual(r1.eval(), 17)

      r2 = control_flow_ops.case([(y > z, f1), (y > x, f2)], default=f3)
      self.assertAllEqual(r2.eval(), 23)

      # Duplicate events can happen, first one is selected
      r3 = control_flow_ops.case([(x < y, f1), (x < y, f2)], default=f3)
      self.assertAllEqual(r3.eval(), 17)

      # Duplicate events cause an error if exclusive = True
      r4 = control_flow_ops.case(
          [(x < y, f1), (x < y, f2)], default=f3, exclusive=True)
      with self.assertRaisesOpError(
          "More than one condition evaluated as True but exclusive=True."):
        r4.eval()

      # Check that the default is called if none of the others are
      r5 = control_flow_ops.case({x > y: f1}, default=f3)
      self.assertAllEqual(r5.eval(), -1)

      ran_once = [False, False, False]

      def break_run_twice(ix):

        def _break():
          ran_once[ix] = True
          return constant_op.constant(ix)

        return _break

      # Should not fail - each conditional gets called exactly once
      # except default.  Default gets called twice: once to create an
      # empty output and once for the actual cond switch.
      r6 = control_flow_ops.case(
          [(x < y, break_run_twice(0)), (x > y, break_run_twice(1))],
          default=lambda: constant_op.constant(2))

      self.assertAllEqual(r6.eval(), 0)

  def testCaseSideEffects(self):
    with self.test_session() as sess:
      v0 = variables.Variable(-1)
      v1 = variables.Variable(-1)
      v2 = variables.Variable(-1)

      a = lambda: control_flow_ops.with_dependencies([state_ops.assign(v0, 0)], 0)
      b = lambda: control_flow_ops.with_dependencies([state_ops.assign(v1, 1)], 1)
      c = lambda: control_flow_ops.with_dependencies([state_ops.assign(v2, 2)], 2)

      x = constant_op.constant(1)
      y = constant_op.constant(2)

      r0 = control_flow_ops.case(
          ((x < y, a), (x > y, b)), default=c, exclusive=True)
      r1 = control_flow_ops.case(
          ((x > y, a), (x < y, b)), default=c, exclusive=True)
      r2 = control_flow_ops.case(
          ((x > y, a), (x > y, b)), default=c, exclusive=True)

      variables.global_variables_initializer().run()
      self.assertAllEqual(sess.run([v0, v1, v2]), [-1] * 3)
      self.assertEqual(2, r2.eval())
      self.assertAllEqual(sess.run([v0, v1, v2]), [-1, -1, 2])

      variables.global_variables_initializer().run()
      self.assertAllEqual(sess.run([v0, v1, v2]), [-1] * 3)
      self.assertEqual(1, r1.eval())
      self.assertAllEqual(sess.run([v0, v1, v2]), [-1, 1, -1])

      variables.global_variables_initializer().run()
      self.assertAllEqual(sess.run([v0, v1, v2]), [-1] * 3)
      self.assertEqual(0, r0.eval())
      self.assertAllEqual(sess.run([v0, v1, v2]), [0, -1, -1])

  def testOneOpCond(self):
    with self.test_session():
      v = variables.Variable(0)
      c = ops.convert_to_tensor(0)
      one = ops.convert_to_tensor(1)
      two = ops.convert_to_tensor(2)
      p = math_ops.greater_equal(c, 1)

      def a():
        return state_ops.assign(v, one)

      def b():
        return state_ops.assign(v, two)

      i = control_flow_ops.cond(p, a, b)
      self.assertTrue(isinstance(i, ops.Tensor))
      variables.global_variables_initializer().run()

      self.assertEqual(0, v.eval())

      # True case: c = 2 is >= 1, v is set to 1.
      self.assertEqual(1, i.eval(feed_dict={c.name: 2}))
      self.assertEqual(1, v.eval())

      # False case: c = 0 is not >= 1, v is set to 2.
      self.assertEqual(2, i.eval(feed_dict={c.name: 0}))
      self.assertEqual(2, v.eval())

  def testWithOpsDependencies(self):
    with self.test_session() as sess:
      v = variables.Variable(0.0)
      c = constant_op.constant(10)

      # Fetching v directly will result in an uninitialized error
      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        sess.run([c, v])

      # Use a control dependency to ensure init_variable is run
      # while asking for c
      real_v = control_flow_ops.with_dependencies(
          name="real_tensor",
          output_tensor=v._ref(),  # pylint: disable=protected-access
          dependencies=[v.initializer])
      c_val, real_v_val = sess.run([c, real_v])

    # Ensure the result of 'real_c' is the same as 'c'
    self.assertAllEqual(10, c_val)

    # Ensure that 'v' is initialized
    self.assertAllClose(0.0, real_v_val)

  def testWithTensorDependencies(self):
    with self.test_session():
      v = variables.Variable(0.0)
      c1 = constant_op.constant(10)
      c2 = constant_op.constant(20)

      # c1_with_init_v depends on the init op for v
      c1_with_init_v = control_flow_ops.with_dependencies(
          name="c1_with_init_v", output_tensor=c1, dependencies=[v.initializer])
      # c2_with_c1 depends on the value of c1_with_init_v
      c2_with_c1_dep = control_flow_ops.with_dependencies(
          name="c2_with_c1_dep",
          output_tensor=c2,
          dependencies=[c1_with_init_v])

      # Fetching v directly will result in an uninitialized error
      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        v.eval()

      # Get the value of 'c2_with_c1_dep', which should cause 'v'
      # to be initialized.
      self.assertAllEqual(20, c2_with_c1_dep.eval())

      # Ensure that 'v' is initialized
      self.assertAllClose(0.0, v.eval())

  def testWithIndexedSlicesDependencies(self):
    with self.test_session():
      v = variables.Variable(
          np.array([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]]).astype(np.float32))
      v_at_1 = ops.IndexedSlices(v, constant_op.constant([1]))
      gather_v_at_1 = array_ops.gather(v_at_1.values, v_at_1.indices)
      v_at_1_after_init = control_flow_ops.with_dependencies([v.initializer],
                                                             v_at_1)
      gather_v_at_1_after_init = array_ops.gather(v_at_1_after_init.values,
                                                  v_at_1_after_init.indices)

      # Fetching gather_v_at_1 will result in an uninitialized error
      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        gather_v_at_1.eval()

      # Getting gather_v_at_1_after_init will work, and initialize v.
      self.assertAllEqual([[10.0, 11.0]], gather_v_at_1_after_init.eval())

      # Double check that 'v' is initialized
      self.assertAllClose([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]], v.eval())

  def testDependenciesDevice(self):
    with ops.Graph().as_default():
      # device set on tensor => same device on dep.
      with ops.device("/job:ps"):
        vd = variables.Variable([0.0])
      with_vd_dep = control_flow_ops.with_dependencies([vd.initializer], vd)
      self.assertTrue("/job:ps" in with_vd_dep.device)

      # No device set on tensor => no device on dep.
      vnod = variables.Variable([0.0])
      with_vnod_dep = control_flow_ops.with_dependencies([vnod.initializer],
                                                         vnod)
      self.assertDeviceEqual(None, with_vnod_dep.device)

      # device set on tensor, default device on graph => default device on dep.
      vdef = variables.Variable([0.0], name="vdef")
      with ops.device("/job:worker/gpu:1"):
        with_vdef_dep = control_flow_ops.with_dependencies([vdef.initializer],
                                                           vdef)
        # The device is empty, but the colocation constraint is set.
        self.assertDeviceEqual("", with_vdef_dep.device)
        self.assertEqual([b"loc:@vdef"], with_vdef_dep.op.colocation_groups())

  def testGroup(self):
    with self.test_session() as sess:
      v1 = variables.Variable([0.0])
      v2 = variables.Variable([1.0])

      # Group init1 and init2 and run.
      init = control_flow_ops.group(v1.initializer, v2.initializer)
      # Fetching v1 directly will result in an uninitialized error
      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        v1.eval()

      # Runs "init" before fetching v1 and v2.
      init.run()
      v1_val, v2_val = sess.run([v1, v2])

    # Ensure that v1 and v2 are initialized
    self.assertAllClose([0.0], v1_val)
    self.assertAllClose([1.0], v2_val)

  def testGroupEmpty(self):
    op = control_flow_ops.group()
    self.assertEqual(op.type, "NoOp")
    self.assertEqual(op.control_inputs, [])

  def testMergeShapes(self):
    # All inputs unknown.
    p1 = array_ops.placeholder(dtypes.float32)
    p2 = array_ops.placeholder(dtypes.float32)
    p3 = array_ops.placeholder(dtypes.float32)
    m, index = control_flow_ops.merge([p1, p2, p3])
    self.assertIs(None, m.get_shape().ndims)
    self.assertEqual([], index.get_shape())

    # All inputs known with different ranks.
    p1 = array_ops.placeholder(dtypes.float32, shape=[1, 2])
    p2 = array_ops.placeholder(dtypes.float32, shape=[1, 2, 3])
    m, index = control_flow_ops.merge([p1, p2])
    self.assertIs(None, m.get_shape().ndims)
    self.assertEqual([], index.get_shape())

    # All inputs known with some dimensions different.
    p1 = array_ops.placeholder(dtypes.float32, shape=[1, 2])
    p2 = array_ops.placeholder(dtypes.float32, shape=[2, 1])
    m, index = control_flow_ops.merge([p1, p2])
    self.assertEqual([None, None], m.get_shape().as_list())
    self.assertEqual([], index.get_shape())

    p1 = array_ops.placeholder(dtypes.float32, shape=[1, 2])
    p2 = array_ops.placeholder(dtypes.float32, shape=[None, 2])
    m, index = control_flow_ops.merge([p1, p2])
    self.assertEqual([None, 2], m.get_shape().as_list())
    self.assertEqual([], index.get_shape())

    p1 = array_ops.placeholder(dtypes.float32, shape=[1, 2])
    p2 = array_ops.placeholder(dtypes.float32, shape=[2, 2])
    m, index = control_flow_ops.merge([p1, p2])
    self.assertEqual([None, 2], m.get_shape().as_list())
    self.assertEqual([], index.get_shape())

    # All inputs known with same dimensions.
    p1 = array_ops.placeholder(dtypes.float32, shape=[1, 2])
    p2 = array_ops.placeholder(dtypes.float32, shape=[1, 2])
    m, index = control_flow_ops.merge([p1, p2])
    self.assertEqual([1, 2], m.get_shape().as_list())
    self.assertEqual([], index.get_shape())

    p1 = array_ops.placeholder(dtypes.float32, shape=[None, 2])
    p2 = array_ops.placeholder(dtypes.float32, shape=[None, 2])
    m, index = control_flow_ops.merge([p1, p2])
    self.assertEqual([None, 2], m.get_shape().as_list())
    self.assertEqual([], index.get_shape())

    p1 = array_ops.placeholder(dtypes.float32, shape=[None, None])
    p2 = array_ops.placeholder(dtypes.float32, shape=[None, None])
    m, index = control_flow_ops.merge([p1, p2])
    self.assertEqual([None, None], m.get_shape().as_list())
    self.assertEqual([], index.get_shape())

  def testRefSelect(self):
    index = array_ops.placeholder(dtypes.int32)

    # All inputs unknown.
    p1 = array_ops.placeholder(dtypes.float32)
    p2 = array_ops.placeholder(dtypes.float32)
    p3 = array_ops.placeholder(dtypes.float32)
    v1 = variables.Variable(p1, validate_shape=False)
    v2 = variables.Variable(p2, validate_shape=False)
    v3 = variables.Variable(p3, validate_shape=False)
    self.assertIs(None, v1.get_shape().ndims)
    s = control_flow_ops.ref_select(index, [v1, v2, v3])
    self.assertIs(None, s.get_shape().ndims)

    # All inputs known but different.
    v1 = variables.Variable([[1, 2]])
    v2 = variables.Variable([[2], [1]])
    s = control_flow_ops.ref_select(index, [v1, v2])
    self.assertIs(None, s.get_shape().ndims)

    # All inputs known and same.
    v1 = variables.Variable([[1, 2]])
    v2 = variables.Variable([[1, 2]])
    s = control_flow_ops.ref_select(index, [v1, v2])
    self.assertEqual([1, 2], s.get_shape())

    # Possibly the same but not guaranteed.
    v1 = variables.Variable([[1., 2.]])
    p2 = array_ops.placeholder(dtypes.float32, shape=[None, 2])
    v2 = variables.Variable(p2, validate_shape=False)
    s = control_flow_ops.ref_select(index, [v1, v2])
    self.assertEqual(None, s.get_shape())

  def testRunLoopTensor(self):
    with self.test_session() as sess:
      tensor_list = []

      def condition(t):
        return t < constant_op.constant(5)

      def body(_):
        tensor_list.append(constant_op.constant(5))
        return constant_op.constant(10)

      result = control_flow_ops.while_loop(condition, body,
                                           [constant_op.constant(4)])
      self.assertEqual(10, sess.run(result))

      # Ensure that we cannot run a tensor that escapes the loop body
      # accidentally.
      with self.assertRaises(ValueError):
        sess.run(tensor_list[0])

  def testWhilePyFuncBasic(self):

    def func(x):
      return np.square(x)

    with self.test_session():
      r = control_flow_ops.while_loop(
          lambda i, v: i < 4,
          lambda i, v: [i + 1, script_ops.py_func(func, [v], [dtypes.float32])[0]],
          [constant_op.constant(0), constant_op.constant(2.0, dtypes.float32)],
          [tensor_shape.unknown_shape(), tensor_shape.unknown_shape()])
      self.assertEqual(r[1].eval(), 65536.0)

  def testWhileFuncBasic(self):

    @function.Defun(dtypes.float32)
    def func(x):
      return math_ops.square(math_ops.square(x))

    with self.test_session():
      x = constant_op.constant(2.0, dtypes.float32)
      r = control_flow_ops.while_loop(
          lambda i, v: i < 2, lambda i, v: [i + 1, func(v)],
          [constant_op.constant(0), x],
          [tensor_shape.unknown_shape(), tensor_shape.unknown_shape()])
      self.assertEqual(r[1].eval(), 65536.0)

      r = gradients_impl.gradients(r, x)[0]
      self.assertEqual(r.eval(), 524288.0)
      self.assertEqual(
          len([op for op in x.graph.get_operations() if op.type == "Stack"]), 1)


class TupleTest(test.TestCase):

  def testTensors(self):
    for v1_first in [True, False]:
      with self.test_session():
        v1 = variables.Variable([1.0])
        add1 = math_ops.add(
            control_flow_ops.with_dependencies([v1.initializer], v1._ref()),  # pylint: disable=protected-access
            2.0)
        v2 = variables.Variable([10.0])
        add2 = math_ops.add(
            control_flow_ops.with_dependencies([v2.initializer], v2._ref()),  # pylint: disable=protected-access
            20.0)
        t1, _, t2 = control_flow_ops.tuple([add1, None, add2])

        # v1 is not initialized.
        with self.assertRaisesOpError("Attempting to use uninitialized value"):
          v1.eval()

        # v2 is not initialized.
        with self.assertRaisesOpError("Attempting to use uninitialized value"):
          v2.eval()

        if v1_first:
          # Getting t1 initializes v2.
          self.assertAllClose([3.0], t1.eval())
          self.assertAllClose([10.0], v2.eval())
        else:
          # Getting t2 initializes v1.
          self.assertAllClose([30.0], t2.eval())
          self.assertAllClose([1.0], v1.eval())

  def testIndexedSlices(self):
    for v1_first in [True, False]:
      with self.test_session():
        v1 = variables.Variable(
            np.array([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]]).astype(
                np.float32))
        v1_at_1 = ops.IndexedSlices(
            control_flow_ops.with_dependencies([v1.initializer], v1._ref()),  # pylint: disable=protected-access
            constant_op.constant([1]))

        v2 = variables.Variable(
            np.array([[0.1, 1.1], [10.1, 11.1], [20.1, 21.1]]).astype(
                np.float32))
        v2_at_1 = ops.IndexedSlices(
            control_flow_ops.with_dependencies([v2.initializer], v2._ref()),  # pylint: disable=protected-access
            constant_op.constant([1]))

        st1, st2 = control_flow_ops.tuple([v1_at_1, v2_at_1])
        g1 = array_ops.gather(st1.values, st1.indices)
        g2 = array_ops.gather(st2.values, st2.indices)

        # v1 is not initialized.
        with self.assertRaisesOpError("Attempting to use uninitialized value"):
          v1.eval()

        # v2 is not initialized.
        with self.assertRaisesOpError("Attempting to use uninitialized value"):
          v2.eval()

        if v1_first:
          # Getting g1 initializes v2.
          self.assertAllClose([[10.0, 11.0]], g1.eval())
          self.assertAllClose([[0.1, 1.1], [10.1, 11.1], [20.1, 21.1]],
                              v2.eval())
        else:
          # Getting g2 initializes v1.
          self.assertAllClose([[10.1, 11.1]], g2.eval())
          self.assertAllClose([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]],
                              v1.eval())

  def testAcceptTensorsAsControlInputs(self):
    with self.test_session():
      var = variables.Variable(0)
      assign = state_ops.assign(var, 1)
      t, = control_flow_ops.tuple(
          [constant_op.constant(0)], control_inputs=[assign])

      # Should trigger the assign.
      t.eval()

      self.assertEquals(1, var.eval())


class AssertTest(test.TestCase):

  def testGuardedAssertDoesNotCopyWhenTrue(self):
    with self.test_session(use_gpu=True) as sess:
      with ops.device(test.gpu_device_name()):
        value = constant_op.constant(1.0)
      with ops.device("/cpu:0"):
        true = constant_op.constant(True)
        guarded_assert = control_flow_ops.Assert(true, [value], name="guarded")
        unguarded_assert = gen_logging_ops._assert(
            true, [value], name="unguarded")
      opts = config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
      guarded_metadata = config_pb2.RunMetadata()
      sess.run(guarded_assert, options=opts, run_metadata=guarded_metadata)
      unguarded_metadata = config_pb2.RunMetadata()
      sess.run(unguarded_assert, options=opts, run_metadata=unguarded_metadata)
      guarded_nodestat_names = [
          n.node_name
          for d in guarded_metadata.step_stats.dev_stats for n in d.node_stats
      ]
      unguarded_nodestat_names = [
          n.node_name
          for d in unguarded_metadata.step_stats.dev_stats for n in d.node_stats
      ]
      guarded_memcpy_nodestat_names = [
          n for n in guarded_nodestat_names if "MEMCPYDtoH" in n
      ]
      unguarded_memcpy_nodestat_names = [
          n for n in unguarded_nodestat_names if "MEMCPYDtoH" in n
      ]
      if "GPU" in [d.device_type for d in device_lib.list_local_devices()]:
        # A copy was performed for the unguarded assert
        self.assertLess(0, len(unguarded_memcpy_nodestat_names))
      # No copy was performed for the guarded assert
      self.assertEqual([], guarded_memcpy_nodestat_names)

if __name__ == "__main__":
  test.main()
