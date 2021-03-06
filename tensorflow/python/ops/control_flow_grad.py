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

"""Gradients for operators defined in control_flow_ops.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import,undefined-variable
from tensorflow.python.ops.control_flow_ops import *
# pylint: enable=wildcard-import


def _SwitchGrad(op, *grad):
  """Gradients for a Switch op is calculated using a Merge op.

  If the switch is a loop switch, it will be visited twice. We create
  the merge on the first visit, and update the other input of the merge
  on the second visit. A next_iteration is also added on second visit.
  """
  graph = ops.get_default_graph()
  # pylint: disable=protected-access
  op_ctxt = op._get_control_flow_context()
  grad_ctxt = graph._get_control_flow_context()
  # pylint: enable=protected-access
  if isinstance(op_ctxt, WhileContext):
    merge_grad = grad_ctxt.grad_state.switch_map.get(op)
    if merge_grad is not None:
      # This is the second time this Switch is visited. It comes from
      # the non-exit branch of the Switch, so update the second input
      # to the Merge.
      # TODO(yuanbyu): Perform shape inference with this new input.
      if grad[1] is not None:
        # pylint: disable=protected-access
        control_flow_ops._AddNextAndBackEdge(merge_grad, grad[1],
                                             enforce_shape_invariant=False)
        # pylint: enable=protected-access
      return None, None
    elif grad[0] is not None:
      # This is the first time this Switch is visited. It comes from
      # the Exit branch, which is grad[0]. grad[1] is empty at this point.
      # Use grad[0] for both inputs to merge for now, but update the second
      # input of merge when we see this Switch the second time.
      merge_grad = merge([grad[0], grad[0]], name="b_switch")[0]
      grad_ctxt.grad_state.switch_map[op] = merge_grad
      return merge_grad, None
    else:
      # This is the first time this Switch is visited. It comes from the
      # Identity branch. Such a Switch has `None` gradient for the Exit branch,
      # meaning the output is not differentiable.
      return None, None
  elif isinstance(op_ctxt, CondContext):
    zero_grad = grad[1 - op_ctxt.branch]
    # At this point, we have created zero_grad guarded by the right switch.
    # Unfortunately, we may still get None here for not trainable data types.
    if zero_grad is None:
      # For resource variables we get None always on the other branch, so bypass
      # this.
      if op.inputs[0].dtype == dtypes.resource:
        return merge(
            [grad[op_ctxt.branch]] * 2, name="cond_resource_grad")[0], None
      return None, None
    return merge(grad, name="cond_grad")[0], None
=======
"""Gradients for operators defined in control_flow_ops.py."""
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
# pylint: disable=wildcard-import,undefined-variable
from tensorflow.python.ops.control_flow_ops import *
from tensorflow.python.ops.gen_control_flow_ops import *


@ops.RegisterGradient("Switch")
def _SwitchGrad(op, *grad):
  op = GetRealOp(op)
  ctxt = op._get_control_flow_context()  # pylint: disable=protected-access
  if isinstance(ctxt, WhileContext):
    merge_op = ctxt.switch_map.get(op)
    if merge_op:
      merge_op._update_input(1, grad[1])
      return None, None
    else:
      merge_op = merge(grad, name="b_switch")[0]
      ctxt.switch_map[op] = merge_op.op
      return merge_op, None
  elif isinstance(ctxt, CondContext):
    good_grad = grad[ctxt.branch]
    zero_grad = grad[1 - ctxt.branch]
    zero_grad = switch(zero_grad, ctxt.pred, name="grad_0")[1 - ctxt.branch]
    return merge([good_grad, zero_grad], name="switch_grad")[0], None
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  else:
    false_grad = switch(grad[0], op.inputs[1])[0]
    true_grad = switch(grad[1], op.inputs[1])[1]
    return merge([false_grad, true_grad])[0], None


<<<<<<< HEAD
ops.RegisterGradient("Switch")(_SwitchGrad)
ops.RegisterGradient("RefSwitch")(_SwitchGrad)
=======
@ops.RegisterGradient("RefSwitch")
def _RefSwitchGrad(op, *grad):
  return _SwitchGrad(op, *grad)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.


@ops.RegisterGradient("Merge")
def _MergeGrad(op, grad, _):
<<<<<<< HEAD
  """Gradients for a Merge op are calculated using a Switch op."""
  input_op = op.inputs[0].op
  graph = ops.get_default_graph()
  # pylint: disable=protected-access
  op_ctxt = control_flow_util.GetOutputContext(input_op)
  grad_ctxt = graph._get_control_flow_context()
  # pylint: enable=protected-access
  if isinstance(op_ctxt, WhileContext):
    # pylint: disable=protected-access
    return control_flow_ops._SwitchRefOrTensor(grad, grad_ctxt.pivot)
    # pylint: enable=protected-access
  elif isinstance(op_ctxt, CondContext):
    pred = op_ctxt.pred
    if grad_ctxt and grad_ctxt.grad_state:
      # This Merge node is part of a cond within a loop.
      # The backprop needs to have the value of this predicate for every
      # iteration. So we must have its values accumulated in the forward, and
      # use the accumulated values as the predicate for this backprop switch.
      grad_state = grad_ctxt.grad_state
      real_pred = grad_state.history_map.get(pred.name)
      if real_pred is None:
        # Remember the value of pred for every iteration.
        grad_ctxt = grad_state.grad_context
        grad_ctxt.Exit()
        history_pred = grad_state.AddForwardAccumulator(pred)
        grad_ctxt.Enter()

        # Add the stack pop op. If pred.op is in a (outer) CondContext,
        # the stack pop will be guarded with a switch.
        real_pred = grad_state.AddBackpropAccumulatedValue(history_pred, pred)
        grad_state.history_map[pred.name] = real_pred
      pred = real_pred
    # pylint: disable=protected-access
    return control_flow_ops._SwitchRefOrTensor(grad, pred, name="cond_grad")
    # pylint: enable=protected-access
  else:
    num_inputs = len(op.inputs)
    cond = [math_ops.equal(op.outputs[1], i) for i in xrange(num_inputs)]
    # pylint: disable=protected-access
    return [control_flow_ops._SwitchRefOrTensor(grad, cond[i])[1]
            for i in xrange(num_inputs)]
    # pylint: enable=protected-access


@ops.RegisterGradient("RefMerge")
def _RefMergeGrad(op, grad, _):
  return _MergeGrad(op, grad, _)
=======
  op = GetRealOp(op)
  input_op = op.inputs[0].op
  # pylint: disable=protected-access
  ctxt = input_op._get_control_flow_context()
  # pylint: enable=protected-access
  if isinstance(ctxt, WhileContext):
    grad_ctxt = ctxt.grad_context
    return switch(grad, grad_ctxt.pivot)
  elif isinstance(ctxt, CondContext):
    return switch(grad, ctxt.pred, name="merge_grad")
  else:
    num_inputs = len(op.inputs)
    cond = [math_ops.equal(op.outputs[1], i) for i in xrange(num_inputs)]
    return [Switch(grad, cond[i])[1] for i in xrange(num_inputs)]
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.


@ops.RegisterGradient("Exit")
def _ExitGrad(op, grad):
<<<<<<< HEAD
  """Gradients for an exit op are calculated using an Enter op."""
  graph = ops.get_default_graph()
  # pylint: disable=protected-access
  op_ctxt = op._get_control_flow_context()
  grad_ctxt = graph._get_control_flow_context()
  # pylint: enable=protected-access
  if not grad_ctxt.back_prop:
    # The flag `back_prop` is set by users to suppress gradient
    # computation for this loop. If the attribute `back_prop` is false,
    # no gradient computation.
    return None

  if op_ctxt.grad_state:
    raise TypeError("Second-order gradient for while loops not supported.")

  if isinstance(grad, ops.Tensor):
    grad_ctxt.AddName(grad.name)
  else:
    if not isinstance(grad, (ops.IndexedSlices, sparse_tensor.SparseTensor)):
      raise TypeError("Type %s not supported" % type(grad))
    grad_ctxt.AddName(grad.values.name)
    grad_ctxt.AddName(grad.indices.name)
    dense_shape = grad.dense_shape
    if dense_shape is not None:
      grad_ctxt.AddName(dense_shape.name)
  grad_ctxt.Enter()
  # pylint: disable=protected-access
  result = control_flow_ops._Enter(
      grad, grad_ctxt.name, is_constant=False,
      parallel_iterations=grad_ctxt.parallel_iterations,
      name="b_exit")
  # pylint: enable=protected-access
  grad_ctxt.loop_enters.append(result)
  grad_ctxt.Exit()
  return result


ops.RegisterGradient("RefExit")(_ExitGrad)
=======
  # pylint: disable=protected-access
  forward_ctxt = op._get_control_flow_context()
  # pylint: enable=protected-access
  if not forward_ctxt.back_prop:
    return None
  grad_ctxt = forward_ctxt.grad_context
  grad_ctxt.AddName(grad.name)
  return enter(grad, grad_ctxt.name, is_constant=False,
               parallel_iterations=forward_ctxt.parallel_iterations,
               name="b_exit")
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.


@ops.RegisterGradient("NextIteration")
def _NextIterationGrad(_, grad):
<<<<<<< HEAD
  """A forward next_iteration is translated into a backprop identity.

  Note that the backprop next_iteration is added in switch grad.
  """
  return grad


@ops.RegisterGradient("RefNextIteration")
def _RefNextIterationGrad(_, grad):
  return _NextIterationGrad(_, grad)
=======
  return next_iteration(grad)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.


@ops.RegisterGradient("Enter")
def _EnterGrad(op, grad):
<<<<<<< HEAD
  """Gradients for an Enter are calculated using an Exit op.

  For loop variables, grad is the gradient so just add an exit.
  For loop invariants, we need to add an accumulator loop.
  """
  graph = ops.get_default_graph()
  # pylint: disable=protected-access
  grad_ctxt = graph._get_control_flow_context()
  # pylint: enable=protected-access
  if not grad_ctxt.back_prop:
    # Skip gradient computation, if the attribute `back_prop` is false.
    return grad
  if grad_ctxt.grad_state is None:
    # Pass the gradient through if we are not in a gradient while context.
    return grad
  if op.get_attr("is_constant"):
    # Add a gradient accumulator for each loop invariant.
    if isinstance(grad, ops.Tensor):
      result = grad_ctxt.AddBackpropAccumulator(op, grad)
    elif isinstance(grad, ops.IndexedSlices):
      result = grad_ctxt.AddBackpropIndexedSlicesAccumulator(op, grad)
    else:
      # TODO(yuanbyu, lukasr): Add support for SparseTensor.
      raise TypeError("Type %s not supported" % type(grad))
  else:
    result = exit(grad)
    grad_ctxt.loop_exits.append(result)
    grad_ctxt.ExitResult([result])
  return result
=======
  op = GetRealOp(op)
  # pylint: disable=protected-access
  forward_ctxt = op._get_control_flow_context()
  # pylint: enable=protected-access
  grad_ctxt = forward_ctxt.grad_context
  if grad_ctxt:
    if op.get_attr("is_constant"):
      # Add a gradient accumulator for every loop invariant.
      result = grad_ctxt.AddBackPropAccumulateLoop(grad)
    else:
      result = exit(grad)
    return result
  else:
    return grad
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.


@ops.RegisterGradient("RefEnter")
def _RefEnterGrad(op, grad):
  return _EnterGrad(op, grad)


@ops.RegisterGradient("LoopCond")
def _LoopCondGrad(_):
<<<<<<< HEAD
  """Stop backprop for the predicate of a while loop."""
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  return None
