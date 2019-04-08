# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Contains LossScale classes."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import loss_scale as loss_scale_module
from tensorflow.python.training import optimizer
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=['train.LossScaleOptimizer'])
class LossScaleOptimizer(optimizer.Optimizer):
  """An optimizer that applies loss scaling.

  The loss scale can either be a fixed constant, chosen by the user, or be
  dynamically determined. Dynamically determining the loss scale is convenient
  as a loss scale does not have to be explicitly chosen. However it reduces
  performance.

  This optimizer wraps another optimizer and applies loss scaling to it. Loss
  scaling is applied whenever gradients are computed.

  Args:
    opt: The Optimizer instance to wrap.
    loss_scale: The loss scale or LossScale class to scale the loss and
      gradients. This can either be an int/float to use a fixed loss scale,
      the string "dynamic" to use dynamic loss scaling, or an instance of a
      LossScale class. The string "dynamic" is equivalent to passing
      `DynamicLossScale()`, and passing an int/float is equivalent
      to passing a FixedLossScale instance with the given loss scale.
  """
  def __init__(self, opt, loss_scale):
    if not isinstance(opt, optimizer.Optimizer):
      raise ValueError('"opt" must be an instance of Optimizer, but got: %s' %
                       type(opt))
    self._optimizer = opt

    use_locking = opt._use_locking # pylint: disable=protected-access
    name = opt.get_name()
    super(LossScaleOptimizer, self).__init__(use_locking, name)

    self._loss_scale = loss_scale_module.get(loss_scale)
    self._track_trackable(self._optimizer, 'base_optimizer')
    self._track_trackable(self._loss_scale, 'loss_scale')


  def _doing_dynamic_loss_scaling(self):
    """Check if `_loss_scale` dynamically manages the loss scale."""
    return isinstance(self._loss_scale,
                      loss_scale_module.DynamicLossScale)


  def compute_gradients(self, loss, var_list=None,
                        gate_gradients=optimizer.Optimizer.GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    """Compute gradients of `loss` for the variables in `var_list`.

    This adjusts the dynamic range of the gradient evalutaion by scaling up
    the `loss` value. The gradient values are then scaled back down by the
    recipricol of the loss scale. This is useful in reduced precision training
    where small gradient values would otherwise underflow the representable
    range.

    Args:
      loss: A Tensor containing the value to minimize or a callable taking
        no arguments which returns the value to minimize. When eager execution
        is enabled it must be a callable.
      var_list: Optional list or tuple of `tf.Variable` to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph
        under the key `GraphKeys.TRAINABLE_VARIABLES`.
      gate_gradients: How to gate the computation of gradients.  Can be
        `GATE_NONE`, `GATE_OP`, or `GATE_GRAPH`.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
      colocate_gradients_with_ops: If True, try colocating gradients with
        the corresponding op.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.

    Returns:
      A list of (gradient, variable) pairs. Variable is always present, but
      gradient can be `None`.
    """
    loss = self._scale_loss(loss)
    grads_and_vars = self._optimizer.compute_gradients(
        loss=loss, var_list=var_list, gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        grad_loss=grad_loss)

    grads = [g for g, _ in grads_and_vars]
    variables = [v for _, v in grads_and_vars]
    scaled_grads = self._scale_grads(grads)
    return list(zip(scaled_grads, variables))

  def _scale_loss(self, loss):
    # The loss is callable for `_compute_gradients`, but not `get_gradients`.
    loss_scale = self._loss_scale()
    if callable(loss):
      return lambda: loss() * loss_scale
    return loss * loss_scale

  def _scale_grads(self, grads):
    loss_scale = self._loss_scale()
    loss_scale_reciprical = 1 / loss_scale
    return [None if g is None else self._indexed_slices(
        g, loss_scale_reciprical) for g in grads]

  def _indexed_slices(self, grad, loss_scale_reciprical):
    if isinstance(grad, ops.IndexedSlices):
      grad_vals = grad.values * loss_scale_reciprical
      return ops.IndexedSlices(grad_vals, grad.indices, grad.dense_shape)
    return grad * loss_scale_reciprical

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Apply gradients to variables.

    This is the second part of `minimize()`. It returns an `Operation` that
    conditionally applies gradients if all gradient values are finite.
    Otherwise no update is performed (nor is `global_step` incremented).

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        `compute_gradients()`.
      global_step: Optional `Variable` to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the name
        passed to the `Optimizer` constructor.

    Returns:
      An `Operation` that conditionally applies the specified gradients. If
      `global_step` was not None, that operation also increments `global_step`.

    Raises:
      RuntimeError: If you should use `_distributed_apply()` instead.
    """
    if distribution_strategy_context.in_cross_replica_context():
      raise ValueError('apply_gradients() must be called in a replica context.')

    if not self._doing_dynamic_loss_scaling():
      return self._optimizer.apply_gradients(grads_and_vars, global_step, name)

    replica_context = distribution_strategy_context.get_replica_context()

    # TODO(nluehr) cleanup GraphKeys.TRAIN_OP
    return replica_context.merge_call(
        self._maybe_apply_gradients_cross_replica,
        args=(grads_and_vars, global_step, name))

  def _distributed_apply(self,
                         distribution,
                         grads_and_vars,
                         global_step=None,
                         name=None):
    """A version of `apply_gradients` for cross replica context.

    When users are in a cross replica strategy, they must call this rather than
    `apply_gradients()`.

    Args:
      distribution: a `DistributionStrategy` object.
      grads_and_vars: List of (gradient, variable) pairs as returned by
        `compute_gradients()` and then aggregated across replicas.
      global_step: Optional (mirrored) `Variable` to increment by one
        after the variables have been updated.
      name: Optional name for the returned operation. Default to the name
        passed to the `Optimizer` constructor.

    Returns:
      An `Operation` that applies the specified gradients across all
      replicas. If `global_step` was not None, that operation also
      increments `global_step`
    """
    self._maybe_apply_gradients_cross_replica(distribution, grads_and_vars,
                                              global_step, name)

  def _maybe_apply_gradients_cross_replica(self, distribution, grads_and_vars,
                                           global_step, name):
    """Conditionally apply gradients in cross replica context."""
    name = name if name is not None else self.get_name()
    grads = [g for g, _ in grads_and_vars]
    loss_scale_update_op, should_apply_grads = (
        self._loss_scale.update(grads))
    maybe_apply_op = smart_cond.smart_cond(
        should_apply_grads,
        lambda: self._apply_gradients_cross_replica(distribution,
                                                    grads_and_vars,
                                                    global_step,
                                                    name+'-wrapped'),
        control_flow_ops.no_op)
    return control_flow_ops.group(maybe_apply_op, loss_scale_update_op,
                                  name=name)

  def _apply_gradients_cross_replica(self, distribution, grads_and_vars,
                                     global_step, name):
    """Unconditionally apply gradients in cross replica context."""
    update_ops = distribution.extended.call_for_each_replica(
        self._optimizer.apply_gradients,
        args=(grads_and_vars, global_step, name))
    return distribution.group(update_ops)

  def _apply_sparse(self, grad, var):
    """This function should never be called"""
    raise RuntimeError("This function should never be called")

  def _apply_dense(self, grad, var):
    """This function should never be called"""
    raise RuntimeError("This function should never be called")

  def _resource_apply_sparse(self, grad, handle, indices):
    """This function should never be called"""
    raise RuntimeError("This function should never be called")

  def _resource_apply_dense(self, grad, handle):
    """This function should never be called"""
    raise RuntimeError("This function should never be called")
