# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains various routines and helper functions for training models.

TODO(nsilberman): Port documentation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python import summary
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import monitored_session
from tensorflow.python.training import optimizer as tf_optimizer

# TODO(nsilberman): move add_gradients_summaries, clip_gradient_norms and
# multiply_gradients into contrib/summaries and contrib/optimizers.py
__all__ = [
    'add_gradients_summaries',
    'clip_gradient_norms',
    'create_train_op',
    'multiply_gradients',
    'train',
]


def add_gradients_summaries(grads_and_vars):
  """Add summaries to gradients.

  Args:
    grads_and_vars: A list of gradient to variable pairs (tuples).

  Returns:
    The list of created summaries.
  """
  summaries = []
  for grad, var in grads_and_vars:
    if grad is not None:
      if isinstance(grad, ops.IndexedSlices):
        grad_values = grad.values
      else:
        grad_values = grad
      summaries.append(summary.histogram_summary(
          var.op.name + ':gradient', grad_values))
      summaries.append(summary.histogram_summary(
          var.op.name + ':gradient_norm', clip_ops.global_norm([grad_values])))
    else:
      logging.info('Var %s has no gradient', var.op.name)

  return summaries


def clip_gradient_norms(gradients_to_variables, max_norm):
  """Clips the gradients by the given value.

  Args:
    gradients_to_variables: A list of gradient to variable pairs (tuples).
    max_norm: the maximum norm value.

  Returns:
    A list of clipped gradient to variable pairs.
  """
  clipped_grads_and_vars = []
  for grad, var in gradients_to_variables:
    if grad is not None:
      if isinstance(grad, ops.IndexedSlices):
        tmp = clip_ops.clip_by_norm(grad.values, max_norm)
        grad = ops.IndexedSlices(tmp, grad.indices, grad.dense_shape)
      else:
        grad = clip_ops.clip_by_norm(grad, max_norm)
    clipped_grads_and_vars.append((grad, var))
  return clipped_grads_and_vars


def multiply_gradients(grads_and_vars, gradient_multipliers):
  """Multiply specified gradients.

  Args:
    grads_and_vars: A list of gradient to variable pairs (tuples).
    gradient_multipliers: A map from either `Variables` or `Variable` op names
      to the coefficient by which the associated gradient should be scaled.

  Returns:
    The updated list of gradient to variable pairs.

  Raises:
    ValueError: If `grads_and_vars` is not a list or if `gradient_multipliers`
    is empty or None or if `gradient_multipliers` is not a dictionary.
  """
  if not isinstance(grads_and_vars, list):
    raise ValueError('`grads_and_vars` must be a list.')
  if not gradient_multipliers:
    raise ValueError('`gradient_multipliers` is empty.')
  if not isinstance(gradient_multipliers, dict):
    raise ValueError('`gradient_multipliers` must be a dict.')

  multiplied_grads_and_vars = []
  for grad, var in grads_and_vars:
    if var in gradient_multipliers or var.op.name in gradient_multipliers:
      key = var if var in gradient_multipliers else var.op.name
      if grad is None:
        raise ValueError('Requested multiple of `None` gradient.')

      if isinstance(grad, ops.IndexedSlices):
        tmp = grad.values * constant_op.constant(
            gradient_multipliers[key], dtype=grad.dtype)
        grad = ops.IndexedSlices(tmp, grad.indices, grad.dense_shape)
      else:
        grad *= constant_op.constant(
            gradient_multipliers[key], dtype=grad.dtype)
    multiplied_grads_and_vars.append((grad, var))
  return multiplied_grads_and_vars


def create_train_op(total_loss,
                    optimizer,
                    global_step=None,
                    update_ops=None,
                    variables_to_train=None,
                    transform_grads_fn=None,
                    summarize_gradients=False,
                    gate_gradients=tf_optimizer.Optimizer.GATE_OP,
                    aggregation_method=None,
                    colocate_gradients_with_ops=False):
  """Creates an `Operation` that evaluates the gradients and returns the loss.

  Args:
    total_loss: An `Output` representing the total loss.
    optimizer: A tf.Optimizer to use for computing the gradients.
    global_step: An `Output` representing the global step variable. If left as
      `None`, then slim.variables.global_step() is used.
    update_ops: An optional list of updates to execute. If `update_ops` is
      `None`, then the update ops are set to the contents of the
      `tf.GraphKeys.UPDATE_OPS` collection. If `update_ops` is not `None`, but
      it doesn't contain all of the update ops in `tf.GraphKeys.UPDATE_OPS`,
      a warning will be displayed.
    variables_to_train: an optional list of variables to train. If None, it will
      default to all tf.trainable_variables().
    transform_grads_fn: A function which takes a single argument, a list of
      gradient to variable pairs (tuples), performs any requested gradient
      updates, such as gradient clipping or multipliers, and returns the updated
      list.
    summarize_gradients: Whether or not add summaries for each gradient.
    gate_gradients: How to gate the computation of gradients. See tf.Optimizer.
    aggregation_method: Specifies the method used to combine gradient terms.
      Valid values are defined in the class `AggregationMethod`.
    colocate_gradients_with_ops: Whether or not to try colocating the gradients
      with the ops that generated them.

  Returns:
    An `Output` that when evaluated, computes the gradients and returns the
      total loss value.
  """
  if global_step is None:
    global_step = variables.get_or_create_global_step()

  # Update ops use GraphKeys.UPDATE_OPS collection if update_ops is None.
  global_update_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS))
  if update_ops is None:
    update_ops = global_update_ops
  else:
    update_ops = set(update_ops)
  if not global_update_ops.issubset(update_ops):
    logging.warning('update_ops in create_train_op does not contain all the '
                    ' update_ops in GraphKeys.UPDATE_OPS')

  # Make sure update_ops are computed before total_loss.
  if update_ops:
    with ops.control_dependencies(update_ops):
      barrier = control_flow_ops.no_op(name='update_barrier')
    total_loss = control_flow_ops.with_dependencies([barrier], total_loss)

  if variables_to_train is None:
    # Default to tf.trainable_variables()
    variables_to_train = tf_variables.trainable_variables()
  else:
    # Make sure that variables_to_train are in tf.trainable_variables()
    for v in variables_to_train:
      assert v in tf_variables.trainable_variables()

  assert variables_to_train

  # Create the gradients. Note that apply_gradients adds the gradient
  # computation to the current graph.
  grads = optimizer.compute_gradients(
      total_loss,
      variables_to_train,
      gate_gradients=gate_gradients,
      aggregation_method=aggregation_method,
      colocate_gradients_with_ops=colocate_gradients_with_ops)

  if transform_grads_fn:
    grads = transform_grads_fn(grads)

  # Summarize gradients.
  if summarize_gradients:
    with ops.name_scope('summarize_grads'):
      add_gradients_summaries(grads)

  # Create gradient updates.
  grad_updates = optimizer.apply_gradients(grads, global_step=global_step)

  with ops.name_scope('train_op'):
    # Make sure total_loss is valid.
    total_loss = array_ops.check_numerics(total_loss,
                                          'LossTensor is inf or nan')

    # Ensure the train_tensor computes grad_updates.
    return control_flow_ops.with_dependencies([grad_updates], total_loss)


def train(
    train_op,
    logdir,
    master='',
    is_chief=True,
    scaffold=None,
    hooks=None,
    chief_only_hooks=None,
    save_checkpoint_secs=600,
    save_summaries_steps=100,
    config=None):
  """Runs the training loop.

  Args:
    train_op: An `Output` that, when executed, will apply the gradients and
      return the loss value.
    logdir: The directory where the graph and checkpoints are saved.
    master: The URL of the master.
    is_chief: Specifies whether or not the training is being run by the primary
      replica during replica training.
    scaffold: An tf.train.Scaffold instance.
    hooks: List of `tf.train.SessionRunHook` callbacks which are run inside the
      training loop.
    chief_only_hooks: List of `tf.train.SessionRunHook` instances which are run
      inside the training loop for the chief trainer only.
    save_checkpoint_secs: The frequency, in seconds, that a checkpoint is saved
      using a default checkpoint saver. If `save_checkpoint_secs` is set to
      `None`, then the default checkpoint saver isn't used.
    save_summaries_steps: The frequency, in number of global steps, that the
      summaries are written to disk using a default summary saver. If
      `save_summaries_steps` is set to `None`, then the default summary saver
      isn't used.
    config: An instance of `tf.ConfigProto`.

  Returns:
    the value of the loss function after training.

  Raises:
    ValueError: if `logdir` is `None` and either `save_checkpoint_secs` or
    `save_summaries_steps` are `None.
  """
  # TODO(nsilberman): move this logic into monitored_session.py
  scaffold = scaffold or monitored_session.Scaffold()

  hooks = hooks or []

  if is_chief:
    session_creator = monitored_session.ChiefSessionCreator(
        scaffold=scaffold,
        checkpoint_dir=logdir,
        master=master,
        config=config)

    if chief_only_hooks:
      hooks.extend(chief_only_hooks)

    hooks.append(basic_session_run_hooks.StepCounterHook(
        output_dir=logdir))

    if save_summaries_steps:
      if logdir is None:
        raise ValueError(
            'logdir cannot be None when save_summaries_steps is None')
      hooks.append(basic_session_run_hooks.SummarySaverHook(
          scaffold=scaffold,
          save_steps=save_summaries_steps,
          output_dir=logdir))

    if save_checkpoint_secs:
      if logdir is None:
        raise ValueError(
            'logdir cannot be None when save_checkpoint_secs is None')
      hooks.append(basic_session_run_hooks.CheckpointSaverHook(
          logdir, save_secs=save_checkpoint_secs, scaffold=scaffold))
  else:
    session_creator = monitored_session.WorkerSessionCreator(
        scaffold=scaffold, master=master, config=config)

  with monitored_session.MonitoredSession(
      session_creator=session_creator, hooks=hooks) as session:
    loss = None
    while not session.should_stop():
      loss = session.run(train_op)
  return loss
