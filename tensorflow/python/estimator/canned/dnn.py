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
# ==============================================================================
"""Deep Neural Network estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import model_fn
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.estimator.canned import optimizers
from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.layers import core as core_layers
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.summary import summary
from tensorflow.python.training import training_util

# The default learning rate of 0.05 is a historical artifact of the initial
# implementation, but seems a reasonable choice.
_LEARNING_RATE = 0.05


def _add_hidden_layer_summary(value, tag):
  summary.scalar('%s_fraction_of_zero_values' % tag, nn.zero_fraction(value))
  summary.histogram('%s_activation' % tag, value)


def _dnn_model_fn(
    features, labels, mode, head, hidden_units, feature_columns,
    optimizer='Adagrad', activation_fn=nn.relu, dropout=None,
    input_layer_partitioner=None, config=None):
  """Deep Neural Net model_fn.

  Args:
    features: Dict of `Tensor` (depends on data passed to `train`).
    labels: `Tensor` of shape [batch_size, 1] or [batch_size] labels of
      dtype `int32` or `int64` in the range `[0, n_classes)`.
    mode: Defines whether this is training, evaluation or prediction.
      See `ModeKeys`.
    head: A `head_lib._Head` instance.
    hidden_units: Iterable of integer number of hidden units per layer.
    feature_columns: Iterable of `feature_column._FeatureColumn` model inputs.
    optimizer: String, `tf.Optimizer` object, or callable that creates the
      optimizer to use for training. If not specified, will use the Adagrad
      optimizer with a default learning rate of 0.05.
    activation_fn: Activation function applied to each layer.
    dropout: When not `None`, the probability we will drop out a given
      coordinate.
    input_layer_partitioner: Partitioner for input layer. Defaults
      to `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
    config: `RunConfig` object to configure the runtime settings.

  Returns:
    predictions: A dict of `Tensor` objects.
    loss: A scalar containing the loss of the step.
    train_op: The op for training.
  """
  optimizer = optimizers.get_optimizer_instance(
      optimizer, learning_rate=_LEARNING_RATE)
  num_ps_replicas = config.num_ps_replicas if config else 0

  partitioner = partitioned_variables.min_max_variable_partitioner(
      max_partitions=num_ps_replicas)
  with variable_scope.variable_scope(
      'dnn',
      values=tuple(six.itervalues(features)),
      partitioner=partitioner):
    input_layer_partitioner = input_layer_partitioner or (
        partitioned_variables.min_max_variable_partitioner(
            max_partitions=num_ps_replicas,
            min_slice_size=64 << 20))
    with variable_scope.variable_scope(
        'input_from_feature_columns',
        values=tuple(six.itervalues(features)),
        partitioner=input_layer_partitioner):
      net = feature_column_lib.input_layer(
          features=features,
          feature_columns=feature_columns)

    for layer_id, num_hidden_units in enumerate(hidden_units):
      with variable_scope.variable_scope(
          'hiddenlayer_%d' % layer_id,
          values=(net,)) as hidden_layer_scope:
        net = core_layers.dense(
            net,
            units=num_hidden_units,
            activation=activation_fn,
            kernel_initializer=init_ops.glorot_uniform_initializer(),
            name=hidden_layer_scope)
        if dropout is not None and mode == model_fn.ModeKeys.TRAIN:
          net = core_layers.dropout(net, rate=dropout, training=True)
      _add_hidden_layer_summary(net, hidden_layer_scope.name)

    with variable_scope.variable_scope(
        'logits',
        values=(net,)) as logits_scope:
      logits = core_layers.dense(
          net,
          units=head.logits_dimension,
          activation=None,
          kernel_initializer=init_ops.glorot_uniform_initializer(),
          name=logits_scope)
    _add_hidden_layer_summary(logits, logits_scope.name)

    def _train_op_fn(loss):
      """Returns the op to optimize the loss."""
      return optimizer.minimize(
          loss,
          global_step=training_util.get_global_step())

    return head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        train_op_fn=_train_op_fn,
        logits=logits)


class DNNRegressor(estimator.Estimator):
  """A regressor for TensorFlow DNN models.

  Example:

  ```python
  sparse_feature_a = sparse_column_with_hash_bucket(...)
  sparse_feature_b = sparse_column_with_hash_bucket(...)

  sparse_feature_a_emb = embedding_column(sparse_id_column=sparse_feature_a,
                                          ...)
  sparse_feature_b_emb = embedding_column(sparse_id_column=sparse_feature_b,
                                          ...)

  estimator = DNNRegressor(
      feature_columns=[sparse_feature_a, sparse_feature_b],
      hidden_units=[1024, 512, 256])

  # Or estimator using the ProximalAdagradOptimizer optimizer with
  # regularization.
  estimator = DNNRegressor(
      feature_columns=[sparse_feature_a, sparse_feature_b],
      hidden_units=[1024, 512, 256],
      optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001
      ))

  # Input builders
  def input_fn_train: # returns x, y
    pass
  estimator.train(input_fn=input_fn_train)

  def input_fn_eval: # returns x, y
    pass
  estimator.evaluate(input_fn=input_fn_eval)
  def input_fn_predict: # returns x, None
    pass
  estimator.predict_scores(input_fn=input_fn_predict)
  ```

  Input of `train` and `evaluate` should have following features,
    otherwise there will be a `KeyError`:

  * if `weight_feature_key` is not `None`, a feature with
    `key=weight_feature_key` whose value is a `Tensor`.
  * for each `column` in `feature_columns`:
    - if `column` is a `SparseColumn`, a feature with `key=column.name`
      whose `value` is a `SparseTensor`.
    - if `column` is a `WeightedSparseColumn`, two features: the first with
      `key` the id column name, the second with `key` the weight column name.
      Both features' `value` must be a `SparseTensor`.
    - if `column` is a `RealValuedColumn`, a feature with `key=column.name`
      whose `value` is a `Tensor`.
  """

  def __init__(self,
               hidden_units,
               feature_columns,
               model_dir=None,
               label_dimension=1,
               weight_feature_key=None,
               optimizer='Adagrad',
               activation_fn=nn.relu,
               dropout=None,
               input_layer_partitioner=None,
               config=None):
    """Initializes a `DNNRegressor` instance.

    Args:
      hidden_units: Iterable of number hidden units per layer. All layers are
        fully connected. Ex. `[64, 32]` means first layer has 64 nodes and
        second one has 32.
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      label_dimension: Number of regression targets per example. This is the
        size of the last dimension of the labels and logits `Tensor` objects
        (typically, these have shape `[batch_size, label_dimension]`).
      weight_feature_key: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example.
      optimizer: An instance of `tf.Optimizer` used to train the model. If
        `None`, will use an Adagrad optimizer.
      activation_fn: Activation function applied to each layer. If `None`, will
        use `tf.nn.relu`.
      dropout: When not `None`, the probability we will drop out a given
        coordinate.
      input_layer_partitioner: Optional. Partitioner for input layer. Defaults
        to `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
      config: `RunConfig` object to configure the runtime settings.

    Returns:
      A `DNNRegressor` estimator.
    """
    def _model_fn(features, labels, mode, config):
      return _dnn_model_fn(
          features=features,
          labels=labels,
          mode=mode,
          # pylint: disable=protected-access
          head=head_lib._regression_head_with_mean_squared_error_loss(
              label_dimension=label_dimension,
              weight_feature_key=weight_feature_key),
          # pylint: enable=protected-access
          hidden_units=hidden_units,
          feature_columns=tuple(feature_columns or []),
          optimizer=optimizer,
          activation_fn=activation_fn,
          dropout=dropout,
          input_layer_partitioner=input_layer_partitioner,
          config=config)
    super(DNNRegressor, self).__init__(
        model_fn=_model_fn, model_dir=model_dir, config=config)
