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
"""Methods to allow dict of numpy arrays."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from six import string_types
from tensorflow.python.estimator.inputs.queues import feeding_functions

# Key name to pack the target into dict of `features`. See
# `_get_unique_target_key` for details.
_TARGET_KEY = '__target_key__'


def _get_unique_target_key(features):
  """Returns a key not existed in the input dict `features`.

  Caller of `input_fn` usually provides `features` (dict of numpy arrays) and
  `target`, but the underlying feeding module expects a single dict of numpy
  arrays as input. So, the `target` needs to be packed into the `features`
  temporarily and unpacked after calling the feeding function. Toward this goal,
  this function returns a key not existed in the `features` to pack the
  `target`.
  """
  target_key = _TARGET_KEY
  while target_key in features:
    target_key += '_n'
  return target_key


def numpy_input_fn(x,
                   y=None,
                   batch_size=128,
                   num_epochs=1,
                   shuffle=None,
                   queue_capacity=1000,
                   num_threads=1):
  """Returns input function that would feed dict of numpy arrays into the model.

  This returns a function outputting `features` and `targets` based on the dict
  of numpy arrays. The dict `features` has the same keys as the `x`. The dict
  `targets` has the same keys as the `y` if `y` is a dict.

  Example:

  ```python
  age = np.arange(4) * 1.0
  height = np.arange(32, 36)
  x = {'age': age, 'height': height}
  y = np.arange(-32, -28)

  with tf.Session() as session:
    input_fn = numpy_io.numpy_input_fn(
        x, y, batch_size=2, shuffle=False, num_epochs=1)
  ```

  Args:
    x: dict of numpy array object.
    y: numpy array object or dict of numpy array object. `None` if absent.
    batch_size: Integer, size of batches to return.
    num_epochs: Integer, number of epochs to iterate over data. If `None` will
      run forever.
    shuffle: Boolean, if True shuffles the queue. Avoid shuffle at prediction
      time.
    queue_capacity: Integer, size of queue to accumulate.
    num_threads: Integer, number of threads used for reading and enqueueing. In
      order to have predicted and repeatable order of reading and enqueueing,
      such as in prediction and evaluation mode, `num_threads` should be 1.

  Returns:
    Function, that has signature of ()->(dict of `features`, `targets`)

  Raises:
    ValueError: if the shape of `y` mismatches the shape of values in `x` (i.e.,
      values in `x` have same shape).
    ValueError: if duplicate keys are in both `x` and `y` when `y` is a dict.
    ValueError: if x or y is a empty dict.
    TypeError: `x` is not a dict or `shuffle` is not bool.
  """

  if not isinstance(shuffle, bool):
    raise TypeError('shuffle must be explicitly set as boolean; '
                    'got {}'.format(shuffle))

  def input_fn():
    """Numpy input function."""
    if not isinstance(x, dict):
      raise TypeError('x must be dict; got {}'.format(type(x).__name__))
    if not x:
      raise ValueError('x cannot be empty')

    # Make a shadow copy and also ensure the order of iteration is consistent.
    ordered_dict_data = collections.OrderedDict(
        sorted(x.items(), key=lambda t: t[0]))
    # Deep copy keys which is a view in python 3
    feature_keys = list(ordered_dict_data.keys())

    if y is None:
      target_keys = None
    elif isinstance(y, dict):
      if not y:
        raise ValueError('y cannot be empty dict, use None instead.')

      ordered_dict_y = collections.OrderedDict(
        sorted(y.items(), key=lambda t: t[0]))
      target_keys = list(ordered_dict_y.keys())

      duplicate_keys = set(feature_keys).intersection(set(target_keys))
      if len(duplicate_keys):
        raise ValueError('{} duplicate keys are found in both x and y: '
                         '{}'.format(len(duplicate_keys), duplicate_keys))

      ordered_dict_data.update(ordered_dict_y)
    else:
      target_keys = _get_unique_target_key(ordered_dict_data)
      ordered_dict_data[target_keys] = y

    if len(set(v.shape[0] for v in ordered_dict_data.values())) != 1:
      shape_dict_of_x = {k: ordered_dict_data[k].shape
                         for k in feature_keys}

      if target_keys is None:
        shape_of_y = None
      elif isinstance(target_keys, string_types):
        shape_of_y = y.shape
      else:
        shape_of_y = {k: ordered_dict_data[k].shape
                      for k in target_keys}

      raise ValueError('Length of tensors in x and y is mismatched. All '
                       'elements in x and y must have the same length.\n'
                       'Shapes in x: {}\n'
                       'Shapes in y: {}\n'.format(shape_dict_of_x, shape_of_y))

    queue = feeding_functions._enqueue_data(  # pylint: disable=protected-access
        ordered_dict_data,
        queue_capacity,
        shuffle=shuffle,
        num_threads=num_threads,
        enqueue_size=batch_size,
        num_epochs=num_epochs)

    batch = (queue.dequeue_many(batch_size) if num_epochs is None
                else queue.dequeue_up_to(batch_size))

    # Remove the first `Tensor` in `batch`, which is the row number.
    if len(batch) > 0:
      batch.pop(0)

    features = dict(zip(feature_keys, batch[:len(feature_keys)]))
    if target_keys is None:
      return features
    elif isinstance(target_keys, string_types):
      target = batch[-1]
      return features, target
    else:
      target = dict(zip(target_keys, batch[-len(target_keys):]))
      return features, target

  return input_fn
