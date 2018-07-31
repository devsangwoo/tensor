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
"""Iteration over tf.data.Datasets when eager execution is enabled."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.data.python.ops import prefetching_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.training.checkpointable import base as checkpointable
from tensorflow.python.training.saver import BaseSaverBuilder


class Iterator(iterator_ops.EagerIterator, checkpointable.CheckpointableBase):
  """An iterator producing tf.Tensor objects from a tf.data.Dataset.

  NOTE: Unlike the iterator created by the
  @{tf.data.Dataset.make_one_shot_iterator} method, this class enables
  additional experimental functionality, such as prefetching to the GPU.
  """

  def __init__(self, dataset):
    """Creates a new iterator over the given dataset.

    For example:
    ```python
    dataset = tf.data.Dataset.range(4)
    for x in Iterator(dataset):
      print(x)
    ```

    Tensors produced will be placed on the device on which this iterator object
    was created.

    Args:
      dataset: A `tf.data.Dataset` object.

    Raises:
      TypeError: If `dataset` is an unsupported type.
      RuntimeError: When invoked without eager execution enabled.
    """
    if isinstance(dataset, prefetching_ops._PrefetchToDeviceDataset):  # pylint: disable=protected-access
      raise TypeError(
          "`tf.contrib.data.prefetch_to_device()` is not compatible with "
          "`tf.contrib.eager.Iterator`. Use `for ... in dataset:` to iterate "
          "over the dataset instead.")

    if not context.context().device_spec.device_type:
      is_remote_device = False
    else:
      is_remote_device = context.context().device_spec.device_type != "CPU"
    if is_remote_device:
      with ops.device(None):
        # Let the placer figure out where to place the various functions etc.
        # created by the CopyToDeviceDataset.
        dataset = dataset.apply(prefetching_ops.copy_to_device(
            context.context().device_name))
        dataset = dataset.prefetch(1)
    super(Iterator, self).__init__(dataset)

  def _next_internal(self):
    """Returns a nested structure of `tf.Tensor`s containing the next element.
    """
    # This runs in sync mode as iterators use an error status to communicate
    # that there is no more data to iterate over.
    # TODO(b/77291417): Fix
    with context.execution_mode(context.SYNC):
      return super(Iterator, self)._next_internal()

  # TODO(shivaniagrawal): Expose checkpointable stateful objects from dataset
  # attributes(potential).

  class _Saveable(BaseSaverBuilder.SaveableObject):
    """SaveableObject for saving/restoring iterator state."""

    def __init__(self, iterator_resource, name):
      serialized_iterator = gen_dataset_ops.serialize_iterator(
          iterator_resource)
      specs = [
          BaseSaverBuilder.SaveSpec(serialized_iterator, "", name + "_STATE")
      ]
      # pylint: disable=protected-access
      super(Iterator._Saveable, self).__init__(iterator_resource, specs, name)

    def restore(self, restored_tensors, restored_shapes):
      with ops.colocate_with(self.op):
        return gen_dataset_ops.deserialize_iterator(self.op,
                                                    restored_tensors[0])

  def _gather_saveables_for_checkpoint(self):

    def _saveable_factory(name):
      return self._Saveable(self._resource, name)

    return {"ITERATOR": _saveable_factory}
