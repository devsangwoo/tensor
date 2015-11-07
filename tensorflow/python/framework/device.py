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

"""Class to represent a device."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from tensorflow.python import tf2
from tensorflow.python.framework import device_spec

if tf2.enabled():
  DeviceSpec = device_spec.DeviceSpecV2
else:
  DeviceSpec = device_spec.DeviceSpecV1
=======
"""Class to represent a device."""
import copy


class Device(object):
  """Represents a Device."""

  def __init__(self, job=None, replica=None, task=None, device_type=None,
               device_index=None):
    """Create a new device object.

    Args:
      job: string.  Optional device job name.
      replica: int.  Optional replica index.
      task: int.  Optional task index.
      device_type: Optional device type string (e.g. "CPU" or "GPU")
      device_index: int.  Optional device index.  If left
        unspecified, device represents 'any' device_index.
    """
    self.job = job
    self.replica = replica
    self.task = task
    if device_type == "cpu" or device_type == "gpu":
      # For backwards compatibility only, we support lowercase variants of
      # cpu and gpu but turn them into uppercase here.
      self.device_type = device_type.upper()
    else:
      self.device_type = device_type
    self.device_index = device_index

  def _clear(self):
    self._job = None
    self._replica = None
    self._task = None
    self.device_type = None
    self.device_index = None

  @property
  def job(self):
    return self._job

  @job.setter
  def job(self, job):
    if job is not None:
      self._job = str(job)
    else:
      self._job = None

  @property
  def replica(self):
    return self._replica

  @replica.setter
  def replica(self, replica):
    if replica is not None:
      self._replica = int(replica)
    else:
      self._replica = None

  @property
  def task(self):
    return self._task

  @task.setter
  def task(self, task):
    if task is not None:
      self._task = int(task)
    else:
      self._task = None

  def parse_from_string(self, spec):
    """Parse a Device name into its components.

    Args:
      spec: a string of the form
       /job:<name>/replica:<id>/task:<id>/device:CPU:<id>
      or
       /job:<name>/replica:<id>/task:<id>/device:GPU:<id>
      as cpu and gpu are mutually exclusive.
      All entries are optional.

    Returns:
      The Device, for convenience.

    Raises:
      ValueError: if the spec was not valid.
    """
    self._clear()
    splits = [x.split(":") for x in spec.split("/")]
    for y in splits:
      ly = len(y)
      if y:
        # NOTE(mdevin): we use the property getters here.
        if ly == 2 and y[0] == "job":
          self.job = y[1]
        elif ly == 2 and y[0] == "replica":
          self.replica = y[1]
        elif ly == 2 and y[0] == "task":
          self.task = y[1]
        elif ((ly == 1 or ly == 2) and
              ((y[0].upper() == "GPU") or (y[0].upper() == "CPU"))):
          if self.device_type is not None:
            raise ValueError("Cannot specify multiple device types: %s" % spec)
          self.device_type = y[0].upper()
          if ly == 2 and y[1] != "*":
            self.device_index = int(y[1])
        elif ly == 3 and y[0] == "device":
          if self.device_type is not None:
            raise ValueError("Cannot specify multiple device types: %s" % spec)
          self.device_type = y[1]
          if y[2] != "*":
            self.device_index = int(y[2])
        elif ly and y[0] != "":  # pylint: disable=g-explicit-bool-comparison
          raise ValueError("Unknown attribute: '%s' in '%s'" % (y[0], spec))

    return self

  def merge_from(self, dev):
    """Merge the properties of "dev" into this Device.

    Args:
      dev: a Device.
    """
    if dev.job is not None:
      self.job = dev.job
    if dev.replica is not None:
      self.replica = dev.replica
    if dev.task is not None:
      self.task = dev.task
    if dev.device_type is not None:
      self.device_type = dev.device_type
    if dev.device_index is not None:
      self.device_index = dev.device_index

  def to_string(self):
    """Return a Device specification string.

    Returns:
      a string of the form /job:<name>/replica:<id>/task:<id>/device:cpu:<id>
      or /job:<name>/replica:<id>/task:<id>/device:cpu:<id>.
    """
    dev = ""
    if self.job is not None:
      dev += "/job:" + self.job
    if self.replica is not None:
      dev += "/replica:" + str(self.replica)
    if self.task is not None:
      dev += "/task:" + str(self.task)
    if self.device_type is not None:
      device_index_string = "*"
      if self.device_index is not None:
        device_index_string = str(self.device_index)
      dev += "/device:%s:%s" % (self.device_type, device_index_string)
    return dev


def from_string(spec):
  """Construct a Device from a string.

  Args:
    spec: a string of the form
     /job:<name>/replica:<id>/task:<id>/device:CPU:<id>
    or
     /job:<name>/replica:<id>/task:<id>/device:GPU:<id>
    as cpu and gpu are mutually exclusive.
    All entries are optional.

  Returns:
    A Device.
  """
  return Device().parse_from_string(spec)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.


def check_valid(spec):
  """Check that a device spec is valid.

  Args:
    spec: a string.

  Raises:
    An exception if the spec is invalid.
  """
<<<<<<< HEAD
  # Construct a DeviceSpec.  It will assert a failure if spec is invalid.
  DeviceSpec.from_string(spec)


def is_device_spec(obj):
  """Abstract away the fact that DeviceSpecV2 is the base class."""
  return isinstance(obj, device_spec.DeviceSpecV2)


def canonical_name(device):
  """Returns a canonical name for the given `DeviceSpec` or device name."""
  if device is None:
    return ""
  if is_device_spec(device):
    return device.to_string()
  else:
    device = DeviceSpec.from_string(device)
    return device.to_string()


# Performance caches
_cached_mergers = {}
_cache_lock = threading.RLock()
_string_merge_cache = {}
=======
  # Construct a device.  It will assert a failure if spec is invalid.
  from_string(spec)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.


def merge_device(spec):
  """Returns a device function that merges devices specifications.

  This can be used to merge partial specifications of devices. The
  innermost setting for a device field takes precedence. For example:

<<<<<<< HEAD
    with tf.device(merge_device("/device:GPU:0"))
      # Nodes created here have device "/device:GPU:0"
      with tf.device(merge_device("/job:worker")):
        # Nodes created here have device "/job:worker/device:GPU:0"
        with tf.device(merge_device("/device:CPU:0")):
          # Nodes created here have device "/job:worker/device:CPU:0"
          with tf.device(merge_device("/job:ps")):
            # Nodes created here have device "/job:ps/device:CPU:0"

  Args:
    spec: A `DeviceSpec` or a device spec string (partially) describing the
=======
    with tf.Device(MergeDevice("/device:GPU:0"))
      # Nodes created here have device "/device:GPU:0"
      with tf.Device(MergeDevice("/job:worker")):
        # Nodes created here have device "/job:worker/device:GPU:0"
        with tf.Device(MergeDevice("/device:CPU:0")):
          # Nodes created here have device "/job:worker/device:CPU:0"
          with tf.Device(MergeDevice("/job:ps")):
            # Nodes created here have device "/job:ps/device:CPU:0"

  Args:
    spec: A device or a device spec string (partially) describing the
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      device that should be used for all nodes created in the scope of
      the returned device function's with block.

  Returns:
<<<<<<< HEAD
    A MergeDevice object with the above-described behavior.
=======
    A device function with the above-described behavior.
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  Raises:
    ValueError: if the spec was not valid.
  """
<<<<<<< HEAD

  if isinstance(spec, MergeDevice):
    return spec

  with _cache_lock:
    merger = _cached_mergers.get(spec)
    if merger:
      return merger

    merger = MergeDevice(spec)
    _cached_mergers[spec] = merger
    return merger


class MergeDevice(object):
  """Wraps a device specification (DeviceSpec or str) with merge functionality.

  When called, this class will merge a node_def with its own spec. It also
  exposes a `shortcut_string_merge` method which can significantly improve
  performance of device placement.
  """

  def __init__(self, spec):
    if isinstance(spec, device_spec.DeviceSpecV2):
      self._spec = spec
    elif isinstance(spec, device_spec.DeviceSpecV1):
      # Capture a snapshot of spec.
      self._spec = spec.__class__.from_string(spec.to_string())
    else:
      self._spec = DeviceSpec.from_string(spec)

  def __call__(self, node_def):
    # In general a user may create a device function which takes into account
    # arbitrary properties of an op. (For instance dynamically placing ops based
    # on type.) So even though the standard DeviceSpec route only uses the
    # device attribute, we take an entire node_def to maintain a consistent
    # signature with general device functions.
    current_device = DeviceSpec.from_string(node_def.device or "")
    return self._spec.make_merged_spec(current_device)

  def shortcut_string_merge(self, node_def):
    """Merge a node def without materializing a full DeviceSpec object.

    Often a device merge is invoked in order to generate a string which can be
    passed into the c api. In such a case, we can cache the
      node_def.device  ->  merge_result_string

    map, and in most cases avoid:
      - Materializing a copy of self._spec (In the case of DeviceSpecV1)
      - Materializing a DeviceSpec for node_def.device
      - A DeviceSpec.merge_from invocation

    In practice the cache hit rate for this function is very high, because the
    number of invocations when iterating through the device stack is much
    larger than the number of devices.

    Args:
      node_def: An Operation (or Operation-like) to merge device constraints
        with self._spec

    Returns:
      A string containing the merged device specification.
    """
    device = node_def.device or ""

    merge_key = (self._spec, device)
    result = _string_merge_cache.get(merge_key)
    if result is None:
      # This update is not atomic, however because the merge is stateless
      # we don't need to lock when updating the cache.
      result = self.__call__(node_def).to_string()
      _string_merge_cache[merge_key] = result

    return result

  def __repr__(self):
    return "{} (spec: {})".format(
        super(MergeDevice, self).__repr__(), self._spec.to_string())

  @property
  def is_null_merge(self):
    """Indicate whether the wrapped spec is empty.

    In the degenerate case where self._spec is an empty specification, a caller
    may wish to skip a merge step entirely. (However this class does not have
    enough information to make that determination.)

    Returns:
      A boolean indicating whether a device merge will be trivial.
    """
    return not bool(self._spec.to_string())
=======
  if not isinstance(spec, Device):
    spec = from_string(spec or "")
  def _device_function(node_def):
    current_device = from_string(node_def.device or "")
    copy_spec = copy.copy(spec)
    copy_spec.merge_from(current_device)  # current_device takes precedence.
    return copy_spec
  return _device_function
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
