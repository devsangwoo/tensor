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

=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
"""Registry mechanism for "registering" classes/functions for general use.

This is typically used with a decorator that calls Register for adding
a class or function to a registry.
"""

<<<<<<< HEAD
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import tf_stack
=======
import traceback

from tensorflow.python.platform import logging
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.


# Registry mechanism below is based on mapreduce.python.mrpython.Register.
_LOCATION_TAG = "location"
_TYPE_TAG = "type"


class Registry(object):
  """Provides a registry for saving objects."""

  def __init__(self, name):
    """Creates a new registry."""
    self._name = name
<<<<<<< HEAD
    self._registry = {}
=======
    self._registry = dict()
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  def register(self, candidate, name=None):
    """Registers a Python object "candidate" for the given "name".

    Args:
<<<<<<< HEAD
      candidate: The candidate object to add to the registry.
      name: An optional string specifying the registry key for the candidate.
=======
      candidate: the candidate object to add to the registry.
      name: an optional string specifying the registry key for the candidate.
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
            If None, candidate.__name__ will be used.
    Raises:
      KeyError: If same name is used twice.
    """
    if not name:
      name = candidate.__name__
    if name in self._registry:
<<<<<<< HEAD
      frame = self._registry[name][_LOCATION_TAG]
      raise KeyError(
          "Registering two %s with name '%s'! "
          "(Previous registration was in %s %s:%d)" %
          (self._name, name, frame.name, frame.filename, frame.lineno))
=======
      (filename, line_number, function_name, _) = (
          self._registry[name][_LOCATION_TAG])
      raise KeyError("Registering two %s with name '%s' !"
                     "(Previous registration was in %s %s:%d)" %
                     (self._name, name, function_name, filename, line_number))
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

    logging.vlog(1, "Registering %s (%s) in %s.", name, candidate, self._name)
    # stack trace is [this_function, Register(), user_function,...]
    # so the user function is #2.
<<<<<<< HEAD
    stack = tf_stack.extract_stack(limit=3)
    stack_index = min(2, len(stack)-1)
    if stack_index >= 0:
      location_tag = stack[stack_index]
    else:
      location_tag = ("UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN")
    self._registry[name] = {_TYPE_TAG: candidate, _LOCATION_TAG: location_tag}

  def list(self):
    """Lists registered items.

    Returns:
      A list of names of registered objects.
    """
    return self._registry.keys()
=======
    stack = traceback.extract_stack()
    self._registry[name] = {_TYPE_TAG: candidate, _LOCATION_TAG: stack[2]}
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  def lookup(self, name):
    """Looks up "name".

    Args:
      name: a string specifying the registry key for the candidate.
    Returns:
      Registered object if found
    Raises:
      LookupError: if "name" has not been registered.
    """
<<<<<<< HEAD
    name = compat.as_str(name)
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    if name in self._registry:
      return self._registry[name][_TYPE_TAG]
    else:
      raise LookupError(
          "%s registry has no entry for: %s" % (self._name, name))
