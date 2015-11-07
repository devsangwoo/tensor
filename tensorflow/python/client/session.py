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
"""A client interface for TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import re
import threading
import warnings

import numpy as np
import wrapt

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_tensorflow as tf_session
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import device
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import session_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.experimental import mixed_precision_global_state
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.util.compat import collections_abc

_python_session_create_counter = monitoring.Counter(
    '/tensorflow/api/python/session_create_counter',
    'Counter for number of sessions created in Python.')
=======
"""A client interface for TensorFlow."""

import re
import sys
import threading

import tensorflow.python.platform

import numpy as np

from tensorflow.python import pywrap_tensorflow as tf_session
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import logging

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

class SessionInterface(object):
  """Base class for implementations of TensorFlow client sessions."""

  @property
  def graph(self):
    """The underlying TensorFlow graph, to be used in building Operations."""
    raise NotImplementedError('graph')

  @property
  def sess_str(self):
    """The TensorFlow process to which this session will connect."""
    raise NotImplementedError('sess_str')

<<<<<<< HEAD
  def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
    """Runs operations in the session. See `BaseSession.run()` for details."""
    raise NotImplementedError('run')

  def partial_run_setup(self, fetches, feeds=None):
    """Sets up the feeds and fetches for partial runs in the session."""
    raise NotImplementedError('partial_run_setup')

  def partial_run(self, handle, fetches, feed_dict=None):
    """Continues the execution with additional feeds and fetches."""
    raise NotImplementedError('partial_run')


def _get_indexed_slices_value_from_fetches(fetched_vals):
  return ops.IndexedSlicesValue(
      fetched_vals[0], fetched_vals[1],
      fetched_vals[2] if len(fetched_vals) == 3 else None)


def _get_feeds_for_indexed_slices(feed, feed_val):
  return list(
      zip([feed.values, feed.indices] if feed.dense_shape is None else
          [feed.values, feed.indices, feed.dense_shape], feed_val))


# List of extensions supported to convert run arguments into actual fetches and
# feeds.
#
# Each element in the list is a tuple of (Type, fetch_fn, feed_fn1, feed_fn2),
# where the function signatures are:
#   fetch_fn : Type -> (list of Tensors,
#                       lambda: list of fetched np.ndarray -> TypeVal)
#   feed_fn1 : Type, TypeVal -> list of (Tensor, value)
#   feed_fn2 : Type -> list of Tensors
#
# `fetch_fn` describes how to expand fetch into its
# component Tensors and how to contract the fetched results back into
# a single return value.
#
# Each feed function describes how to unpack a single fed value and map it to
# feeds of one or more tensors and their corresponding values: `feed_fn1` is
# used to feed a run, `feed_fn2` to set up a partial run.
#
# TODO(touts): We could reimplement these as specialized _FeedMapper
# implementations after we refactor the feed handling code to use them.
#
# Eventually, this registration could be opened up to support custom Tensor
# expansions.
# pylint: disable=g-long-lambda
_REGISTERED_EXPANSIONS = [
    # SparseTensors are fetched as SparseTensorValues. They can be fed
    # SparseTensorValues or normal tuples.
    (sparse_tensor.SparseTensor, lambda fetch: ([
        fetch.indices, fetch.values, fetch.dense_shape
    ], lambda fetched_vals: sparse_tensor.SparseTensorValue(*fetched_vals)),
     lambda feed, feed_val: list(
         zip([feed.indices, feed.values, feed.dense_shape], feed_val)),
     lambda feed: [feed.indices, feed.values, feed.dense_shape]),
    # IndexedSlices are fetched as IndexedSlicesValues. They can be fed
    # IndexedSlicesValues or normal tuples.
    (ops.IndexedSlices,
     lambda fetch: ([fetch.values, fetch.indices] if fetch.dense_shape is None
                    else [fetch.values, fetch.indices, fetch.dense_shape
                         ], _get_indexed_slices_value_from_fetches),
     _get_feeds_for_indexed_slices,
     lambda feed: [feed.values, feed.indices] if feed.dense_shape is None else
     [feed.values, feed.indices, feed.dense_shape]),
    # The default catches all other types and performs no expansions.
    (object, lambda fetch: ([fetch], lambda fetched_vals: fetched_vals[0]),
     lambda feed, feed_val: [(feed, feed_val)], lambda feed: [feed])
]

# pylint: enable=g-long-lambda


def _convert_to_numpy_obj(numpy_dtype, obj):
  """Explicitly convert obj based on numpy type except for string type."""
  return numpy_dtype(obj) if numpy_dtype is not object else str(obj)


def register_session_run_conversion_functions(
    tensor_type,
    fetch_function,
    feed_function=None,
    feed_function_for_partial_run=None):
  """Register fetch and feed conversion functions for `tf.Session.run()`.

  This function registers a triple of conversion functions for fetching and/or
  feeding values of user-defined types in a call to tf.Session.run().

  An example

  ```python
     class SquaredTensor(object):
       def __init__(self, tensor):
         self.sq = tf.square(tensor)
     #you can define conversion functions as follows:
     fetch_function = lambda squared_tensor:([squared_tensor.sq],
                                             lambda val: val[0])
     feed_function = lambda feed, feed_val: [(feed.sq, feed_val)]
     feed_function_for_partial_run = lambda feed: [feed.sq]
     #then after invoking this register function, you can use as follows:
     session.run(squared_tensor1,
                 feed_dict = {squared_tensor2 : some_numpy_array})
  ```

  Args:
    tensor_type: The type for which you want to register a conversion function.
    fetch_function: A callable that takes an object of type `tensor_type` and
      returns a tuple, where the first element is a list of `tf.Tensor` objects,
      and the second element is a callable that takes a list of ndarrays and
      returns an object of some value type that corresponds to `tensor_type`.
      fetch_function describes how to expand fetch into its component Tensors
      and how to contract the fetched results back into a single return value.
    feed_function: A callable that takes feed_key and feed_value as input, and
      returns a list of tuples (feed_tensor, feed_val), feed_key must have type
      `tensor_type`, and feed_tensor must have type `tf.Tensor`. Each feed
      function describes how to unpack a single fed value and map it to feeds of
      one or more tensors and their corresponding values.
    feed_function_for_partial_run: A callable for specifying tensor values to
      feed when setting up a partial run, which takes a `tensor_type` type
      object as input, and returns a list of Tensors.

  Raises:
    ValueError: If `tensor_type` has already been registered.
  """
  for conversion_function in _REGISTERED_EXPANSIONS:
    if issubclass(conversion_function[0], tensor_type):
      raise ValueError('%s has already been registered so ignore it.' %
                       tensor_type)

  _REGISTERED_EXPANSIONS.insert(0, (tensor_type, fetch_function, feed_function,
                                    feed_function_for_partial_run))


def _is_attrs_instance(obj):
  """Returns True if the given obj is an instance of attrs-decorated class."""
  return getattr(obj.__class__, '__attrs_attrs__', None) is not None


def _get_attrs_values(obj):
  """Returns the list of values from an attrs instance."""
  attrs = getattr(obj.__class__, '__attrs_attrs__')
  return [getattr(obj, a.name) for a in attrs]


class _FetchMapper(object):
  """Definition of the interface provided by fetch mappers.

  Fetch mappers are utility classes used by the _FetchHandler to handle
  arbitrary structures for the `fetch` argument to `Session.run()`.

  The `fetch` argument can be of various shapes: single tensor or op, list of
  fetches, tuple of fetches, namedtuple of fetches, or dict of fetches.  The
  structures can be arbitrarily nested.

  The low level run() API only wants a list of tensor or op names.  The various
  `_FetchMapper` subclasses below take care of handling the different shapes:
  uniquifying the fetches, and constructing results with the original shape.
  """

  def unique_fetches(self):
    """Return the list of unique tensors or ops needed by this fetch mapper.

    Returns:
      A list of tensors or ops.
    """
    raise NotImplementedError('Must be implemented by subclasses')

  def build_results(self, values):
    """Build results that match the original shape of the fetch.

    Args:
      values: List of values returned by run(). The values correspond exactly to
        the list tensors or ops returned by unique_fetches().

    Returns:
      A struct of the same shape as the original fetch object handled by
      this fetch mapper.  In the returned struct, the original fetches are
      replaced by their fetched values.
    """
    raise NotImplementedError('Must be implemented by subclasses')

  @staticmethod
  def for_fetch(fetch):
    """Creates fetch mapper that handles the structure of `fetch`.

    The default graph must be the one from which we want to fetch values when
    this function is called.

    Args:
      fetch: An arbitrary fetch structure: singleton, list, tuple, namedtuple,
        or dict.

    Returns:
      An instance of a subclass of `_FetchMapper` that handles the shape.
    """
    if fetch is None:
      raise TypeError('Fetch argument %r has invalid type %r' %
                      (fetch, type(fetch)))
    elif isinstance(fetch, (list, tuple)):
      # NOTE(touts): This is also the code path for namedtuples.
      return _ListFetchMapper(fetch)
    elif isinstance(fetch, collections_abc.Mapping):
      return _DictFetchMapper(fetch)
    elif _is_attrs_instance(fetch):
      return _AttrsFetchMapper(fetch)
    else:
      # Look for a handler in the registered expansions.
      for tensor_type, fetch_fn, _, _ in _REGISTERED_EXPANSIONS:
        if isinstance(fetch, tensor_type):
          fetches, contraction_fn = fetch_fn(fetch)
          return _ElementFetchMapper(fetches, contraction_fn)
    # Did not find anything.
    raise TypeError('Fetch argument %r has invalid type %r' %
                    (fetch, type(fetch)))


class _ElementFetchMapper(_FetchMapper):
  """Fetch mapper for singleton tensors and ops."""

  def __init__(self, fetches, contraction_fn):
    """Creates an _ElementFetchMapper.

    This is the fetch mapper used for leaves in the fetch struct.  Because of
    the expansions mechanism, a leaf can actually fetch more than one tensor.

    Also note that the fetches here can be just strings (tensor or op names) or
    any other object that the graph knows how to convert to a tensor, such as a
    Variable.  So we have to run each fetch through `as_graph_element()` to get
    the corresponding tensor or op.

    Args:
      fetches: List of objects, as returned by a fetch_fn defined in
        _REGISTERED_EXPANSIONS.
      contraction_fn: Callable as returned by a fetch_fn.
    """
    self._unique_fetches = []
    for fetch in fetches:
      try:
        self._unique_fetches.append(ops.get_default_graph().as_graph_element(
            fetch, allow_tensor=True, allow_operation=True))
      except TypeError as e:
        raise TypeError('Fetch argument %r has invalid type %r, '
                        'must be a string or Tensor. (%s)' %
                        (fetch, type(fetch), str(e)))
      except ValueError as e:
        raise ValueError('Fetch argument %r cannot be interpreted as a '
                         'Tensor. (%s)' % (fetch, str(e)))
      except KeyError as e:
        raise ValueError('Fetch argument %r cannot be interpreted as a '
                         'Tensor. (%s)' % (fetch, str(e)))
    self._contraction_fn = contraction_fn

  def unique_fetches(self):
    return self._unique_fetches

  def build_results(self, values):
    if not values:
      # 'Operation' case
      return None
    else:
      return self._contraction_fn(values)


def _uniquify_fetches(fetch_mappers):
  """Uniquifies fetches from a list of fetch_mappers.

  This is a utility function used by _ListFetchMapper and _DictFetchMapper.  It
  gathers all the unique fetches from a list of mappers and builds a list
  containing all of them but without duplicates (unique_fetches).

  It also returns a 2-D list of integers (values_indices) indicating at which
  index in unique_fetches the fetches of the mappers are located.

  This list is as follows:
    values_indices[mapper_index][mapper_fetch_index] = unique_fetches_index

  Args:
    fetch_mappers: list of fetch mappers.

  Returns:
    A list of fetches.
    A 2-D list of integers.
  """
  unique_fetches = []
  value_indices = []
  seen_fetches = {}
  for m in fetch_mappers:
    m_value_indices = []
    for f in m.unique_fetches():
      j = seen_fetches.get(id(f))
      if j is None:
        j = len(seen_fetches)
        seen_fetches[id(f)] = j
        unique_fetches.append(f)
      m_value_indices.append(j)
    value_indices.append(m_value_indices)
  return unique_fetches, value_indices


class _ListFetchMapper(_FetchMapper):
  """Fetch mapper for lists, tuples, and namedtuples."""

  def __init__(self, fetches):
    """Creates a _ListFetchMapper.

    Args:
      fetches: List, tuple, or namedtuple of fetches.
    """
    if isinstance(fetches, wrapt.ObjectProxy):
      self._fetch_type = type(fetches.__wrapped__)
    else:
      self._fetch_type = type(fetches)
    self._mappers = [_FetchMapper.for_fetch(fetch) for fetch in fetches]
    self._unique_fetches, self._value_indices = _uniquify_fetches(self._mappers)

  def unique_fetches(self):
    return self._unique_fetches

  def build_results(self, values):
    # Create the list of results for each mapper.
    results = []
    for m, vi in zip(self._mappers, self._value_indices):
      results.append(m.build_results([values[j] for j in vi]))
    # Return a value of the original type of the fetches.
    if issubclass(self._fetch_type, list):
      return results
    elif self._fetch_type == tuple:
      return tuple(results)
    else:
      # This is the code path for namedtuple.
      return self._fetch_type(*results)


class _DictFetchMapper(_FetchMapper):
  """Fetch mapper for dicts."""

  def __init__(self, fetches):
    """Creates a _DictFetchMapper.

    Args:
      fetches: Dict of fetches.
    """
    self._fetch_type = type(fetches)
    self._keys = fetches.keys()
    self._mappers = [
        _FetchMapper.for_fetch(fetch) for fetch in fetches.values()
    ]
    self._unique_fetches, self._value_indices = _uniquify_fetches(self._mappers)

  def unique_fetches(self):
    return self._unique_fetches

  def build_results(self, values):
    results = self._fetch_type()
    for k, m, vi in zip(self._keys, self._mappers, self._value_indices):
      results[k] = m.build_results([values[j] for j in vi])
    return results


class _AttrsFetchMapper(_FetchMapper):
  """Fetch mapper for attrs decorated classes."""

  def __init__(self, fetches):
    """Creates a _AttrsFetchMapper.

    Args:
      fetches: An instance of an attrs decorated class.
    """
    values = _get_attrs_values(fetches)
    self._fetch_type = type(fetches)
    self._mappers = [_FetchMapper.for_fetch(fetch) for fetch in values]
    self._unique_fetches, self._value_indices = _uniquify_fetches(self._mappers)

  def unique_fetches(self):
    return self._unique_fetches

  def build_results(self, values):
    results = []
    for m, vi in zip(self._mappers, self._value_indices):
      results.append(m.build_results([values[j] for j in vi]))
    return self._fetch_type(*results)


class _FetchHandler(object):
  """Handler for structured fetches.

  Given a graph, a user-provided structure for fetches, and a feed dict, this
  class takes care of generating a list of tensor names to fetch and op names
  to run for a low level `run()` call.

  Given the results of the low level run call, this class can also rebuild a
  result structure matching the user-provided structure for fetches, but
  containing the corresponding results.
  """

  # TODO(touts): Make this class also take care of destructuring the feed
  # dict instead of doing it in the callers.

  def __init__(self, graph, fetches, feeds, feed_handles=None):
    """Creates a fetch handler.

    Args:
      graph: Graph of the fetches.   Used to check for fetchability and to
        convert all fetches to tensors or ops as needed.
      fetches: An arbitrary fetch structure: singleton, list, tuple, namedtuple,
        or dict.
      feeds: A feed dict where keys are Tensors.
      feed_handles: A dict from feed Tensors to TensorHandle objects used as
        direct feeds.
    """
    with graph.as_default():
      self._fetch_mapper = _FetchMapper.for_fetch(fetches)
    self._fetches = []
    self._targets = []
    self._feeds = feeds
    self._feed_handles = feed_handles or {}
    self._ops = []
    self._fetch_handles = {}
    for fetch in self._fetch_mapper.unique_fetches():
      if isinstance(fetch, ops.Operation):
        self._assert_fetchable(graph, fetch)
        self._targets.append(fetch)
        self._ops.append(True)
      else:
        self._assert_fetchable(graph, fetch.op)
        self._fetches.append(fetch)
        self._ops.append(False)
      # Remember the fetch if it is for a tensor handle.
      if (isinstance(fetch, ops.Tensor) and
          (fetch.op.type == 'GetSessionHandle' or
           fetch.op.type == 'GetSessionHandleV2')):
        self._fetch_handles[fetch.experimental_ref()] = fetch.op.inputs[0].dtype
    self._final_fetches = [
        x for x in self._fetches if x.experimental_ref() not in feeds
    ]

  def _assert_fetchable(self, graph, op):
    if not graph.is_fetchable(op):
      raise errors.InaccessibleTensorError(
          'Operation %r has been marked as not fetchable. Typically this'
          ' happens when it is defined in another function or code block.'
          ' Use return values,explicit Python locals or TensorFlow collections'
          ' to access it.'
          % op.name)

  def fetches(self):
    """Return the unique names of tensors to fetch.

    Returns:
      A list of strings.
    """
    return self._final_fetches

  def targets(self):
    """Return the unique names of ops to run.

    Returns:
      A list of strings.
    """
    return self._targets

  def build_results(self, session, tensor_values):
    """Build results matching the original fetch shape.

    `tensor_values` must be a list of the same length as
    the one returned by `fetches()`, and holding the requested
    fetch values.

    This method builds a struct with the same shape as the original `fetches`
    passed to the constructor, in which the fetches are replaced by their
    fetched value.

    Args:
      session: The enclosing session.  Used for tensor handles.
      tensor_values: List of values matching the list returned by fetches().

    Returns:
      A structure of the same shape as the original `fetches` argument but
        containing tensors or None (for fetched ops).
    """
    full_values = []
    assert len(self._final_fetches) == len(tensor_values)
    i = 0
    j = 0
    for is_op in self._ops:
      if is_op:
        full_values.append(None)
      else:
        # If the fetch was in the feeds, use the fed value, otherwise
        # use the returned value.
        if self._fetches[i].experimental_ref() in self._feed_handles:
          # A fetch had a corresponding direct TensorHandle feed. Call eval()
          # to obtain the Tensor value from the TensorHandle.
          value = self._feed_handles[self._fetches[i].experimental_ref()].eval()
        else:
          value = self._feeds.get(self._fetches[i].experimental_ref())
        if value is None:
          value = tensor_values[j]
          j += 1
        dtype = self._fetch_handles.get(self._fetches[i].experimental_ref())
        if dtype:
          full_values.append(session_ops.TensorHandle(value, dtype, session))
        else:
          full_values.append(value)
        i += 1
    assert j == len(tensor_values)
    return self._fetch_mapper.build_results(full_values)


def _name_list(tensor_list):
  """Utility function for transitioning to the new session API.

  Args:
    tensor_list: a list of `Tensor`s.

  Returns:
    A list of each `Tensor`s name (as byte arrays).
  """
  return [compat.as_bytes(t.name) for t in tensor_list]


class _DeviceAttributes(object):
  """Struct-like object describing a device's attributes.

  Each device has 3 key properties:
   - name: the fully-qualified TensorFlow path to the device. For
        example: /job:worker/replica:0/task:3/device:CPU:0
   - device_type: the type of the device (e.g. CPU, GPU, TPU, etc.)
   - memory_limit_bytes: the maximum amount of memory available on the device
        (in bytes).
  """

  def __init__(self, name, device_type, memory_limit_bytes, incarnation):
    self._name = device.canonical_name(name)
    self._device_type = device_type
    self._memory_limit_bytes = memory_limit_bytes
    self._incarnation = incarnation

  @property
  def name(self):
    return self._name

  @property
  def device_type(self):
    return self._device_type

  @property
  def memory_limit_bytes(self):
    return self._memory_limit_bytes

  @property
  def incarnation(self):
    return self._incarnation

  def __repr__(self):
    return '_DeviceAttributes(%s, %s, %d, %d)' % (
        self.name,
        self.device_type,
        self.memory_limit_bytes,
        self.incarnation,
    )
=======
  def run(self, fetches, feed_dict=None):
    """Runs operations in the session. See `Session.run()` for details."""
    raise NotImplementedError('Run')
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.


class BaseSession(SessionInterface):
  """A class for interacting with a TensorFlow computation.

  The BaseSession enables incremental graph building with inline
  execution of Operations and evaluation of Tensors.
  """

  def __init__(self, target='', graph=None, config=None):
    """Constructs a new TensorFlow session.

    Args:
      target: (Optional) The TensorFlow execution engine to connect to.
<<<<<<< HEAD
      graph: (Optional) The graph to be used. If this argument is None, the
        default graph will be used.
      config: (Optional) ConfigProto proto used to configure the session. If no
        config is specified, the global default will be used. The global default
        can be configured via the tf.config APIs.

    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        creating the TensorFlow session.
      TypeError: If one of the arguments has the wrong type.
    """
    _python_session_create_counter.get_cell().increase_by(1)
    if graph is None:
      self._graph = ops.get_default_graph()
    else:
      if not isinstance(graph, ops.Graph):
        raise TypeError('graph must be a tf.Graph, but got %s' % type(graph))
      self._graph = graph

    self._closed = False

    if target is not None:
      try:
        self._target = compat.as_bytes(target)
      except TypeError:
        if isinstance(target, config_pb2.ConfigProto):
          raise TypeError('target must be a string, but got %s.'
                          ' Did you do "Session(config)" instead of'
                          ' "Session(config=config)"?' % type(target))
        raise TypeError('target must be a string, but got %s' % type(target))
    else:
      self._target = None

    self._delete_lock = threading.Lock()
    self._dead_handles = []

    if config is None:
      config = context.context().config

    if not isinstance(config, config_pb2.ConfigProto):
      raise TypeError('config must be a tf.ConfigProto, but got %s' %
                      type(config))

    if (mixed_precision_global_state.mixed_precision_graph_rewrite_is_enabled
        and config.graph_options.rewrite_options.auto_mixed_precision !=
        rewriter_config_pb2.RewriterConfig.OFF):
      new_config = config_pb2.ConfigProto()
      new_config.CopyFrom(config)
      new_config.graph_options.rewrite_options.auto_mixed_precision = (
          rewriter_config_pb2.RewriterConfig.ON)
      config = new_config
    elif (config.graph_options.rewrite_options.auto_mixed_precision !=
          rewriter_config_pb2.RewriterConfig.ON):
      mixed_precision_global_state.non_mixed_precision_session_created = True

    self._config = config
    self._add_shapes = config.graph_options.infer_shapes

    self._session = None
    opts = tf_session.TF_NewSessionOptions(target=self._target, config=config)
    try:
      # pylint: disable=protected-access
      self._session = tf_session.TF_NewSessionRef(self._graph._c_graph, opts)
      # pylint: enable=protected-access
    finally:
      tf_session.TF_DeleteSessionOptions(opts)

  def list_devices(self):
    """Lists available devices in this session.

    ```python
    devices = sess.list_devices()
    for d in devices:
      print(d.name)
    ```

    Where:
      Each element in the list has the following properties
      name: A string with the full name of the device. ex:
          `/job:worker/replica:0/task:3/device:CPU:0`
      device_type: The type of the device (e.g. `CPU`, `GPU`, `TPU`.)
      memory_limit: The maximum amount of memory available on the device.
          Note: depending on the device, it is possible the usable memory could
          be substantially less.

    Raises:
      tf.errors.OpError: If it encounters an error (e.g. session is in an
      invalid state, or network errors occur).

    Returns:
      A list of devices in the session.
    """
    raw_device_list = tf_session.TF_SessionListDevices(self._session)
    device_list = []
    size = tf_session.TF_DeviceListCount(raw_device_list)
    for i in range(size):
      name = tf_session.TF_DeviceListName(raw_device_list, i)
      device_type = tf_session.TF_DeviceListType(raw_device_list, i)
      memory = tf_session.TF_DeviceListMemoryBytes(raw_device_list, i)
      incarnation = tf_session.TF_DeviceListIncarnation(raw_device_list, i)
      device_list.append(
          _DeviceAttributes(name, device_type, memory, incarnation))
    tf_session.TF_DeleteDeviceList(raw_device_list)
    return device_list
=======
      graph: (Optional) The graph to be used. If this argument is None,
        the default graph will be used.
      config: (Optional) ConfigProto proto used to configure the session.

    Raises:
      RuntimeError: If an error occurs while creating the TensorFlow
        session.
    """
    if graph is None:
      self._graph = ops.get_default_graph()
    else:
      self._graph = graph

    self._opened = False
    self._closed = False

    self._current_version = 0
    self._extend_lock = threading.Lock()
    self._target = target

    self._session = None

    try:
      opts = tf_session.TF_NewSessionOptions(target=target, config=config)
      status = tf_session.TF_NewStatus()
      self._session = tf_session.TF_NewSession(opts, status)
      if tf_session.TF_GetCode(status) != 0:
        message = tf_session.TF_Message(status)
        raise RuntimeError(message)

    finally:
      tf_session.TF_DeleteSessionOptions(opts)
      tf_session.TF_DeleteStatus(status)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  def close(self):
    """Closes this session.

    Calling this method frees all resources associated with the session.

    Raises:
<<<<<<< HEAD
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        closing the TensorFlow session.
    """
    if self._session and not self._closed:
      self._closed = True
      tf_session.TF_CloseSession(self._session)

  def __del__(self):
    # cleanly ignore all exceptions
    try:
      self.close()
    except Exception:  # pylint: disable=broad-except
      pass
    if self._session is not None:
      try:
        tf_session.TF_DeleteSession(self._session)
      except (AttributeError, TypeError):
        # At shutdown, `c_api_util`, `tf_session`, or
        # `tf_session.TF_DeleteSession` may have been garbage collected, causing
        # the above method calls to fail. In this case, silently leak since the
        # program is about to terminate anyway.
        pass
      self._session = None
=======
      RuntimeError: If an error occurs while closing the session.
    """
    with self._extend_lock:
      if self._opened and not self._closed:
        self._closed = True
        try:
          status = tf_session.TF_NewStatus()
          tf_session.TF_CloseSession(self._session, status)
          if tf_session.TF_GetCode(status) != 0:
            raise RuntimeError(tf_session.TF_Message(status))
        finally:
          tf_session.TF_DeleteStatus(status)

  def __del__(self):
    self.close()
    try:
      status = tf_session.TF_NewStatus()
      if self._session is not None:
        tf_session.TF_DeleteSession(self._session, status)
        if tf_session.TF_GetCode(status) != 0:
          raise RuntimeError(tf_session.TF_Message(status))
        self._session = None
    finally:
      tf_session.TF_DeleteStatus(status)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  @property
  def graph(self):
    """The graph that was launched in this session."""
    return self._graph

  @property
  def graph_def(self):
    """A serializable version of the underlying TensorFlow graph.

    Returns:
      A graph_pb2.GraphDef proto containing nodes for all of the Operations in
      the underlying TensorFlow graph.
    """
<<<<<<< HEAD
    return self._graph.as_graph_def(add_shapes=self._add_shapes)
=======
    return self._graph.as_graph_def()
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  @property
  def sess_str(self):
    return self._target

  def as_default(self):
    """Returns a context manager that makes this object the default session.

    Use with the `with` keyword to specify that calls to
<<<<<<< HEAD
    `tf.Operation.run` or `tf.Tensor.eval` should be executed in
=======
    [`Operation.run()`](framework.md#Operation.run) or
    [`Tensor.run()`](framework.md#Tensor.run) should be executed in
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    this session.

    ```python
    c = tf.constant(..)
<<<<<<< HEAD
    sess = tf.compat.v1.Session()

    with sess.as_default():
      assert tf.compat.v1.get_default_session() is sess
      print(c.eval())
    ```

    To get the current default session, use `tf.compat.v1.get_default_session`.
=======
    sess = tf.Session()

    with sess.as_default():
      assert tf.get_default_session() is sess
      print c.eval()
    ```

    To get the current default session, use
    [`tf.get_default_session()`](#get_default_session).

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

    *N.B.* The `as_default` context manager *does not* close the
    session when you exit the context, and you must close the session
    explicitly.

    ```python
    c = tf.constant(...)
<<<<<<< HEAD
    sess = tf.compat.v1.Session()
    with sess.as_default():
      print(c.eval())
    # ...
    with sess.as_default():
      print(c.eval())
=======
    sess = tf.Session()
    with sess.as_default():
      print c.eval()
    # ...
    with sess.as_default():
      print c.eval()
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

    sess.close()
    ```

<<<<<<< HEAD
    Alternatively, you can use `with tf.compat.v1.Session():` to create a
    session that is automatically closed on exiting the context,
    including when an uncaught exception is raised.

    *N.B.* The default session is a property of the current thread. If you
=======
    Alternatively, you can use `with tf.Session():` to create a
    session that is automatically closed on exiting the context,
    including when an uncaught exception is raised.

    *N.B.* The default graph is a property of the current thread. If you
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    create a new thread, and wish to use the default session in that
    thread, you must explicitly add a `with sess.as_default():` in that
    thread's function.

<<<<<<< HEAD
    *N.B.* Entering a `with sess.as_default():` block does not affect
    the current default graph. If you are using multiple graphs, and
    `sess.graph` is different from the value of
    `tf.compat.v1.get_default_graph`, you must explicitly enter a
    `with sess.graph.as_default():` block to make `sess.graph` the default
    graph.

    Returns:
      A context manager using this session as the default session.
    """
    return ops.default_session(self)

  def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
    """Runs operations and evaluates tensors in `fetches`.
=======
    Returns:
      A context manager using this session as the default session.

    """
    return ops.default_session(self)

  # Eventually, this registration could be opened up to support custom
  # Tensor expansions. Expects tuples of (Type, fetch_fn, feed_fn),
  # where the signatures are:
  #   fetch_fn : Type -> (list of Tensors,
  #                       lambda: list of fetched np.ndarray -> TypeVal)
  #   feed_fn  : Type, TypeVal -> list of (Tensor, value)
  # Conceptually, fetch_fn describes how to expand fetch into its
  # component Tensors and how to contracting the fetched results back into
  # a single return value. feed_fn describes how to unpack a single fed
  # value and map it to feeds of a Tensor and its corresponding value.
  # pylint: disable=g-long-lambda
  _REGISTERED_EXPANSIONS = [
      # SparseTensors are fetched as SparseTensorValues. They can be fed
      # SparseTensorValues or normal tuples.
      (ops.SparseTensor,
       lambda fetch: (
           [fetch.indices, fetch.values, fetch.shape],
           lambda fetched_vals: ops.SparseTensorValue(*fetched_vals)),
       lambda feed, feed_val: list(zip(
           [feed.indices, feed.values, feed.shape], feed_val))),
      # The default catches all types and performs no expansions.
      (object,
       lambda fetch: ([fetch], lambda fetched_vals: fetched_vals[0]),
       lambda feed, feed_val: [(feed, feed_val)])]
  # pylint: enable=g-long-lambda

  def run(self, fetches, feed_dict=None):
    """Runs the operations and evaluates the tensors in `fetches`.
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

    This method runs one "step" of TensorFlow computation, by
    running the necessary graph fragment to execute every `Operation`
    and evaluate every `Tensor` in `fetches`, substituting the values in
    `feed_dict` for the corresponding input values.

<<<<<<< HEAD
    The `fetches` argument may be a single graph element, or an arbitrarily
    nested list, tuple, namedtuple, dict, or OrderedDict containing graph
    elements at its leaves.  A graph element can be one of the following types:

    * A `tf.Operation`.
      The corresponding fetched value will be `None`.
    * A `tf.Tensor`.
      The corresponding fetched value will be a numpy ndarray containing the
      value of that tensor.
    * A `tf.SparseTensor`.
      The corresponding fetched value will be a
      `tf.compat.v1.SparseTensorValue`
      containing the value of that sparse tensor.
    * A `get_tensor_handle` op.  The corresponding fetched value will be a
      numpy ndarray containing the handle of that tensor.
    * A `string` which is the name of a tensor or operation in the graph.

    The value returned by `run()` has the same shape as the `fetches` argument,
    where the leaves are replaced by the corresponding values returned by
    TensorFlow.

    Example:

    ```python
       a = tf.constant([10, 20])
       b = tf.constant([1.0, 2.0])
       # 'fetches' can be a singleton
       v = session.run(a)
       # v is the numpy array [10, 20]
       # 'fetches' can be a list.
       v = session.run([a, b])
       # v is a Python list with 2 numpy arrays: the 1-D array [10, 20] and the
       # 1-D array [1.0, 2.0]
       # 'fetches' can be arbitrary lists, tuples, namedtuple, dicts:
       MyData = collections.namedtuple('MyData', ['a', 'b'])
       v = session.run({'k1': MyData(a, b), 'k2': [b, a]})
       # v is a dict with
       # v['k1'] is a MyData namedtuple with 'a' (the numpy array [10, 20]) and
       # 'b' (the numpy array [1.0, 2.0])
       # v['k2'] is a list with the numpy array [1.0, 2.0] and the numpy array
       # [10, 20].
    ```
=======
    The `fetches` argument may be a list of graph elements or a single
    graph element, and these determine the return value of this
    method. A graph element can be one of the following types:

    * If the *i*th element of `fetches` is an
      [`Operation`](framework.md#Operation), the *i*th return value
      will be `None`.
    * If the *i*th element of `fetches` is a
      [`Tensor`](framework.md#Tensor), the *i*th return value will
      be a numpy ndarray containing the value of that tensor.
    * If the *i*th element of `fetches` is a
      [`SparseTensor`](sparse_ops.md#SparseTensor), the *i*th
      return value will be a
      [`SparseTensorValue`](sparse_ops.md#SparseTensorValue)
      containing the value of that sparse tensor.
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

    The optional `feed_dict` argument allows the caller to override
    the value of tensors in the graph. Each key in `feed_dict` can be
    one of the following types:

<<<<<<< HEAD
    * If the key is a `tf.Tensor`, the
      value may be a Python scalar, string, list, or numpy ndarray
      that can be converted to the same `dtype` as that
      tensor. Additionally, if the key is a
      `tf.compat.v1.placeholder`, the shape of
      the value will be checked for compatibility with the placeholder.
    * If the key is a
      `tf.SparseTensor`,
      the value should be a
      `tf.compat.v1.SparseTensorValue`.
    * If the key is a nested tuple of `Tensor`s or `SparseTensor`s, the value
      should be a nested tuple with the same structure that maps to their
      corresponding values as above.

    Each value in `feed_dict` must be convertible to a numpy array of the dtype
    of the corresponding key.

    The optional `options` argument expects a [`RunOptions`] proto. The options
    allow controlling the behavior of this particular step (e.g. turning tracing
    on).

    The optional `run_metadata` argument expects a [`RunMetadata`] proto. When
    appropriate, the non-Tensor output of this step will be collected there. For
    example, when users turn on tracing in `options`, the profiled info will be
    collected into this argument and passed back.

    Args:
      fetches: A single graph element, a list of graph elements, or a dictionary
        whose values are graph elements or lists of graph elements (described
        above).
      feed_dict: A dictionary that maps graph elements to values (described
        above).
      options: A [`RunOptions`] protocol buffer
      run_metadata: A [`RunMetadata`] protocol buffer

    Returns:
      Either a single value if `fetches` is a single graph element, or
      a list of values if `fetches` is a list, or a dictionary with the
      same keys as `fetches` if that is a dictionary (described above).
      Order in which `fetches` operations are evaluated inside the call
      is undefined.
=======
    * If the key is a [`Tensor`](framework.md#Tensor), the
      value may be a Python scalar, string, list, or numpy ndarray
      that can be converted to the same `dtype` as that
      tensor. Additionally, if the key is a
      [placeholder](io_ops.md#placeholder), the shape of the value
      will be checked for compatibility with the placeholder.
    * If the key is a [`SparseTensor`](sparse_ops.md#SparseTensor),
      the value should be a
      [`SparseTensorValue`](sparse_ops.md#SparseTensorValue).

    Args:
      fetches: A single graph element, or a list of graph elements
        (described above).
      feed_dict: A dictionary that maps graph elements to values
        (described above).

    Returns:
      Either a single value if `fetches` is a single graph element, or
      a list of values if `fetches` is a list (described above).
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

    Raises:
      RuntimeError: If this `Session` is in an invalid state (e.g. has been
        closed).
      TypeError: If `fetches` or `feed_dict` keys are of an inappropriate type.
      ValueError: If `fetches` or `feed_dict` keys are invalid or refer to a
        `Tensor` that doesn't exist.
<<<<<<< HEAD
    """
    options_ptr = tf_session.TF_NewBufferFromString(
        compat.as_bytes(options.SerializeToString())) if options else None
    run_metadata_ptr = tf_session.TF_NewBuffer() if run_metadata else None

    try:
      result = self._run(None, fetches, feed_dict, options_ptr,
                         run_metadata_ptr)
      if run_metadata:
        proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)
        run_metadata.ParseFromString(compat.as_bytes(proto_data))
    finally:
      if run_metadata_ptr:
        tf_session.TF_DeleteBuffer(run_metadata_ptr)
      if options:
        tf_session.TF_DeleteBuffer(options_ptr)
    return result

  def partial_run(self, handle, fetches, feed_dict=None):
    """Continues the execution with more feeds and fetches.

    This is EXPERIMENTAL and subject to change.

    To use partial execution, a user first calls `partial_run_setup()` and
    then a sequence of `partial_run()`. `partial_run_setup` specifies the
    list of feeds and fetches that will be used in the subsequent
    `partial_run` calls.

    The optional `feed_dict` argument allows the caller to override
    the value of tensors in the graph. See run() for more information.

    Below is a simple example:

    ```python
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)
    r2 = math_ops.multiply(r1, c)

    h = sess.partial_run_setup([r1, r2], [a, b, c])
    res = sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
    res = sess.partial_run(h, r2, feed_dict={c: res})
    ```

    Args:
      handle: A handle for a sequence of partial runs.
      fetches: A single graph element, a list of graph elements, or a dictionary
        whose values are graph elements or lists of graph elements (see
        documentation for `run`).
      feed_dict: A dictionary that maps graph elements to values (described
        above).

    Returns:
      Either a single value if `fetches` is a single graph element, or
      a list of values if `fetches` is a list, or a dictionary with the
      same keys as `fetches` if that is a dictionary
      (see documentation for `run`).

    Raises:
      tf.errors.OpError: Or one of its subclasses on error.
    """
    # TODO(touts): Support feeding and fetching the same tensor.
    return self._run(handle, fetches, feed_dict, None, None)

  def partial_run_setup(self, fetches, feeds=None):
    """Sets up a graph with feeds and fetches for partial run.

    This is EXPERIMENTAL and subject to change.

    Note that contrary to `run`, `feeds` only specifies the graph elements.
    The tensors will be supplied by the subsequent `partial_run` calls.

    Args:
      fetches: A single graph element, or a list of graph elements.
      feeds: A single graph element, or a list of graph elements.

    Returns:
      A handle for partial run.

    Raises:
      RuntimeError: If this `Session` is in an invalid state (e.g. has been
        closed).
      TypeError: If `fetches` or `feed_dict` keys are of an inappropriate type.
      tf.errors.OpError: Or one of its subclasses if a TensorFlow error happens.
    """

    def _feed_fn(feed):
      for tensor_type, _, _, feed_fn in _REGISTERED_EXPANSIONS:
        if isinstance(feed, tensor_type):
          return feed_fn(feed)
      raise TypeError('Feed argument %r has invalid type %r' %
                      (feed, type(feed)))
=======

    """
    def _fetch_fn(fetch):
      for tensor_type, fetch_fn, _ in BaseSession._REGISTERED_EXPANSIONS:
        if isinstance(fetch, tensor_type):
          return fetch_fn(fetch)
      raise TypeError('Fetch argument %r has invalid type %r'
                      % (fetch, type(fetch)))

    def _feed_fn(feed, feed_val):
      for tensor_type, _, feed_fn in BaseSession._REGISTERED_EXPANSIONS:
        if isinstance(feed, tensor_type):
          return feed_fn(feed, feed_val)
      raise TypeError('Feed argument %r has invalid type %r'
                      % (feed, type(feed)))
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

    # Check session.
    if self._closed:
      raise RuntimeError('Attempted to use a closed Session.')
<<<<<<< HEAD
    if self.graph.version == 0:
      raise RuntimeError('The Session graph is empty.  Add operations to the '
                         'graph before calling run().')

    if feeds is None:
      feeds = []
    # Create request.
    feed_list = []

    # Validate and process feed_list.
    is_list_feed = isinstance(feeds, (list, tuple))
    if not is_list_feed:
      feeds = [feeds]
    for feed in feeds:
      for subfeed in _feed_fn(feed):
        try:
          subfeed_t = self.graph.as_graph_element(
              subfeed, allow_tensor=True, allow_operation=False)
          # pylint: disable=protected-access
          feed_list.append(subfeed_t._as_tf_output())
          # pylint: enable=protected-access
        except Exception as e:
          e.message = ('Cannot interpret feed_list key as Tensor: ' + e.message)
          e.args = (e.message,)
          raise e

    # Validate and process fetches.
    # TODO(touts): Support feeding and fetching the same tensor.
    fetch_handler = _FetchHandler(self._graph, fetches, {})

    # Set up a graph with feeds and fetches for partial run.
    def _setup_fn(session, feed_list, fetch_list, target_list):
      self._extend_graph()
      return tf_session.TF_SessionPRunSetup_wrapper(session, feed_list,
                                                    fetch_list, target_list)

    # pylint: disable=protected-access
    final_fetches = [t._as_tf_output() for t in fetch_handler.fetches()]
    final_targets = [op._c_op for op in fetch_handler.targets()]
    # pylint: enable=protected-access

    return self._do_call(_setup_fn, self._session, feed_list, final_fetches,
                         final_targets)

  def _run(self, handle, fetches, feed_dict, options, run_metadata):
    """Perform either run or partial_run, depending the presence of `handle`."""

    def _feed_fn(feed, feed_val):
      for tensor_type, _, feed_fn, _ in _REGISTERED_EXPANSIONS:
        if isinstance(feed, tensor_type):
          return feed_fn(feed, feed_val)
      raise TypeError('Feed argument %r has invalid type %r' %
                      (feed, type(feed)))

    # Check session.
    if self._closed:
      raise RuntimeError('Attempted to use a closed Session.')
    if self.graph.version == 0:
      raise RuntimeError('The Session graph is empty.  Add operations to the '
                         'graph before calling run().')

    # Create request.
    feed_dict_tensor = {}
    feed_map = {}

    # Validate and process feed_dict.
    feed_handles = {}
    if feed_dict:
      feed_dict = nest.flatten_dict_items(feed_dict)
      for feed, feed_val in feed_dict.items():
        for subfeed, subfeed_val in _feed_fn(feed, feed_val):
          try:
            subfeed_t = self.graph.as_graph_element(
                subfeed, allow_tensor=True, allow_operation=False)
          except Exception as e:
            raise TypeError('Cannot interpret feed_dict key as Tensor: ' +
                            e.args[0])

          if isinstance(subfeed_val, ops.Tensor):
            raise TypeError('The value of a feed cannot be a tf.Tensor object. '
                            'Acceptable feed values include Python scalars, '
                            'strings, lists, numpy ndarrays, or TensorHandles. '
                            'For reference, the tensor object was ' +
                            str(feed_val) + ' which was passed to the '
                            'feed with key ' + str(feed) + '.')

          subfeed_dtype = subfeed_t.dtype.as_numpy_dtype
          if isinstance(subfeed_val, int) and _convert_to_numpy_obj(
              subfeed_dtype, subfeed_val) != subfeed_val:
            raise TypeError(
                'Type of feed value ' + str(subfeed_val) + ' with type ' +
                str(type(subfeed_val)) +
                ' is not compatible with Tensor type ' + str(subfeed_dtype) +
                '. Try explicitly setting the type of the feed tensor'
                ' to a larger type (e.g. int64).')

          is_tensor_handle_feed = isinstance(subfeed_val,
                                             session_ops.TensorHandle)
          if is_tensor_handle_feed:
            np_val = subfeed_val.to_numpy_array()
            feed_handles[subfeed_t.experimental_ref()] = subfeed_val
          else:
            np_val = np.asarray(subfeed_val, dtype=subfeed_dtype)

          if (not is_tensor_handle_feed and
              not subfeed_t.get_shape().is_compatible_with(np_val.shape)):
            raise ValueError(
                'Cannot feed value of shape %r for Tensor %r, '
                'which has shape %r' %
                (np_val.shape, subfeed_t.name, str(subfeed_t.get_shape())))
          if not self.graph.is_feedable(subfeed_t):
            raise ValueError('Tensor %s may not be fed.' % subfeed_t)

          feed_dict_tensor[subfeed_t.experimental_ref()] = np_val
          feed_map[compat.as_bytes(subfeed_t.name)] = (subfeed_t, subfeed_val)

    # Create a fetch handler to take care of the structure of fetches.
    fetch_handler = _FetchHandler(
        self._graph, fetches, feed_dict_tensor, feed_handles=feed_handles)

    # Run request and get response.
    # We need to keep the returned movers alive for the following _do_run().
    # These movers are no longer needed when _do_run() completes, and
    # are deleted when `movers` goes out of scope when this _run() ends.
    # TODO(yuanbyu, keveman): Revisit whether we should just treat feeding
    # of a handle from a different device as an error.
    _ = self._update_with_movers(feed_dict_tensor, feed_map)
    final_fetches = fetch_handler.fetches()
    final_targets = fetch_handler.targets()
    # We only want to really perform the run if fetches or targets are provided,
    # or if the call is a partial run that specifies feeds.
    if final_fetches or final_targets or (handle and feed_dict_tensor):
      results = self._do_run(handle, final_targets, final_fetches,
                             feed_dict_tensor, options, run_metadata)
    else:
      results = []
    return fetch_handler.build_results(self, results)

  def make_callable(self, fetches, feed_list=None, accept_options=False):
    """Returns a Python callable that runs a particular step.

    The returned callable will take `len(feed_list)` arguments whose types
    must be compatible feed values for the respective elements of `feed_list`.
    For example, if element `i` of `feed_list` is a `tf.Tensor`, the `i`th
    argument to the returned callable must be a numpy ndarray (or something
    convertible to an ndarray) with matching element type and shape. See
    `tf.Session.run` for details of the allowable feed key and value types.

    The returned callable will have the same return type as
    `tf.Session.run(fetches, ...)`. For example, if `fetches` is a `tf.Tensor`,
    the callable will return a numpy ndarray; if `fetches` is a `tf.Operation`,
    it will return `None`.

    Args:
      fetches: A value or list of values to fetch. See `tf.Session.run` for
        details of the allowable fetch types.
      feed_list: (Optional.) A list of `feed_dict` keys. See `tf.Session.run`
        for details of the allowable feed key types.
      accept_options: (Optional.) If `True`, the returned `Callable` will be
        able to accept `tf.compat.v1.RunOptions` and `tf.compat.v1.RunMetadata`
        as optional keyword arguments `options` and `run_metadata`,
        respectively, with the same syntax and semantics as `tf.Session.run`,
        which is useful for certain use cases (profiling and debugging) but will
        result in measurable slowdown of the `Callable`'s
        performance. Default: `False`.

    Returns:
      A function that when called will execute the step defined by
      `feed_list` and `fetches` in this session.

    Raises:
      TypeError: If `fetches` or `feed_list` cannot be interpreted
        as arguments to `tf.Session.run`.
    """
    if feed_list is not None:
      if not isinstance(feed_list, (list, tuple)):
        raise TypeError('`feed_list` must be a list or tuple.')
      # Delegate any non-empty feed lists to the existing `run()` logic.
      # TODO(mrry): Refactor the feed handling logic from
      # `Session._run()` so that we can convert the feeds to a list of
      # strings here.
      def _generic_run(*feed_args, **kwargs):
        feed_dict = {
            feed: feed_val for feed, feed_val in zip(feed_list, feed_args)
        }
        return self.run(fetches, feed_dict=feed_dict, **kwargs)

      return _generic_run

    # Ensure any changes to the graph are reflected in the runtime.
    # Note that we don't need to do this on subsequent calls to the
    # returned object, because the arguments to `fetches` must already be
    # in the graph.
    self._extend_graph()

    # Create a fetch handler to take care of the structure of fetches.
    fetch_handler = _FetchHandler(self._graph, fetches, {})
    # pylint: disable=protected-access
    fetch_list = [t._as_tf_output() for t in fetch_handler.fetches()]
    target_list = [op._c_op for op in fetch_handler.targets()]

    # pylint: enable=protected-access

    def _callable_template_with_options_and_metadata(fetch_list,
                                                     target_list,
                                                     fetch_handler,
                                                     options=None,
                                                     run_metadata=None):
      """Template callable that accepts RunOptions and RunMetadata."""
      options_ptr = tf_session.TF_NewBufferFromString(
          compat.as_bytes(options.SerializeToString())) if options else None
      run_metadata_ptr = tf_session.TF_NewBuffer() if run_metadata else None
      try:
        results = self._call_tf_sessionrun(options_ptr, {}, fetch_list,
                                           target_list, run_metadata_ptr)
        if fetch_handler:
          results = fetch_handler.build_results(self, results)
        else:
          results = results[0] if results else None
        if run_metadata:
          proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)
          run_metadata.ParseFromString(compat.as_bytes(proto_data))
      finally:
        if run_metadata_ptr:
          tf_session.TF_DeleteBuffer(run_metadata_ptr)
        if options:
          tf_session.TF_DeleteBuffer(options_ptr)
      return results

    if accept_options:
      return functools.partial(_callable_template_with_options_and_metadata,
                               fetch_list, target_list, fetch_handler)
    elif isinstance(fetches, ops.Operation):
      # Special case for fetching a single operation, because the
      # function will have no return value.
      assert not fetch_list
      assert len(target_list) == 1

      def _single_operation_run():
        self._call_tf_sessionrun(None, {}, [], target_list, None)

      return _single_operation_run
    elif isinstance(fetches, ops.Tensor):
      # Special case for fetching a single tensor, because the
      # function can return the result of `TF_Run()` directly.
      assert len(fetch_list) == 1
      assert not target_list

      def _single_tensor_run():
        results = self._call_tf_sessionrun(None, {}, fetch_list, [], None)
        return results[0]

      return _single_tensor_run
    else:
      # In all other cases, we must use `fetch_handler` to build the
      # results for us.
      def _fetch_handler_run():
        results = self._call_tf_sessionrun(None, {}, fetch_list, target_list,
                                           None)
        return fetch_handler.build_results(self, results)

      return _fetch_handler_run

  # Captures the name of a node in an error status. The regex below matches
  # both the old and the new formats:
  # Old format: [[Node: <node_name> = ...]]
  # New format: [[{{node <node_name>}} = ...]]
  _NODEDEF_NAME_RE = re.compile(
      r'\[\[(Node: )?(\{\{node )?([^\} ]*)(\}\})?\s*=*')

  def _do_run(self, handle, target_list, fetch_list, feed_dict, options,
              run_metadata):
    """Runs a step based on the given fetches and feeds.

    Args:
      handle: a handle for partial_run. None if this is just a call to run().
      target_list: A list of operations to be run, but not fetched.
      fetch_list: A list of tensors to be fetched.
      feed_dict: A dictionary that maps tensors to numpy ndarrays.
      options: A (pointer to a) [`RunOptions`] protocol buffer, or None
      run_metadata: A (pointer to a) [`RunMetadata`] protocol buffer, or None
=======

    # Validate and process fetches.
    is_list_fetch = isinstance(fetches, (list, tuple))
    if not is_list_fetch:
      fetches = [fetches]

    unique_fetch_targets = set()
    target_list = []

    fetch_info = []
    for fetch in fetches:
      subfetches, fetch_contraction_fn = _fetch_fn(fetch)
      subfetch_names = []
      for subfetch in subfetches:
        try:
          fetch_t = self.graph.as_graph_element(subfetch, allow_tensor=True,
                                                allow_operation=True)
          if isinstance(fetch_t, ops.Operation):
            target_list.append(fetch_t.name)
          else:
            subfetch_names.append(fetch_t.name)
        except TypeError as e:
          raise TypeError('Fetch argument %r of %r has invalid type %r, '
                          'must be a string or Tensor. (%s)'
                          % (subfetch, fetch, type(subfetch), e.message))
        except ValueError as e:
          raise ValueError('Fetch argument %r of %r cannot be interpreted as a '
                           'Tensor. (%s)' % (subfetch, fetch, e.message))
        except KeyError as e:
          raise ValueError('Fetch argument %r of %r cannot be interpreted as a '
                           'Tensor. (%s)' % (subfetch, fetch, e.message))
      unique_fetch_targets.update(subfetch_names)
      fetch_info.append((subfetch_names, fetch_contraction_fn))

    unique_fetch_targets = list(unique_fetch_targets)

    # Create request.
    feed_dict_string = {}

    # Validate and process feed_dict.
    if feed_dict:
      for feed, feed_val in feed_dict.iteritems():
        for subfeed, subfeed_val in _feed_fn(feed, feed_val):
          try:
            subfeed_t = self.graph.as_graph_element(subfeed, allow_tensor=True,
                                                    allow_operation=False)
          except Exception as e:
            e.message = ('Cannot interpret feed_dict key as Tensor: '
                         + e.message)
            e.args = (e.message,)
            raise e
          np_val = np.array(subfeed_val, dtype=subfeed_t.dtype.as_numpy_dtype)
          if subfeed_t.op.type == 'Placeholder':
            if not subfeed_t.get_shape().is_compatible_with(np_val.shape):
              raise ValueError(
                  'Cannot feed value of shape %r for Tensor %r, '
                  'which has shape %r'
                  % (np_val.shape, subfeed_t.name,
                     tuple(subfeed_t.get_shape().dims)))
          feed_dict_string[str(subfeed_t.name)] = np_val

    # Run request and get response.
    results = self._do_run(target_list, unique_fetch_targets, feed_dict_string)

    # User may have fetched the same tensor multiple times, but we
    # only fetch them from the runtime once.  Furthermore, they may
    # be wrapped as a tuple of tensors.  Here we map the results back
    # to what the client asked for.
    fetched_results = dict(zip(unique_fetch_targets, results))
    ret = []
    for fetch_names, fetch_contraction_fn in fetch_info:
      if fetch_names:
        fetched_vals = [fetched_results[name] for name in fetch_names]
        ret.append(fetch_contraction_fn(fetched_vals))
      else:
        ret.append(None)

    if is_list_fetch:
      return ret
    else:
      return ret[0]

  # Captures the name of a node in an error status.
  _NODEDEF_NAME_RE = re.compile(r'\[\[Node: ([^ ]*?) =')

  def _do_run(self, target_list, fetch_list, feed_dict):
    """Runs a step based on the given fetches and feeds.

    Args:
      target_list: A list of strings corresponding to names of tensors
        or operations to be run to, but not fetched.
      fetch_list: A list of strings corresponding to names of tensors to be
        fetched and operations to be run.
      feed_dict: A dictionary that maps tensor names to numpy ndarrays.
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

    Returns:
      A list of numpy ndarrays, corresponding to the elements of
      `fetch_list`.  If the ith element of `fetch_list` contains the
      name of an operation, the first Tensor output of that operation
      will be returned for that element.
<<<<<<< HEAD

    Raises:
      tf.errors.OpError: Or one of its subclasses on error.
    """
    # pylint: disable=protected-access
    feeds = dict((t.deref()._as_tf_output(), v) for t, v in feed_dict.items())
    fetches = [t._as_tf_output() for t in fetch_list]
    targets = [op._c_op for op in target_list]

    # pylint: enable=protected-access

    def _run_fn(feed_dict, fetch_list, target_list, options, run_metadata):
      # Ensure any changes to the graph are reflected in the runtime.
      self._extend_graph()
      return self._call_tf_sessionrun(options, feed_dict, fetch_list,
                                      target_list, run_metadata)

    def _prun_fn(handle, feed_dict, fetch_list):
      if target_list:
        raise RuntimeError('partial_run() requires empty target_list.')
      return self._call_tf_sessionprun(handle, feed_dict, fetch_list)

    if handle is None:
      return self._do_call(_run_fn, feeds, fetches, targets, options,
                           run_metadata)
    else:
      return self._do_call(_prun_fn, handle, feeds, fetches)

  def _do_call(self, fn, *args):
    try:
      return fn(*args)
    except errors.OpError as e:
      message = compat.as_text(e.message)
      m = BaseSession._NODEDEF_NAME_RE.search(message)
      node_def = None
      op = None
      if m is not None:
        node_name = m.group(3)
=======
    """
    try:
      # Ensure any changes to the graph are reflected in the runtime.
      with self._extend_lock:
        if self._graph.version > self._current_version:
          graph_def = self._graph.as_graph_def(
              from_version=self._current_version)

          try:
            status = tf_session.TF_NewStatus()
            tf_session.TF_ExtendGraph(
                self._session, graph_def.SerializeToString(), status)
            if tf_session.TF_GetCode(status) != 0:
              raise RuntimeError(tf_session.TF_Message(status))
            self._opened = True
          finally:
            tf_session.TF_DeleteStatus(status)

          self._current_version = self._graph.version

      return tf_session.TF_Run(self._session, feed_dict, fetch_list,
                               target_list)

    except tf_session.StatusNotOK as e:
      e_type, e_value, e_traceback = sys.exc_info()
      m = BaseSession._NODEDEF_NAME_RE.search(e.error_message)
      if m is not None:
        node_name = m.group(1)
        node_def = None
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
        try:
          op = self._graph.get_operation_by_name(node_name)
          node_def = op.node_def
        except KeyError:
<<<<<<< HEAD
          pass
      message = error_interpolation.interpolate(message, self._graph)
      if 'only supports NHWC tensor format' in message:
        message += ('\nA possible workaround: Try disabling Grappler optimizer'
                    '\nby modifying the config for creating the session eg.'
                    '\nsession_config.graph_options.rewrite_options.'
                    'disable_meta_optimizer = True')
      raise type(e)(node_def, op, message)

  def _extend_graph(self):
    with self._graph._session_run_lock():  # pylint: disable=protected-access
      tf_session.ExtendSession(self._session)

  # The threshold to run garbage collection to delete dead tensors.
  _DEAD_HANDLES_THRESHOLD = 10

  def _register_dead_handle(self, handle):
    # Register a dead handle in the session. Delete the dead tensors when
    # the number of dead tensors exceeds certain threshold.
    tensors_to_delete = None
    with self._delete_lock:
      self._dead_handles.append(handle)
      if len(self._dead_handles) == BaseSession._DEAD_HANDLES_THRESHOLD:
        tensors_to_delete = self._dead_handles
        self._dead_handles = []
    # Delete the dead tensors.
    if tensors_to_delete:
      feeds = {}
      fetches = []
      for deleter_key, tensor_handle in enumerate(tensors_to_delete):
        holder, deleter = session_ops._get_handle_deleter(
            self.graph, deleter_key, tensor_handle)
        feeds[holder] = tensor_handle
        fetches.append(deleter)
      self.run(fetches, feed_dict=feeds)

  def _update_with_movers(self, feed_dict, feed_map):
    # If a tensor handle that is fed to a device incompatible placeholder,
    # we move the tensor to the right device, generate a new tensor handle,
    # and update `feed_dict` to use the new handle.
    handle_movers = []
    for feed_name, val in feed_map.items():
      mover = session_ops._get_handle_mover(self.graph, *val)
      if mover:
        handle_movers.append((feed_name, val[1], mover))
    # Transfer a tensor to the right device if needed.
    if not handle_movers:
      return []
    else:
      feeds = {}
      fetches = []
      for _, handle, mover in handle_movers:
        feeds[mover[0]] = handle
        fetches.append(mover[1])
      handles = self.run(fetches, feed_dict=feeds)
      for handle_mover, handle in zip(handle_movers, handles):
        np_val = np.array(handle.handle, dtype=np.object)
        feed_name = handle_mover[0]
        feed_tensor = feed_map[feed_name][0]
        feed_dict[feed_tensor.experimental_ref()] = np_val
      return handles

  def _call_tf_sessionrun(self, options, feed_dict, fetch_list, target_list,
                          run_metadata):
    return tf_session.TF_SessionRun_wrapper(self._session, options, feed_dict,
                                            fetch_list, target_list,
                                            run_metadata)

  def _call_tf_sessionprun(self, handle, feed_dict, fetch_list):
    return tf_session.TF_SessionPRun_wrapper(self._session, handle, feed_dict,
                                             fetch_list)

  # pylint: disable=protected-access
  class _Callable(object):
    """Experimental wrapper for the C++ `Session::MakeCallable()` API."""

    def __init__(self, session, callable_options):
      self._session = session
      self._handle = None
      options_ptr = tf_session.TF_NewBufferFromString(
          compat.as_bytes(callable_options.SerializeToString()))
      try:
        self._handle = tf_session.TF_SessionMakeCallable(
            session._session, options_ptr)
      finally:
        tf_session.TF_DeleteBuffer(options_ptr)

    def __call__(self, *args, **kwargs):
      # TODO(b/74355905): Support argument and return value nested structures,
      # and tensor-like objects such as SparseTensors.
      run_metadata = kwargs.get('run_metadata', None)
      try:
        run_metadata_ptr = tf_session.TF_NewBuffer() if run_metadata else None
        ret = tf_session.TF_SessionRunCallable(self._session._session,
                                               self._handle, args,
                                               run_metadata_ptr)
        if run_metadata:
          proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)
          run_metadata.ParseFromString(compat.as_bytes(proto_data))
      finally:
        if run_metadata_ptr:
          tf_session.TF_DeleteBuffer(run_metadata_ptr)
      return ret

    def __del__(self):
      # NOTE(mrry): It is possible that `self._session.__del__()` could be
      # called before this destructor, in which case `self._session._session`
      # will be `None`.
      if (self._handle is not None and self._session._session is not None and
          not self._session._closed):
        tf_session.TF_SessionReleaseCallable(self._session._session,
                                             self._handle)

  # pylint: enable=protected-access

  # TODO(b/74355905): Reimplement `Session.make_callable()` using this method
  # where possible.
  def _make_callable_from_options(self, callable_options):
    """Returns a handle to a "callable" with the given options.

    Args:
      callable_options: A `CallableOptions` protocol buffer message describing
        the computation that will be performed by the callable.

    Returns:
      A handle to the new callable.
    """
    self._extend_graph()
    return BaseSession._Callable(self, callable_options)


@tf_export(v1=['Session'])
=======
          op = None
        # pylint: disable=protected-access
        raise errors._make_specific_exception(node_def, op, e.error_message,
                                              e.code)
        # pylint: enable=protected-access
      raise e_type, e_value, e_traceback


>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
class Session(BaseSession):
  """A class for running TensorFlow operations.

  A `Session` object encapsulates the environment in which `Operation`
  objects are executed, and `Tensor` objects are evaluated. For
  example:

  ```python
  # Build a graph.
  a = tf.constant(5.0)
  b = tf.constant(6.0)
  c = a * b

  # Launch the graph in a session.
<<<<<<< HEAD
  sess = tf.compat.v1.Session()

  # Evaluate the tensor `c`.
  print(sess.run(c))
  ```

  A session may own resources, such as
  `tf.Variable`, `tf.queue.QueueBase`,
  and `tf.compat.v1.ReaderBase`. It is important to release
  these resources when they are no longer required. To do this, either
  invoke the `tf.Session.close` method on the session, or use
=======
  sess = tf.Session()

  # Evaluate the tensor `c`.
  print sess.run(c)
  ```

  A session may own resources, such as
  [variables](state_ops.md#Variable), [queues](io_ops.md#QueueBase),
  and [readers](io_ops.md#ReaderBase). It is important to release
  these resources when they are no longer required. To do this, either
  invoke the [`close()`](#Session.close) method on the session, or use
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  the session as a context manager. The following two examples are
  equivalent:

  ```python
  # Using the `close()` method.
<<<<<<< HEAD
  sess = tf.compat.v1.Session()
=======
  sess = tf.Session()
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  sess.run(...)
  sess.close()

  # Using the context manager.
<<<<<<< HEAD
  with tf.compat.v1.Session() as sess:
    sess.run(...)
  ```

  The
  [`ConfigProto`](https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto)
=======
  with tf.Session() as sess:
    sess.run(...)
  ```

  The [`ConfigProto`]
  (https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/core/framework/config.proto)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  protocol buffer exposes various configuration options for a
  session. For example, to create a session that uses soft constraints
  for device placement, and log the resulting placement decisions,
  create a session as follows:

  ```python
  # Launch the graph in a session that allows soft device placement and
  # logs the placement decisions.
<<<<<<< HEAD
  sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=True))
  ```
=======
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=True))
  ```

  @@__init__
  @@run
  @@close

  @@graph

  @@as_default

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  """

  def __init__(self, target='', graph=None, config=None):
    """Creates a new TensorFlow session.

    If no `graph` argument is specified when constructing the session,
    the default graph will be launched in the session. If you are
<<<<<<< HEAD
    using more than one graph (created with `tf.Graph()`) in the same
=======
    using more than one graph (created with `tf.Graph()` in the same
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    process, you will have to use different sessions for each graph,
    but each graph can be used in multiple sessions. In this case, it
    is often clearer to pass the graph to be launched explicitly to
    the session constructor.

    Args:
<<<<<<< HEAD
      target: (Optional.) The execution engine to connect to. Defaults to using
        an in-process engine. See
        [Distributed TensorFlow](https://tensorflow.org/deploy/distributed) for
          more examples.
      graph: (Optional.) The `Graph` to be launched (described above).
      config: (Optional.) A
        [`ConfigProto`](https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto)
          protocol buffer with configuration options for the session.
    """
    super(Session, self).__init__(target, graph, config=config)
    # NOTE(mrry): Create these on first `__enter__` to avoid a reference cycle.
    self._default_graph_context_manager = None
    self._default_session_context_manager = None

  def __enter__(self):
    if self._default_graph_context_manager is None:
      self._default_graph_context_manager = self.graph.as_default()
    else:
      raise RuntimeError('Session context managers are not re-entrant. '
                         'Use `Session.as_default()` if you want to enter '
                         'a session multiple times.')
    if self._default_session_context_manager is None:
      self._default_session_context_manager = self.as_default()
    self._default_graph_context_manager.__enter__()
    return self._default_session_context_manager.__enter__()
=======
      target: (Optional.) The execution engine to connect to.
        Defaults to using an in-process engine. At present, no value
        other than the empty string is supported.
      graph: (Optional.) The `Graph` to be launched (described above).
      config: (Optional.) A [`ConfigProto`](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/core/framework/config.proto)
        protocol buffer with configuration options for the session.

    """
    super(Session, self).__init__(target, graph, config=config)
    self._context_managers = [self.graph.as_default(), self.as_default()]

  def __enter__(self):
    for context_manager in self._context_managers:
      context_manager.__enter__()
    return self
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  def __exit__(self, exec_type, exec_value, exec_tb):
    if exec_type is errors.OpError:
      logging.error('Session closing due to OpError: %s', (exec_value,))
<<<<<<< HEAD
    try:
      self._default_session_context_manager.__exit__(exec_type, exec_value,
                                                     exec_tb)
    except RuntimeError as error:
      if error == exec_value:
        # NOTE(skyewm): for some reason, in Python3,
        # _default_session_context_manager.__exit__ will re-raise the "not
        # re-entrant" exception raised in __enter__ above (note that if we're
        # here, we're in the outer session context manager, since __exit__ is
        # not called when __enter__ raises an exception). We still want to
        # continue cleaning up this context manager before the exception is
        # further propagated, so we ignore it here (note that it'll continue
        # being propagated after this method completes).
        pass
      else:
        raise
    self._default_graph_context_manager.__exit__(exec_type, exec_value, exec_tb)

    self._default_session_context_manager = None
    self._default_graph_context_manager = None

    # If we are closing due to an exception, set a time limit on our Close() to
    # avoid blocking forever.
    # TODO(b/120204635) remove this when deadlock is fixed.
    if exec_type:
      close_thread = threading.Thread(
          name='SessionCloseThread', target=self.close)
      close_thread.daemon = True
      close_thread.start()
      close_thread.join(30.0)
      if close_thread.is_alive():
        logging.error(
            'Session failed to close after 30 seconds. Continuing after this '
            'point may leave your program in an undefined state.')
    else:
      self.close()

  @staticmethod
  def reset(target, containers=None, config=None):
    """Resets resource containers on `target`, and close all connected sessions.

    A resource container is distributed across all workers in the
    same cluster as `target`.  When a resource container on `target`
    is reset, resources associated with that container will be cleared.
    In particular, all Variables in the container will become undefined:
    they lose their values and shapes.

    NOTE:
    (i) reset() is currently only implemented for distributed sessions.
    (ii) Any sessions on the master named by `target` will be closed.

    If no resource containers are provided, all containers are reset.

    Args:
      target: The execution engine to connect to.
      containers: A list of resource container name strings, or `None` if all of
        all the containers are to be reset.
      config: (Optional.) Protocol buffer with configuration options.

    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        resetting containers.
    """
    if target is not None:
      target = compat.as_bytes(target)
    if containers is not None:
      containers = [compat.as_bytes(c) for c in containers]
    else:
      containers = []
    tf_session.TF_Reset(target, containers, config)


@tf_export(v1=['InteractiveSession'])
class InteractiveSession(BaseSession):
  """A TensorFlow `Session` for use in interactive contexts, such as a shell.

  The only difference with a regular `Session` is that an `InteractiveSession`
  installs itself as the default session on construction.
  The methods `tf.Tensor.eval`
  and `tf.Operation.run`
  will use that session to run ops.

  This is convenient in interactive shells and [IPython
  notebooks](http://ipython.org), as it avoids having to pass an explicit
  `Session` object to run ops.

  For example:

  ```python
  sess = tf.compat.v1.InteractiveSession()
  a = tf.constant(5.0)
  b = tf.constant(6.0)
  c = a * b
  # We can just use 'c.eval()' without passing 'sess'
  print(c.eval())
  sess.close()
  ```

  Note that a regular session installs itself as the default session when it
  is created in a `with` statement.  The common usage in non-interactive
  programs is to follow that pattern:

  ```python
  a = tf.constant(5.0)
  b = tf.constant(6.0)
  c = a * b
  with tf.compat.v1.Session():
    # We can also use 'c.eval()' here.
    print(c.eval())
  ```
  """

  _count_lock = threading.Lock()
  _active_session_count = 0  # GUARDED_BY(_count_lock)

  def __init__(self, target='', graph=None, config=None):
    """Creates a new interactive TensorFlow session.

    If no `graph` argument is specified when constructing the session,
    the default graph will be launched in the session. If you are
    using more than one graph (created with `tf.Graph()`) in the same
    process, you will have to use different sessions for each graph,
    but each graph can be used in multiple sessions. In this case, it
    is often clearer to pass the graph to be launched explicitly to
    the session constructor.

    Args:
      target: (Optional.) The execution engine to connect to. Defaults to using
        an in-process engine.
      graph: (Optional.) The `Graph` to be launched (described above).
      config: (Optional) `ConfigProto` proto used to configure the session.
    """
    if not config:
      # If config is not provided, choose some reasonable defaults for
      # interactive use:
      #
      #   - Grow GPU memory as needed at the cost of fragmentation.
      gpu_options = config_pb2.GPUOptions(allow_growth=True)
      config = config_pb2.ConfigProto(gpu_options=gpu_options)
    # Interactive sessions always place pruned graphs.
    config.graph_options.place_pruned_graph = True

    super(InteractiveSession, self).__init__(target, graph, config)
    with InteractiveSession._count_lock:
      if InteractiveSession._active_session_count > 0:
        warnings.warn('An interactive session is already active. This can '
                      'cause out-of-memory errors in some cases. You must '
                      'explicitly call `InteractiveSession.close()` to release '
                      'resources held by the other session(s).')
      InteractiveSession._active_session_count += 1
    # NOTE(mrry): We do not use `Session._closed` here because it has unhelpful
    # semantics (in particular, it is not set to true if `Session.close()` is
    # called on a session that has not been "opened" by running a step) and we
    # cannot change those semantics without breaking existing code.
    self._explicitly_closed = False

    self._default_session = self.as_default()
    self._default_session.enforce_nesting = False
=======

    for context_manager in reversed(self._context_managers):
      context_manager.__exit__(exec_type, exec_value, exec_tb)

    self.close()


class InteractiveSession(BaseSession):
  """A TensorFlow `Session` for use in interactive contexts, such as a shell.

  In some cases, such as interactive shells and IPython notebooks, it is
  useful to be able to define a `Session` without using a with block: this
  style enables statements to be executed immediately, rather than at the
  termination of the block. In that case, it must be closed using
  `Session.close()`. For example:

  ```python
  sess = InteractiveSession()
  a = tf.constant(5.0)
  b = tf.constant(6.0)
  c = a * b
  print c.run()
  sess.close()
  ```

  @@__init__
  @@close
  """

  def __init__(self, target='', graph=None):
    """Initializes an `InteractiveSession` object similar to `Session`.

    Args:
      target: Optional. The TensorFlow execution engine to connect to.
      graph: Optional. The `Graph` object to be used. If this argument is None,
        the default graph will be used.
    """
    super(InteractiveSession, self).__init__(target, graph)
    self._default_session = self.as_default()
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    self._default_session.__enter__()
    self._explicit_graph = graph
    if self._explicit_graph is not None:
      self._default_graph = graph.as_default()
<<<<<<< HEAD
      self._default_graph.enforce_nesting = False
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      self._default_graph.__enter__()

  def close(self):
    """Closes an `InteractiveSession`."""
    super(InteractiveSession, self).close()
<<<<<<< HEAD
    with InteractiveSession._count_lock:
      if not self._explicitly_closed:
        InteractiveSession._active_session_count -= 1
        self._explicitly_closed = True
      else:
        return
    if self._explicit_graph is not None:
      self._default_graph.__exit__(None, None, None)
      self._default_graph = None
    self._default_session.__exit__(None, None, None)
    self._default_session = None
=======
    if self._explicit_graph is not None:
      self._default_graph.__exit__(None, None, None)
    self._default_session.__exit__(None, None, None)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
