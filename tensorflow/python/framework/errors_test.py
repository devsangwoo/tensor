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
"""Tests for tensorflow.python.framework.errors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import pickle
import warnings

from tensorflow.core.lib.core import error_codes_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class ErrorsTest(test.TestCase):

  def _CountReferences(self, typeof):
    """Count number of references to objects of type |typeof|."""
    objs = gc.get_objects()
    ref_count = 0
    for o in objs:
      try:
        if isinstance(o, typeof):
          ref_count += 1
      # Certain versions of python keeps a weakref to deleted objects.
      except ReferenceError:
        pass
    return ref_count

  def testUniqueClassForEachErrorCode(self):
    for error_code, exc_type in [
        (errors.CANCELLED, errors_impl.CancelledError),
        (errors.UNKNOWN, errors_impl.UnknownError),
        (errors.INVALID_ARGUMENT, errors_impl.InvalidArgumentError),
        (errors.DEADLINE_EXCEEDED, errors_impl.DeadlineExceededError),
        (errors.NOT_FOUND, errors_impl.NotFoundError),
        (errors.ALREADY_EXISTS, errors_impl.AlreadyExistsError),
        (errors.PERMISSION_DENIED, errors_impl.PermissionDeniedError),
        (errors.UNAUTHENTICATED, errors_impl.UnauthenticatedError),
        (errors.RESOURCE_EXHAUSTED, errors_impl.ResourceExhaustedError),
        (errors.FAILED_PRECONDITION, errors_impl.FailedPreconditionError),
        (errors.ABORTED, errors_impl.AbortedError),
        (errors.OUT_OF_RANGE, errors_impl.OutOfRangeError),
        (errors.UNIMPLEMENTED, errors_impl.UnimplementedError),
        (errors.INTERNAL, errors_impl.InternalError),
        (errors.UNAVAILABLE, errors_impl.UnavailableError),
        (errors.DATA_LOSS, errors_impl.DataLossError),
    ]:
      # pylint: disable=protected-access
      self.assertTrue(
          isinstance(
              errors_impl._make_specific_exception(None, None, None,
                                                   error_code), exc_type))
      # error_code_from_exception_type and exception_type_from_error_code should
      # be consistent with operation result.
      self.assertEqual(error_code,
                       errors_impl.error_code_from_exception_type(exc_type))
=======
"""Tests for tensorflow.python.framework.errors."""
import tensorflow.python.platform

import warnings

import tensorflow as tf

from tensorflow.core.lib.core import error_codes_pb2

class ErrorsTest(tf.test.TestCase):

  def testUniqueClassForEachErrorCode(self):
    for error_code, exc_type in [
        (tf.errors.CANCELLED, tf.errors.CancelledError),
        (tf.errors.UNKNOWN, tf.errors.UnknownError),
        (tf.errors.INVALID_ARGUMENT, tf.errors.InvalidArgumentError),
        (tf.errors.DEADLINE_EXCEEDED, tf.errors.DeadlineExceededError),
        (tf.errors.NOT_FOUND, tf.errors.NotFoundError),
        (tf.errors.ALREADY_EXISTS, tf.errors.AlreadyExistsError),
        (tf.errors.PERMISSION_DENIED, tf.errors.PermissionDeniedError),
        (tf.errors.UNAUTHENTICATED, tf.errors.UnauthenticatedError),
        (tf.errors.RESOURCE_EXHAUSTED, tf.errors.ResourceExhaustedError),
        (tf.errors.FAILED_PRECONDITION, tf.errors.FailedPreconditionError),
        (tf.errors.ABORTED, tf.errors.AbortedError),
        (tf.errors.OUT_OF_RANGE, tf.errors.OutOfRangeError),
        (tf.errors.UNIMPLEMENTED, tf.errors.UnimplementedError),
        (tf.errors.INTERNAL, tf.errors.InternalError),
        (tf.errors.UNAVAILABLE, tf.errors.UnavailableError),
        (tf.errors.DATA_LOSS, tf.errors.DataLossError),
        ]:
      # pylint: disable=protected-access
      self.assertTrue(isinstance(
          tf.errors._make_specific_exception(None, None, None, error_code),
          exc_type))
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      # pylint: enable=protected-access

  def testKnownErrorClassForEachErrorCodeInProto(self):
    for error_code in error_codes_pb2.Code.values():
      # pylint: disable=line-too-long
<<<<<<< HEAD
      if error_code in (
          error_codes_pb2.OK, error_codes_pb2.
          DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_
      ):
=======
      if error_code in (error_codes_pb2.OK,
                        error_codes_pb2.DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_):
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
        continue
      # pylint: enable=line-too-long
      with warnings.catch_warnings(record=True) as w:
        # pylint: disable=protected-access
<<<<<<< HEAD
        exc = errors_impl._make_specific_exception(None, None, None, error_code)
        # pylint: enable=protected-access
      self.assertEqual(0, len(w))  # No warning is raised.
      self.assertTrue(isinstance(exc, errors_impl.OpError))
      self.assertTrue(errors_impl.OpError in exc.__class__.__bases__)
=======
        exc = tf.errors._make_specific_exception(None, None, None, error_code)
        # pylint: enable=protected-access
      self.assertEqual(0, len(w))  # No warning is raised.
      self.assertTrue(isinstance(exc, tf.errors.OpError))
      self.assertTrue(tf.errors.OpError in exc.__class__.__bases__)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  def testUnknownErrorCodeCausesWarning(self):
    with warnings.catch_warnings(record=True) as w:
      # pylint: disable=protected-access
<<<<<<< HEAD
      exc = errors_impl._make_specific_exception(None, None, None, 37)
      # pylint: enable=protected-access
    self.assertEqual(1, len(w))
    self.assertTrue("Unknown error code: 37" in str(w[0].message))
    self.assertTrue(isinstance(exc, errors_impl.OpError))

    with warnings.catch_warnings(record=True) as w:
      # pylint: disable=protected-access
      exc = errors_impl.error_code_from_exception_type("Unknown")
      # pylint: enable=protected-access
    self.assertEqual(1, len(w))
    self.assertTrue("Unknown class exception" in str(w[0].message))
    self.assertTrue(isinstance(exc, errors_impl.OpError))

  def testStatusDoesNotLeak(self):
    try:
      pywrap_tensorflow.DeleteFile(compat.as_bytes("/DOES_NOT_EXIST/"))
    except:
      pass
    gc.collect()
    self.assertEqual(0, self._CountReferences(c_api_util.ScopedTFStatus))

  def testPickleable(self):
    for error_code in [
        errors.CANCELLED,
        errors.UNKNOWN,
        errors.INVALID_ARGUMENT,
        errors.DEADLINE_EXCEEDED,
        errors.NOT_FOUND,
        errors.ALREADY_EXISTS,
        errors.PERMISSION_DENIED,
        errors.UNAUTHENTICATED,
        errors.RESOURCE_EXHAUSTED,
        errors.FAILED_PRECONDITION,
        errors.ABORTED,
        errors.OUT_OF_RANGE,
        errors.UNIMPLEMENTED,
        errors.INTERNAL,
        errors.UNAVAILABLE,
        errors.DATA_LOSS,
    ]:
      # pylint: disable=protected-access
      exc = errors_impl._make_specific_exception(None, None, None, error_code)
      # pylint: enable=protected-access
      unpickled = pickle.loads(pickle.dumps(exc))
      self.assertEqual(exc.node_def, unpickled.node_def)
      self.assertEqual(exc.op, unpickled.op)
      self.assertEqual(exc.message, unpickled.message)
      self.assertEqual(exc.error_code, unpickled.error_code)


if __name__ == "__main__":
  test.main()
=======
      exc = tf.errors._make_specific_exception(None, None, None, 37)
      # pylint: enable=protected-access
    self.assertEqual(1, len(w))
    self.assertTrue("Unknown error code: 37" in str(w[0].message))
    self.assertTrue(isinstance(exc, tf.errors.OpError))


if __name__ == "__main__":
  tf.test.main()
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
