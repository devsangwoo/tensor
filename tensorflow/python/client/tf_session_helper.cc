<<<<<<< HEAD
/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#include "tensorflow/python/client/tf_session_helper.h"

#include <cstring>

<<<<<<< HEAD
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/equal_graph_def.h"
#include "tensorflow/python/client/session_ref.h"
#include "tensorflow/python/lib/core/ndarray_tensor.h"
#include "tensorflow/python/lib/core/ndarray_tensor_bridge.h"
#include "tensorflow/python/lib/core/safe_ptr.h"
=======
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/port.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

namespace tensorflow {

namespace {

<<<<<<< HEAD
static const char* kFeedDictErrorMsg =
    "feed_dict must be a dictionary mapping strings to NumPy arrays.";
}  // end namespace

TF_Session* TF_NewSessionRef(TF_Graph* graph, const TF_SessionOptions* opts,
                             TF_Status* status) {
  TF_Session* tf_session = TF_NewSession(graph, opts, status);
  if (tf_session == nullptr) {
    return nullptr;
  }

  Session* session = reinterpret_cast<Session*>(tf_session->session);
  SessionRef* session_ref = new SessionRef(session);
  tf_session->session = session_ref;
  return tf_session;
}

void TF_Run_wrapper_helper(TF_DeprecatedSession* session, const char* handle,
                           const TF_Buffer* run_options, PyObject* feed_dict,
                           const NameVector& output_names,
                           const NameVector& target_nodes,
                           TF_Status* out_status, PyObjectVector* out_values,
                           TF_Buffer* run_outputs) {
  // 1. Convert the feed inputs to the appropriate form for TF_Run.
  if (!PyDict_Check(feed_dict)) {
    Set_TF_Status_from_Status(out_status,
                              errors::InvalidArgument(kFeedDictErrorMsg));
    return;
  }

  NameVector input_names;
  std::vector<Safe_TF_TensorPtr> inputs_safe;  // Used to delete tensors.
  TF_TensorVector inputs_unsafe;  // Used to contain the arg to TF_Run.

  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;
  int index = 0;
  Status s;

  while (PyDict_Next(feed_dict, &pos, &key, &value)) {
    char* key_string = PyBytes_AsString(key);
    if (!key_string) {
      Set_TF_Status_from_Status(out_status,
                                errors::InvalidArgument(kFeedDictErrorMsg));
      return;
    }
    input_names.push_back(key_string);

    inputs_safe.emplace_back(make_safe(static_cast<TF_Tensor*>(nullptr)));
    s = PyArrayToTF_Tensor(value, &inputs_safe.back());
    if (!s.ok()) {
      Set_TF_Status_from_Status(out_status, s);
      return;
    }
    inputs_unsafe.push_back(inputs_safe.back().get());
    ++index;
  }

  // 2. Allocate a container for the output data.
  TF_TensorVector outputs(output_names.size());

  // In case any tensors were leftover from previous runs we might as well clear
  // them here.
  ClearDecrefCache();

  // 3. Actually call TF_Run().
  Py_BEGIN_ALLOW_THREADS;
  if (handle == nullptr) {
    TF_Run(session, run_options, input_names.data(), inputs_unsafe.data(),
           input_names.size(), const_cast<const char**>(output_names.data()),
           outputs.data(), output_names.size(),
           const_cast<const char**>(target_nodes.data()), target_nodes.size(),
           run_outputs, out_status);
  } else {
    TF_PRun(session, handle, input_names.data(), inputs_unsafe.data(),
            input_names.size(), const_cast<const char**>(output_names.data()),
            outputs.data(), output_names.size(),
            const_cast<const char**>(target_nodes.data()), target_nodes.size(),
            out_status);
  }

  Py_END_ALLOW_THREADS;

  // Decref any numpy arrays we are not using anymore.
  ClearDecrefCache();

  if (TF_GetCode(out_status) != TF_OK) {
    return;
  }

  // 4. We now own the fetched tensors, so set up a safe container to
  // delete them when we exit this scope.
  std::vector<Safe_TF_TensorPtr> tf_outputs_safe;
  for (const auto& output : outputs) {
    tf_outputs_safe.emplace_back(make_safe(output));
  }

  // 5. Convert the fetched tensors into numpy ndarrays. Store them in a safe
  // container so that we do not leak
  std::vector<Safe_PyObjectPtr> py_outputs_safe;
  for (size_t i = 0; i < output_names.size(); ++i) {
    PyObject* py_array;
    s = TF_TensorToPyArray(std::move(tf_outputs_safe[i]), &py_array);
    if (!s.ok()) {
      Set_TF_Status_from_Status(out_status, s);
      return;
    }
    py_outputs_safe.emplace_back(
        make_safe(PyArray_Return(reinterpret_cast<PyArrayObject*>(py_array))));
  }

  // 6. If we reach this point, we have successfully built a list of objects
  // so we can release them from the safe container.
  for (auto& output : py_outputs_safe) {
    out_values->push_back(output.release());
  }
=======
// Container types for the various temporary values used internally in
// the wrapper.

// A TF_TensorVector is a vector of borrowed pointers to TF_Tensors.
typedef gtl::InlinedVector<TF_Tensor*, 8> TF_TensorVector;

// Safe containers for (an) owned TF_Tensor(s). On destruction, the
// tensor will be deleted by TF_DeleteTensor.
typedef std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)>
    Safe_TF_TensorPtr;
typedef std::vector<Safe_TF_TensorPtr> Safe_TF_TensorVector;
Safe_TF_TensorPtr make_safe(TF_Tensor* tensor) {
  return Safe_TF_TensorPtr(tensor, TF_DeleteTensor);
}

// Safe container for an owned TF_Status. On destruction, the status
// will be deleted by TF_DeleteStatus.
typedef std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)>
    Safe_TF_StatusPtr;
Safe_TF_StatusPtr make_safe(TF_Status* status) {
  return Safe_TF_StatusPtr(status, TF_DeleteStatus);
}

Status PyArrayDescr_to_TF_DataType(PyArray_Descr* descr,
                                   TF_DataType* out_tf_datatype) {
  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;
  if (PyDict_Next(descr->fields, &pos, &key, &value)) {
    const char* key_string = PyString_AsString(key);
    if (!key_string) {
      return errors::Internal("Corrupt numpy type descriptor");
    }
    tensorflow::string key = key_string;
    // The typenames here should match the field names in the custom struct
    // types constructed in test_util.py.
    // TODO(mrry,keveman): Investigate Numpy type registration to replace this
    // hard-coding of names.
    if (key == "quint8") {
      *out_tf_datatype = TF_QUINT8;
    } else if (key == "qint8") {
      *out_tf_datatype = TF_QINT8;
    } else if (key == "qint32") {
      *out_tf_datatype = TF_QINT32;
    } else {
      return errors::Internal("Unsupported numpy data type");
    }
    return Status::OK();
  }
  return errors::Internal("Unsupported numpy data type");
}

Status PyArray_TYPE_to_TF_DataType(PyArrayObject* array,
                                   TF_DataType* out_tf_datatype) {
  int pyarray_type = PyArray_TYPE(array);
  PyArray_Descr* descr = array->descr;
  switch (pyarray_type) {
    case NPY_FLOAT32:
      *out_tf_datatype = TF_FLOAT;
      break;
    case NPY_FLOAT64:
      *out_tf_datatype = TF_DOUBLE;
      break;
    case NPY_INT32:
      *out_tf_datatype = TF_INT32;
      break;
    case NPY_UINT8:
      *out_tf_datatype = TF_UINT8;
      break;
    case NPY_INT16:
      *out_tf_datatype = TF_INT16;
      break;
    case NPY_INT8:
      *out_tf_datatype = TF_INT8;
      break;
    case NPY_INT64:
      *out_tf_datatype = TF_INT64;
      break;
    case NPY_BOOL:
      *out_tf_datatype = TF_BOOL;
      break;
    case NPY_COMPLEX64:
      *out_tf_datatype = TF_COMPLEX;
      break;
    case NPY_OBJECT:
      *out_tf_datatype = TF_STRING;
      break;
    case NPY_VOID:
      // Quantized types are currently represented as custom struct types.
      // PyArray_TYPE returns NPY_VOID for structs, and we should look into
      // descr to derive the actual type.
      return PyArrayDescr_to_TF_DataType(descr, out_tf_datatype);
    default:
      // TODO(mrry): Support these.
      return errors::Internal("Unsupported feed type");
  }
  return Status::OK();
}

Status TF_DataType_to_PyArray_TYPE(TF_DataType tf_datatype,
                                   int* out_pyarray_type) {
  switch (tf_datatype) {
    case TF_FLOAT:
      *out_pyarray_type = NPY_FLOAT32;
      break;
    case TF_DOUBLE:
      *out_pyarray_type = NPY_FLOAT64;
      break;
    case TF_INT32:
      *out_pyarray_type = NPY_INT32;
      break;
    case TF_UINT8:
      *out_pyarray_type = NPY_UINT8;
      break;
    case TF_INT16:
      *out_pyarray_type = NPY_INT16;
      break;
    case TF_INT8:
      *out_pyarray_type = NPY_INT8;
      break;
    case TF_INT64:
      *out_pyarray_type = NPY_INT64;
      break;
    case TF_BOOL:
      *out_pyarray_type = NPY_BOOL;
      break;
    case TF_COMPLEX:
      *out_pyarray_type = NPY_COMPLEX64;
      break;
    case TF_STRING:
      *out_pyarray_type = NPY_OBJECT;
      break;
    // TODO(keveman): These should be changed to NPY_VOID, and the type used for
    // the resulting numpy array should be the custom struct types that we
    // expect for quantized types.
    case TF_QINT8:
      *out_pyarray_type = NPY_INT8;
      break;
    case TF_QUINT8:
      *out_pyarray_type = NPY_UINT8;
      break;
    case TF_QINT32:
      *out_pyarray_type = NPY_INT32;
      break;
    case TF_BFLOAT16:
      *out_pyarray_type = NPY_UINT16;
      break;
    default:
      return errors::Internal("Unsupported fetch type");
  }
  return Status::OK();
}

// Iterate over the string array 'array', extract the ptr and len of each string
// element and call f(ptr, len).
template <typename F>
Status PyStringArrayMap(PyArrayObject* array, F f) {
  Safe_PyObjectPtr iter = tensorflow::make_safe(
      PyArray_IterNew(reinterpret_cast<PyObject*>(array)));
  while (PyArray_ITER_NOTDONE(iter.get())) {
    auto item = tensorflow::make_safe(
        PyArray_GETITEM(array, PyArray_ITER_DATA(iter.get())));
    if (!item.get()) {
      return errors::Internal("Unable to get element from the feed.");
    }
    char* ptr;
    Py_ssize_t len;
    int success = PyString_AsStringAndSize(item.get(), &ptr, &len);
    if (success != 0) {
      return errors::Internal("Unable to get element from the feed.");
    }
    f(ptr, len);
    PyArray_ITER_NEXT(iter.get());
  }
  return Status::OK();
}

// Encode the strings in 'array' into a contiguous buffer and return the base of
// the buffer. The caller takes ownership of the buffer.
Status EncodePyStringArray(PyArrayObject* array, tensorflow::int64 nelems,
                           size_t* size, void** buffer) {
  // Compute bytes needed for encoding.
  *size = 0;
  TF_RETURN_IF_ERROR(
      PyStringArrayMap(array, [&size](char* ptr, Py_ssize_t len) {
        *size += sizeof(tensorflow::uint64) +
                 tensorflow::core::VarintLength(len) + len;
      }));
  // Encode all strings.
  std::unique_ptr<char[]> base_ptr(new char[*size]);
  char* base = base_ptr.get();
  char* data_start = base + sizeof(tensorflow::uint64) * nelems;
  char* dst = data_start;  // Where next string is encoded.
  tensorflow::uint64* offsets = reinterpret_cast<tensorflow::uint64*>(base);

  TF_RETURN_IF_ERROR(PyStringArrayMap(
      array, [&base, &data_start, &dst, &offsets](char* ptr, Py_ssize_t len) {
        *offsets = (dst - data_start);
        offsets++;
        dst = tensorflow::core::EncodeVarint64(dst, len);
        memcpy(dst, ptr, len);
        dst += len;
      }));
  CHECK_EQ(dst, base + *size);
  *buffer = base_ptr.release();
  return Status::OK();
}

// Determine the pointer and offset of the string at offset 'i' in the string
// tensor 'src', whose total length is 'num_elements'.
static Status TF_StringTensor_GetPtrAndLen(const TF_Tensor* src,
                                           tensorflow::int64 num_elements,
                                           tensorflow::int64 i,
                                           const char** ptr,
                                           tensorflow::uint64* len) {
  const char* input = reinterpret_cast<const char*>(TF_TensorData(src));
  const size_t src_size = TF_TensorByteSize(src);
  const char* data_start = input + sizeof(tensorflow::uint64) * num_elements;
  const char* limit = input + src_size;
  tensorflow::uint64 offset =
      reinterpret_cast<const tensorflow::uint64*>(input)[i];
  const char* p =
      tensorflow::core::GetVarint64Ptr(data_start + offset, limit, len);
  if (offset >= (limit - data_start) || !p || (*len > (limit - p))) {
    return errors::InvalidArgument("Malformed TF_STRING tensor; element ", i,
                                   " out of range");
  }
  *ptr = p;
  return Status::OK();
}

// Copy the string at offset 'i' in the (linearized) string tensor 'tensor' into
// 'pyarray' at offset pointed by the 'i_ptr' iterator.
static Status CopyStringToPyArrayElement(PyArrayObject* pyarray, void* i_ptr,
                                         TF_Tensor* tensor,
                                         tensorflow::int64 num_elements,
                                         tensorflow::int64 i) {
  const char* ptr;
  tensorflow::uint64 len;
  TF_RETURN_IF_ERROR(
      TF_StringTensor_GetPtrAndLen(tensor, num_elements, i, &ptr, &len));
  auto py_string = tensorflow::make_safe(PyString_FromStringAndSize(ptr, len));
  int success =
      PyArray_SETITEM(pyarray, PyArray_ITER_DATA(i_ptr), py_string.get());
  if (success != 0) {
    return errors::Internal("Error setting element ", i);
  }
  return Status::OK();
}

// Converts the given TF_Tensor to a Numpy array.
// If the returned status is OK, the caller becomes the owner of *out_array.
Status TF_Tensor_to_PyObject(TF_Tensor* tensor, PyObject** out_array) {
  // A fetched operation will correspond to a null tensor, and a None
  // in Python.
  if (tensor == nullptr) {
    Py_INCREF(Py_None);
    *out_array = Py_None;
    return Status::OK();
  }

  const int ndims = TF_NumDims(tensor);
  gtl::InlinedVector<npy_intp, 4> dims(ndims);
  tensorflow::int64 nelems = 1;
  for (int i = 0; i < ndims; ++i) {
    dims[i] = TF_Dim(tensor, i);
    nelems *= dims[i];
  }

  // Convert TensorFlow dtype to numpy type descriptor.
  int type_num;
  TF_RETURN_IF_ERROR(
      TF_DataType_to_PyArray_TYPE(TF_TensorType(tensor), &type_num));
  PyArray_Descr* descr = PyArray_DescrFromType(type_num);

  // Copy the TF_TensorData into a newly-created ndarray and return it.
  // TODO(mrry): Perhaps investigate zero-copy approaches. This would involve
  // creating an ndarray-like object that wraps the TF_Tensor buffer, and
  // maps its destructor to TF_DeleteTensor.
  Safe_PyObjectPtr safe_out_array =
      tensorflow::make_safe(PyArray_Empty(ndims, dims.data(), descr, 0));
  if (!safe_out_array) {
    return errors::Internal("Could not allocate ndarray");
  }
  PyArrayObject* py_array =
      reinterpret_cast<PyArrayObject*>(safe_out_array.get());
  if (PyArray_NBYTES(py_array) != TF_TensorByteSize(tensor)) {
    if (TF_TensorType(tensor) == TF_STRING) {
      // Copy element by element.
      auto iter = tensorflow::make_safe(PyArray_IterNew(safe_out_array.get()));
      for (tensorflow::int64 i = 0; i < nelems; ++i) {
        auto s =
            CopyStringToPyArrayElement(py_array, iter.get(), tensor, nelems, i);
        if (!s.ok()) {
          return s;
        }
        PyArray_ITER_NEXT(iter.get());
      }
    } else {
      return errors::Internal("ndarray was ", PyArray_NBYTES(py_array),
                              " bytes but TF_Tensor was ",
                              TF_TensorByteSize(tensor), " bytes");
    }
  } else {
    memcpy(py_array->data, TF_TensorData(tensor), PyArray_NBYTES(py_array));
  }

  // PyArray_Return turns rank 0 arrays into numpy scalars
  *out_array = PyArray_Return(
      reinterpret_cast<PyArrayObject*>(safe_out_array.release()));
  return Status::OK();
}

tensorflow::Status TF_Status_to_Status(TF_Status* tf_status) {
  TF_Code code = TF_GetCode(tf_status);
  const string message(TF_Message(tf_status));

  switch (code) {
    case TF_OK:
      return Status::OK();
    case TF_CANCELLED:
      return errors::Cancelled(message);
    case TF_UNKNOWN:
      return errors::Unknown(message);
    case TF_INVALID_ARGUMENT:
      return errors::InvalidArgument(message);
    case TF_DEADLINE_EXCEEDED:
      return errors::DeadlineExceeded(message);
    case TF_NOT_FOUND:
      return errors::NotFound(message);
    case TF_ALREADY_EXISTS:
      return errors::AlreadyExists(message);
    case TF_PERMISSION_DENIED:
      return errors::PermissionDenied(message);
    case TF_UNAUTHENTICATED:
      return errors::Unauthenticated(message);
    case TF_RESOURCE_EXHAUSTED:
      return errors::ResourceExhausted(message);
    case TF_FAILED_PRECONDITION:
      return errors::FailedPrecondition(message);
    case TF_ABORTED:
      return errors::Aborted(message);
    case TF_OUT_OF_RANGE:
      return errors::OutOfRange(message);
    case TF_UNIMPLEMENTED:
      return errors::Unimplemented(message);
    case TF_INTERNAL:
      return errors::Internal(message);
    case TF_UNAVAILABLE:
      return errors::Unavailable(message);
    case TF_DATA_LOSS:
      return errors::DataLoss(message);
    default:
      return errors::Internal("Got error with unknown code: ", code, " ",
                              message);
  }
}

static bool numpy_imported = false;

}  // namespace

Safe_PyObjectPtr make_safe(PyObject* o) {
  return Safe_PyObjectPtr(o, Py_DECREF_wrapper);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

// Wrapper for TF_Run that converts the arguments to appropriate types.
// If *out_status is OK, the caller becomes the owner of the PyObjects
// in *out_values.
<<<<<<< HEAD
void TF_Run_wrapper(TF_DeprecatedSession* session, const TF_Buffer* run_options,
                    PyObject* feed_dict, const NameVector& output_names,
                    const NameVector& target_nodes, TF_Status* out_status,
                    PyObjectVector* out_values, TF_Buffer* run_outputs) {
  TF_Run_wrapper_helper(session, nullptr, run_options, feed_dict, output_names,
                        target_nodes, out_status, out_values, run_outputs);
  ClearDecrefCache();
}

namespace {
void MakeCallableHelper(tensorflow::Session* session,
                        const TF_Buffer* callable_options, int64_t* out_handle,
                        TF_Status* out_status) {
  tensorflow::CallableOptions callable_options_proto;
  if (callable_options != nullptr &&
      !callable_options_proto.ParseFromArray(callable_options->data,
                                             callable_options->length)) {
    Set_TF_Status_from_Status(
        out_status,
        errors::InvalidArgument("Unparseable CallableOptions proto"));
    return;
  }
  tensorflow::Session::CallableHandle handle;
  Status s = session->MakeCallable(callable_options_proto, &handle);
  if (!s.ok()) {
    Set_TF_Status_from_Status(out_status, s);
    return;
  }
  *out_handle = handle;
}
}  // namespace

void TF_DeprecatedSessionMakeCallable(TF_DeprecatedSession* session,
                                      const TF_Buffer* callable_options,
                                      int64_t* out_handle, TF_Status* status) {
  MakeCallableHelper(session->session, callable_options, out_handle, status);
}
void TF_SessionMakeCallable(TF_Session* session,
                            const TF_Buffer* callable_options,
                            int64_t* out_handle, TF_Status* status) {
  MakeCallableHelper(session->session, callable_options, out_handle, status);
}

namespace {
void RunCallableHelper(tensorflow::Session* session, int64_t handle,
                       PyObject* feed_values, TF_Status* out_status,
                       PyObjectVector* out_values, TF_Buffer* run_metadata) {
  // Convert feed values to a vector of tensorflow::Tensor objects.
  std::vector<Tensor> input_tensors;
  Status s;
  {
    feed_values =
        PySequence_Fast(feed_values, "feed_values must be a sequence");
    if (feed_values == nullptr) return;
    Safe_PyObjectPtr feed_values_holder(make_safe(feed_values));
    Py_ssize_t len = PySequence_Fast_GET_SIZE(feed_values);
    input_tensors.reserve(len);
    for (Py_ssize_t i = 0; i < len; ++i) {
      PyObject* elem = PySequence_Fast_GET_ITEM(feed_values, i);
      if (!elem) {
        Set_TF_Status_from_Status(
            out_status, errors::Internal("Could not get feed value ", i));
        return;
      }
      Tensor t;
      s = NdarrayToTensor(elem, &t);
      if (!s.ok()) {
        Set_TF_Status_from_Status(out_status, s);
        return;
      }
      input_tensors.push_back(std::move(t));
    }
  }

  // Allocate a RunMetadata protobuf object to receive the metadata,
  // if the caller is expecting any.
  std::unique_ptr<RunMetadata> run_metadata_proto;
  if (run_metadata != nullptr) {
    run_metadata_proto.reset(new RunMetadata);
  }

  // Run the callable.
  std::vector<Tensor> output_tensors;
  Py_BEGIN_ALLOW_THREADS;
  s = session->RunCallable(handle, input_tensors, &output_tensors,
                           run_metadata_proto.get());
  Py_END_ALLOW_THREADS;

  if (!s.ok()) {
    Set_TF_Status_from_Status(out_status, s);
    return;
  }

  // If requested, serialize the RunMetadata to pass it back to the caller.
  if (run_metadata != nullptr) {
    s = MessageToBuffer(*run_metadata_proto, run_metadata);
    if (!s.ok()) {
      Set_TF_Status_from_Status(out_status, s);
      return;
    }
  }

  // Convert results to NumPy arrays. Since this can fail, stage the
  // results via a safe container that takes care of decreasing the
  // reference count on failure.
  std::vector<Safe_PyObjectPtr> py_outputs_safe;
  py_outputs_safe.reserve(output_tensors.size());
  for (const Tensor& output : output_tensors) {
    PyObject* py_array;
    s = TensorToNdarray(output, &py_array);
    if (!s.ok()) {
      Set_TF_Status_from_Status(out_status, s);
      return;
    }
    py_outputs_safe.push_back(
        make_safe(PyArray_Return(reinterpret_cast<PyArrayObject*>(py_array))));
  }

  // If we reach this point, we have successfully built a list of objects
  // so we can release them from the safe container.
  out_values->reserve(py_outputs_safe.size());
  for (auto& output : py_outputs_safe) {
    out_values->push_back(output.release());
  }
}
}  // namespace

void TF_DeprecatedSessionRunCallable(TF_DeprecatedSession* session,
                                     int64_t handle, PyObject* feed_values,
                                     PyObjectVector* out_values,
                                     TF_Buffer* run_metadata,
                                     TF_Status* status) {
  RunCallableHelper(session->session, handle, feed_values, status, out_values,
                    run_metadata);
  ClearDecrefCache();
}
void TF_SessionRunCallable(TF_Session* session, int64_t handle,
                           PyObject* feed_values, PyObjectVector* out_values,
                           TF_Buffer* run_metadata, TF_Status* status) {
  RunCallableHelper(session->session, handle, feed_values, status, out_values,
                    run_metadata);
  ClearDecrefCache();
}

void TF_DeprecatedSessionReleaseCallable(TF_DeprecatedSession* session,
                                         int64_t handle, TF_Status* status) {
  Set_TF_Status_from_Status(status, session->session->ReleaseCallable(handle));
}
void TF_SessionReleaseCallable(TF_Session* session, int64_t handle,
                               TF_Status* status) {
  Set_TF_Status_from_Status(status, session->session->ReleaseCallable(handle));
}

// Wrapper for TF_PRunSetup that converts the arguments to appropriate types.
// If *out_status is OK, the caller becomes the owner of *out_handle.
void TF_PRunSetup_wrapper(TF_DeprecatedSession* session,
                          const NameVector& input_names,
                          const NameVector& output_names,
                          const NameVector& target_nodes, TF_Status* out_status,
                          const char** out_handle) {
  Py_BEGIN_ALLOW_THREADS;
  TF_PRunSetup(
      session, const_cast<const char**>(input_names.data()), input_names.size(),
      const_cast<const char**>(output_names.data()), output_names.size(),
      const_cast<const char**>(target_nodes.data()), target_nodes.size(),
      out_handle, out_status);
  Py_END_ALLOW_THREADS;
}

// Wrapper for TF_PRun that converts the arguments to appropriate types.
// If *out_status is OK, the caller becomes the owner of the PyObjects
// in *out_values.
void TF_PRun_wrapper(TF_DeprecatedSession* session, const char* handle,
                     PyObject* feed_dict, const NameVector& output_names,
                     TF_Status* out_status, PyObjectVector* out_values) {
  TF_Run_wrapper_helper(session, handle, nullptr, feed_dict, output_names,
                        NameVector(), out_status, out_values, nullptr);
  ClearDecrefCache();
}

// Wrapper for TF_Reset that converts the string vectors to character arrays.
void TF_Reset_wrapper(const TF_SessionOptions* opt,
                      const NameVector& containers, TF_Status* status) {
  TF_Reset(opt, const_cast<const char**>(containers.data()), containers.size(),
           status);
}

void TF_SessionRun_wrapper_helper(TF_Session* session, const char* handle,
                                  const TF_Buffer* run_options,
                                  const std::vector<TF_Output>& inputs,
                                  const std::vector<PyObject*>& input_ndarrays,
                                  const std::vector<TF_Output>& outputs,
                                  const std::vector<TF_Operation*>& targets,
                                  TF_Buffer* run_metadata,
                                  TF_Status* out_status,
                                  std::vector<PyObject*>* py_outputs) {
  DCHECK_EQ(inputs.size(), input_ndarrays.size());
  DCHECK(py_outputs != nullptr);
  DCHECK(py_outputs->empty());
  Status s;

  // Convert input ndarray PyObjects to TF_Tensors. We maintain a continuous
  // array of TF_Tensor*s as well as scoped containers to make sure they're
  // cleaned up properly.
  //
  // Memory management:
  // PyArrayToTF_Tensor() creates a new ndarray PyObject from the input
  // ndarray. We manage the new ndarray's lifetime in order to keep the
  // underlying data buffer alive (the new ndarray also guarantees a contiguous
  // data buffer). The new ndarray's data buffer is used to create the
  // corresponding TF_Tensor. The TF_Tensor's deallocator will queue the new
  // ndarray to be decref'd by the next ClearDecrefCache() call (we can't call
  // Py_DECREF in the deallocator directly because the GIL must be held).
  //
  // Note that TF_Tensor may directly delegate its data and deallocator to a
  // TensorBuffer, which may outlive the TF_Tensor (e.g. if the tensor gets
  // queued or assigned to a variable).
  TF_TensorVector input_vals;
  std::vector<Safe_TF_TensorPtr> input_vals_safe;
  for (PyObject* ndarray : input_ndarrays) {
    input_vals_safe.emplace_back(make_safe(static_cast<TF_Tensor*>(nullptr)));
    s = PyArrayToTF_Tensor(ndarray, &input_vals_safe.back());
    if (!s.ok()) {
      Set_TF_Status_from_Status(out_status, s);
      return;
    }
    input_vals.push_back(input_vals_safe.back().get());
  }

  // Allocate space for output TF_Tensor*s
  TF_TensorVector output_vals(outputs.size());

  // Clear up any unused memory leftover from previous runs
  ClearDecrefCache();

  // Call TF_SessionRun() (and release GIL during execution)
  Py_BEGIN_ALLOW_THREADS;
  if (handle == nullptr) {
    TF_SessionRun(session, run_options, inputs.data(), input_vals.data(),
                  inputs.size(), outputs.data(), output_vals.data(),
                  outputs.size(), targets.data(), targets.size(), run_metadata,
                  out_status);
  } else {
    TF_SessionPRun(session, handle, inputs.data(), input_vals.data(),
                   inputs.size(), outputs.data(), output_vals.data(),
                   outputs.size(), targets.data(), targets.size(), out_status);
  }
  Py_END_ALLOW_THREADS;

  // Create scoped containers for output tensors
  std::vector<Safe_TF_TensorPtr> output_vals_safe;
  for (TF_Tensor* output : output_vals) {
    output_vals_safe.emplace_back(make_safe(output));
  }

  // Convert outputs to ndarrays (in scoped containers)
  std::vector<Safe_PyObjectPtr> py_outputs_safe;
  for (size_t i = 0; i < outputs.size(); ++i) {
    PyObject* py_array;
    s = TF_TensorToPyArray(std::move(output_vals_safe[i]), &py_array);
    if (!s.ok()) {
      Set_TF_Status_from_Status(out_status, s);
      return;
    }
    py_outputs_safe.emplace_back(
        make_safe(PyArray_Return(reinterpret_cast<PyArrayObject*>(py_array))));
  }

  // If we reach this point, we have successfully built a list of objects so we
  // can release them from the safe container into the return vector.
  for (size_t i = 0; i < outputs.size(); ++i) {
    py_outputs->push_back(py_outputs_safe[i].release());
  }
}

void TF_SessionRun_wrapper(TF_Session* session, const TF_Buffer* run_options,
                           const std::vector<TF_Output>& inputs,
                           const std::vector<PyObject*>& input_ndarrays,
                           const std::vector<TF_Output>& outputs,
                           const std::vector<TF_Operation*>& targets,
                           TF_Buffer* run_metadata, TF_Status* out_status,
                           std::vector<PyObject*>* py_outputs) {
  TF_SessionRun_wrapper_helper(session, nullptr, run_options, inputs,
                               input_ndarrays, outputs, targets, run_metadata,
                               out_status, py_outputs);
  // Release any unused ndarray references (see memory management comment in
  // TF_SessionRun_wrapper_helper)
  ClearDecrefCache();
}

string EqualGraphDefWrapper(const string& actual, const string& expected) {
  GraphDef actual_def;
  if (!actual_def.ParseFromString(actual)) {
    return "actual is not a valid serialized GraphDef";
  }
  GraphDef expected_def;
  if (!expected_def.ParseFromString(expected)) {
    return "expected is not a valid serialized GraphDef";
  }
  string diff;
  return EqualGraphDef(actual_def, expected_def, &diff) ? "" : diff;
}

string EqualAttrValueWrapper(const string& actual, const string& expected) {
  AttrValue actual_attr_value;
  if (!actual_attr_value.ParseFromString(actual)) {
    return "actual is not a valid serialized AttrValue";
  }

  AttrValue expected_attr_value;
  if (!expected_attr_value.ParseFromString(expected)) {
    return "expected is not a valid serialized AttrValue";
  }

  string diff;
  if (!AreAttrValuesEqual(actual_attr_value, expected_attr_value)) {
    diff = strings::Printf(
        "Actual AttrValue %s does not match Expected AttrValue %s.",
        SummarizeAttrValue(actual_attr_value).c_str(),
        SummarizeAttrValue(expected_attr_value).c_str());
  }
  return diff;
}

// Return value set to 6 inlined elements so it fits in a 64-byte cache line.
tensorflow::gtl::InlinedVector<int64_t, 6> TF_GraphGetTensorShapeHelper(
    TF_Graph* graph, TF_Output output, TF_Status* out_status,
    bool* unknown_shape) {
  // Allocate a single variable for holding the result for RVO.
  tensorflow::gtl::InlinedVector<int64_t, 6> result;
  *unknown_shape = false;
  int num_dims = TF_GraphGetTensorNumDims(graph, output, out_status);
  if (TF_GetCode(out_status) != TF_OK) {
    return result;
  }
  // If shape is unknown, set boolean and return.
  if (num_dims == -1) {
    *unknown_shape = true;
    return result;
  }

  // If shape is a scalar, avoid another C call and just return {}.
  if (num_dims == 0) {
    return result;
  }

  result.resize(num_dims);
  TF_GraphGetTensorShape(graph, output, result.data(), num_dims, out_status);
  return result;
}

void TF_SessionPRunSetup_wrapper(TF_Session* session,
                                 const std::vector<TF_Output>& inputs,
                                 const std::vector<TF_Output>& outputs,
                                 const std::vector<TF_Operation*>& targets,
                                 const char** out_handle,
                                 TF_Status* out_status) {
  // Call TF_SessionPRunSetup() (and release GIL during execution)
  Py_BEGIN_ALLOW_THREADS;
  TF_SessionPRunSetup(session, inputs.data(), inputs.size(), outputs.data(),
                      outputs.size(), targets.data(), targets.size(),
                      out_handle, out_status);
  Py_END_ALLOW_THREADS;
}

void TF_SessionPRun_wrapper(TF_Session* session, const char* handle,
                            const std::vector<TF_Output>& inputs,
                            const std::vector<PyObject*>& input_ndarrays,
                            const std::vector<TF_Output>& outputs,
                            TF_Status* out_status,
                            std::vector<PyObject*>* py_outputs) {
  const std::vector<TF_Operation*> targets;
  TF_SessionRun_wrapper_helper(session, handle,
                               nullptr,  // run_options
                               inputs, input_ndarrays, outputs, targets,
                               nullptr,  // run_metadata
                               out_status, py_outputs);
  // Release any unused ndarray references (see memory management comment in
  // TF_SessionRun_wrapper_helper)
  ClearDecrefCache();
}

std::vector<TF_Output> GetOperationInputs(TF_Operation* oper) {
  int num_inputs = TF_OperationNumInputs(oper);
  std::vector<TF_Output> inputs(num_inputs);
  TF_OperationAllInputs(oper, inputs.data(), inputs.size());
  return inputs;
}

std::vector<TF_Operation*> TF_OperationGetControlInputs_wrapper(
    TF_Operation* oper) {
  std::vector<TF_Operation*> control_inputs(TF_OperationNumControlInputs(oper));
  TF_OperationGetControlInputs(oper, control_inputs.data(),
                               control_inputs.size());
  return control_inputs;
}

std::vector<TF_Operation*> TF_OperationGetControlOutputs_wrapper(
    TF_Operation* oper) {
  std::vector<TF_Operation*> control_outputs(
      TF_OperationNumControlOutputs(oper));
  TF_OperationGetControlOutputs(oper, control_outputs.data(),
                                control_outputs.size());
  return control_outputs;
}

std::vector<const char*> TF_OperationOutputConsumers_wrapper(
    TF_Output oper_out) {
  int num_consumers = TF_OperationOutputNumConsumers(oper_out);
  std::vector<TF_Input> consumers(num_consumers);
  TF_OperationOutputConsumers(oper_out, consumers.data(), num_consumers);

  std::vector<const char*> consumer_names(num_consumers);
  for (int i = 0; i < num_consumers; ++i) {
    consumer_names[i] = TF_OperationName(consumers[i].oper);
  }
  return consumer_names;
}

TF_Function* TF_GraphToFunction_wrapper(
    const TF_Graph* fn_body, const char* fn_name, bool append_hash_to_fn_name,
    const std::vector<TF_Operation*>* opers,
    const std::vector<TF_Output>& inputs, const std::vector<TF_Output>& outputs,
    const NameVector& output_names,
    const std::vector<TF_Operation*>* control_outputs,
    const NameVector& control_output_names, const TF_FunctionOptions* opts,
    const char* description, TF_Status* out_status) {
  if (!output_names.empty() && output_names.size() != outputs.size()) {
    Set_TF_Status_from_Status(
        out_status,
        errors::InvalidArgument(
            "output names must be either empty or equal in size to outputs. ",
            "output names size = ", output_names.size(),
            " outputs size = ", outputs.size()));
    return nullptr;
  }

  int nopers = -1;
  const TF_Operation* const* opers_array = nullptr;
  if (opers != nullptr) {
    nopers = opers->size();
    opers_array = opers->data();
  }

  const char** output_names_ptr =
      output_names.empty() ? nullptr
                           : const_cast<const char**>(output_names.data());

  const char** control_output_names_ptr =
      control_output_names.empty()
          ? nullptr
          : const_cast<const char**>(control_output_names.data());

  return TF_GraphToFunctionWithControlOutputs(
      fn_body, fn_name, append_hash_to_fn_name, nopers, opers_array,
      inputs.size(), inputs.data(), outputs.size(), outputs.data(),
      output_names_ptr,
      control_outputs == nullptr ? 0 : control_outputs->size(),
      control_outputs == nullptr ? nullptr : control_outputs->data(),
      control_output_names_ptr, opts, description, out_status);
}

void TF_GraphSetOutputHandleShapesAndTypes_wrapper(
    TF_Graph* graph, TF_Output output,
    const std::vector<std::vector<int64_t>>& shapes,
    const std::vector<int>& ranks, const std::vector<TF_DataType>& types,
    TF_Status* status) {
  std::vector<const int64_t*> shapes_pointers(shapes.size());
  for (int i = 0; i < shapes.size(); ++i) {
    shapes_pointers[i] = ranks[i] <= 0 ? nullptr : &shapes[i][0];
  }
  TF_GraphSetOutputHandleShapesAndTypes(graph, output, shapes.size(),
                                        shapes_pointers.data(), ranks.data(),
                                        types.data(), status);
}

void CreatePlaceholder(TF_Graph* graph, TF_Status* s, string&& name,
                       TF_DataType dtype, TF_Output* output) {
  TF_OperationDescription* desc =
      TF_NewOperation(graph, "Placeholder", name.data());
  TF_SetAttrType(desc, "dtype", dtype);
  TF_Operation* op = TF_FinishOperation(desc, s);
  output->oper = op;
  output->index = 0;
}

std::vector<TF_Output> TF_CreatePlaceholders(TF_Graph* graph, PyObject* dtypes,
                                             const char* prefix,
                                             TF_Status* status) {
  std::vector<TF_Output> outputs;
  dtypes = PySequence_Fast(dtypes, "dtypes must be a sequence");
  if (dtypes == nullptr) {
    Set_TF_Status_from_Status(status, errors::Internal("dtypes is nullptr"));
    return outputs;
  }
  Safe_PyObjectPtr dtypes_holder(make_safe(dtypes));
  Py_ssize_t len = PySequence_Fast_GET_SIZE(dtypes);
  outputs.reserve(len);
  for (size_t i = 0; i < len; i++) {
    PyObject* dtype = PySequence_Fast_GET_ITEM(dtypes, i);
    if (!dtype) {
      Set_TF_Status_from_Status(status,
                                errors::Internal("Could not get dtype ", i));
      return outputs;
    }
#if PY_MAJOR_VERSION >= 3
    TF_DataType tf_datatype = static_cast<TF_DataType>(PyLong_AsLong(dtype));
#else
    TF_DataType tf_datatype = static_cast<TF_DataType>(PyInt_AsLong(dtype));
#endif
    outputs.push_back(TF_Output());
    CreatePlaceholder(graph, status, strings::StrCat(prefix, i), tf_datatype,
                      &outputs.back());
    if (!status->status.ok()) break;
  }
  return outputs;
}

void TF_GraphSetTensorShape_wrapper(TF_Graph* graph, TF_Output output,
                                    const std::vector<int64_t>& dims,
                                    bool unknown_shape, TF_Status* status) {
  if (unknown_shape) {
    TF_GraphSetTensorShape(graph, output, nullptr, -1, status);
    return;
  }
  TF_GraphSetTensorShape(graph, output, dims.data(), dims.size(), status);
}

std::vector<string> TF_ImportGraphDefResultsMissingUnusedInputMappings_wrapper(
    TF_ImportGraphDefResults* results) {
  int num_missing_unused_input_mappings;
  const char** src_names;
  int* src_indexes;
  TF_ImportGraphDefResultsMissingUnusedInputMappings(
      results, &num_missing_unused_input_mappings, &src_names, &src_indexes);
  std::vector<string> input_strs(num_missing_unused_input_mappings);
  for (int i = 0; i < num_missing_unused_input_mappings; ++i) {
    input_strs[i] = TensorId(src_names[i], src_indexes[i]).ToString();
  }
  return input_strs;
}

PyObject* TF_TryEvaluateConstant_wrapper(TF_Graph* graph, TF_Output output,
                                         TF_Status* status) {
  TF_Tensor* result_tensor;
  bool evaluated =
      TF_TryEvaluateConstant(graph, output, &result_tensor, status);
  if (!evaluated || TF_GetCode(status) != TF_OK) Py_RETURN_NONE;

  Safe_TF_TensorPtr safe_result_tensor(result_tensor);
  PyObject* out;
  Status s = TF_TensorToPyArray(std::move(safe_result_tensor), &out);
  Set_TF_Status_from_Status(status, s);
  if (!s.ok()) Py_RETURN_NONE;
  return PyArray_Return(reinterpret_cast<PyArrayObject*>(out));
=======
void TF_Run_wrapper(TF_Session* session, const FeedVector& inputs,
                    const NameVector& output_names,
                    const NameVector& target_nodes, Status* out_status,
                    PyObjectVector* out_values) {
  // 0. Ensure that numpy has been imported.
  if (!numpy_imported) {
    import_array();
    numpy_imported = true;
  }

  // 1. Convert the feed inputs to the appropriate form for TF_Run.
  NameVector input_names;
  Safe_PyObjectVector
      py_inputs_safe;  // Used to decref the input arrays on failure.
  Safe_TF_TensorVector inputs_safe;  // Used to delete tensors on failure.
  TF_TensorVector inputs_unsafe;     // Used to contain the arg to TF_Run.

  for (const auto& name_and_array : inputs) {
    py_inputs_safe.emplace_back(
        make_safe(reinterpret_cast<PyObject*>(name_and_array.second)));
  }

  for (int i = 0; i < inputs.size(); ++i) {
    input_names.push_back(inputs[i].first);
    PyArrayObject* array = inputs[i].second;

    // Convert numpy dtype to TensorFlow dtype.
    TF_DataType dtype;
    *out_status = PyArray_TYPE_to_TF_DataType(array, &dtype);
    if (!out_status->ok()) {
      return;
    }

    tensorflow::int64 nelems = 1;
    gtl::InlinedVector<tensorflow::int64, 4> dims;
    for (int i = 0; i < PyArray_NDIM(array); ++i) {
      dims.push_back(PyArray_SHAPE(array)[i]);
      nelems *= dims[i];
    }

    // Create a TF_Tensor based on the fed data. In the case of non-string data
    // type, this steals a reference to array, which will be relinquished when
    // the underlying buffer is deallocated. For string, a new temporary buffer
    // is allocated into which the strings are encoded.
    if (dtype != TF_STRING) {
      // NOTE(mrry): We currently copy the numpy array into a new
      // buffer to avoid possible issues on deallocation (such as
      // having to acquire the Python Global Interpreter Lock).
      // TODO(mrry): Investigate in what cases we can safely acquire
      size_t size = PyArray_NBYTES(array);
      // NOTE(mrry): 32 is the upper bound on current alignment
      // requirements for tensorflow::Tensor. We hard code this here to
      // avoid taking a dependency on Eigen in the client code.
      void* data = tensorflow::cpu_allocator()->AllocateRaw(32, size);
      std::memcpy(data, array->data, size);
      inputs_safe.emplace_back(make_safe(
          TF_NewTensor(dtype, dims.data(), dims.size(), data, size,
                       [](void* data, size_t len, void* arg) {
                         tensorflow::cpu_allocator()->DeallocateRaw(data);
                       },
                       nullptr)));
      // The destruction of the numpy array will now be handled by the
      // inputs_safe destructor.
      py_inputs_safe[i].reset();
    } else {
      size_t size;
      void* encoded;
      Status s = EncodePyStringArray(array, nelems, &size, &encoded);
      if (!s.ok()) {
        *out_status = s;
        return;
      }
      inputs_safe.emplace_back(
          make_safe(TF_NewTensor(dtype, dims.data(), dims.size(), encoded, size,
                                 [](void* data, size_t len, void* arg) {
                                   delete[] reinterpret_cast<char*>(data);
                                 },
                                 array)));
      // The destruction of the numpy array will now be handled by the
      // inputs_safe destructor.
      py_inputs_safe[i].reset();
    }
    inputs_unsafe.push_back(inputs_safe.back().get());
  }

  // 2. Allocate a container for the output data.
  TF_TensorVector outputs(output_names.size());

  Safe_TF_StatusPtr status = make_safe(TF_NewStatus());

  // 3. Actually call TF_Run().
  Py_BEGIN_ALLOW_THREADS;
  TF_Run(session, input_names.data(), inputs_unsafe.data(), input_names.size(),
         const_cast<const char**>(output_names.data()), outputs.data(),
         output_names.size(), const_cast<const char**>(target_nodes.data()),
         target_nodes.size(), status.get());
  Py_END_ALLOW_THREADS;

  // 4. The TensorFlow runtime has taken ownership of the fed tensors,
  // so we release the safe pointers to them.
  for (auto& input : inputs_safe) {
    input.release();
  }

  if (TF_GetCode(status.get()) != TF_OK) {
    *out_status = TF_Status_to_Status(status.get());
    return;
  }

  // 5. We now own the fetched tensors, so set up a safe container to
  // delete them when we exit this scope.
  Safe_TF_TensorVector tf_outputs_safe;
  for (const auto& output : outputs) {
    tf_outputs_safe.emplace_back(make_safe(output));
  }

  // 6. Convert the fetched tensors into numpy ndarrays. Store them in a safe
  // container so that we do not leak
  Safe_PyObjectVector py_outputs_safe;
  for (int i = 0; i < output_names.size(); ++i) {
    PyObject* py_array;
    *out_status = TF_Tensor_to_PyObject(outputs[i], &py_array);
    if (!out_status->ok()) {
      return;
    }
    py_outputs_safe.emplace_back(make_safe(py_array));
  }

  // 7. If we reach this point, we have successfully built a list of objects
  // so we can release them from the safe container.
  for (auto& output : py_outputs_safe) {
    out_values->push_back(output.release());
  }
  *out_status = Status::OK();
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

}  // namespace tensorflow
