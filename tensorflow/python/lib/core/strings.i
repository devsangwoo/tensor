<<<<<<< HEAD
/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
// Wrapper functions to provide a scripting-language-friendly interface
// to our string libraries.
//
// NOTE: as of 2005-01-13, this SWIG file is not used to generate a pywrap
//       library for manipulation of various string-related types or access
//       to the special string functions (Python has plenty). This SWIG file
//       should be %import'd so that other SWIG wrappers have proper access
//       to the types in //strings (such as the StringPiece object). We may
//       generate a pywrap at some point in the future.
//
// NOTE: (Dan Ardelean) as of 2005-11-15 added typemaps to convert Java String
//       arguments to C++ StringPiece& objects. This is required because a
//       StringPiece class does not make sense - the code SWIG generates for a
//       StringPiece class is useless, because it releases the buffer set in
//       StringPiece after creating the object. C++ StringPiece objects rely on
//       the buffer holding the data being allocated externally.

// NOTE: for now, we'll just start with what is needed, and add stuff
//       as it comes up.

%{
#include "tensorflow/core/lib/core/stringpiece.h"
<<<<<<< HEAD

// Handles str in Python 2, bytes in Python 3.
// Returns true on success, false on failure.
bool _BytesToStringPiece(PyObject* obj, tensorflow::StringPiece* result) {
  if (obj == Py_None) {
    *result = tensorflow::StringPiece();
  } else {
    char* ptr;
    Py_ssize_t len;
    if (PyBytes_AsStringAndSize(obj, &ptr, &len) == -1) {
      // Python has raised an error (likely TypeError or UnicodeEncodeError).
      return false;
    }
    *result = tensorflow::StringPiece(ptr, len);
  }
  return true;
}
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
%}

%typemap(typecheck) tensorflow::StringPiece = char *;
%typemap(typecheck) const tensorflow::StringPiece & = char *;

<<<<<<< HEAD
// "tensorflow::StringPiece" arguments must be specified as a 'str' or 'bytes' object.
%typemap(in) tensorflow::StringPiece {
  if (!_BytesToStringPiece($input, &$1)) SWIG_fail;
=======
// "tensorflow::StringPiece" arguments can be provided by a simple Python 'str' string
// or a 'unicode' object. If 'unicode', it's translated using the default
// encoding, i.e., sys.getdefaultencoding(). If passed None, a tensorflow::StringPiece
// of zero length with a NULL pointer is provided.
%typemap(in) tensorflow::StringPiece {
  if ($input != Py_None) {
    char * buf;
    Py_ssize_t len;
%#if PY_VERSION_HEX >= 0x03030000
    /* Do unicode handling as PyBytes_AsStringAndSize doesn't in Python 3. */
    if (PyUnicode_Check($input)) {
      buf = PyUnicode_AsUTF8AndSize($input, &len);
      if (buf == NULL)
        SWIG_fail;
    } else {
%#elif PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 3
%#  error "Unsupported Python 3.x C API version (3.3 or later required)."
%#endif
      if (PyBytes_AsStringAndSize($input, &buf, &len) == -1) {
        // Python has raised an error (likely TypeError or UnicodeEncodeError).
        SWIG_fail;
      }
%#if PY_VERSION_HEX >= 0x03030000
    }
%#endif
    $1.set(buf, len);
  }
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

// "const tensorflow::StringPiece&" arguments can be provided the same as
// "tensorflow::StringPiece", whose typemap is defined above.
%typemap(in) const tensorflow::StringPiece & (tensorflow::StringPiece temp) {
<<<<<<< HEAD
  if (!_BytesToStringPiece($input, &temp)) SWIG_fail;
  $1 = &temp;
}

// C++ functions returning tensorflow::StringPiece will simply return bytes in
// Python, or None if the StringPiece contained a NULL pointer.
%typemap(out) tensorflow::StringPiece {
  if ($1.data()) {
    $result = PyBytes_FromStringAndSize($1.data(), $1.size());
=======
  if ($input != Py_None) {
    char * buf;
    Py_ssize_t len;
%#if PY_VERSION_HEX >= 0x03030000
    /* Do unicode handling as PyBytes_AsStringAndSize doesn't in Python 3. */
    if (PyUnicode_Check($input)) {
      buf = PyUnicode_AsUTF8AndSize($input, &len);
      if (buf == NULL)
        SWIG_fail;
    } else {
%#elif PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 3
%#  error "Unsupported Python 3.x C API version (3.3 or later required)."
%#endif
      if (PyBytes_AsStringAndSize($input, &buf, &len) == -1) {
        // Python has raised an error (likely TypeError or UnicodeEncodeError).
        SWIG_fail;
      }
%#if PY_VERSION_HEX >= 0x03030000
    }
%#endif
    temp.set(buf, len);
  }
  $1 = &temp;
}

// C++ functions returning tensorflow::StringPiece will simply return bytes in Python,
// or None if the StringPiece contained a NULL pointer.
%typemap(out) tensorflow::StringPiece {
  if ($1.data()) {
    $result = PyString_FromStringAndSize($1.data(), $1.size());
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  } else {
    Py_INCREF(Py_None);
    $result = Py_None;
  }
}
<<<<<<< HEAD

// Converts a C++ string vector to a list of Python bytes objects.
%typemap(out) std::vector<string> {
  const int size = $1.size();
  auto temp_string_list = tensorflow::make_safe(PyList_New(size));
  if (!temp_string_list) {
    SWIG_fail;
  }
  std::vector<tensorflow::Safe_PyObjectPtr> converted;
  converted.reserve(size);
  for (const string& op : $1) {
    // Always treat strings as bytes, consistent with the typemap
    // for string.
    PyObject* py_str = PyBytes_FromStringAndSize(op.data(), op.size());
    if (!py_str) {
      SWIG_fail;
    }
    converted.emplace_back(tensorflow::make_safe(py_str));
  }
  for (int i = 0; i < converted.size(); ++i) {
    PyList_SET_ITEM(temp_string_list.get(), i, converted[i].release());
  }
  $result = temp_string_list.release();
}
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
