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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {

Status ScalarInputsAndOutputs(InferenceContext* c) {
  ShapeHandle unused;
  for (int i = 0; i < c->num_inputs(); ++i) {
    TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 0, &unused));
  }
  for (int i = 0; i < c->num_outputs(); ++i) {
    c->set_output(i, c->Scalar());
  }
  return Status::OK();
}

Status TwoElementVectorAndScalarOutputs(InferenceContext* c) {
  ShapeHandle handle;
  DimensionHandle unused_handle;
  for (int i = 0; i < c->num_inputs(); ++i) {
    TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &handle));
    TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_handle));
  }
  for (int i = 0; i < c->num_outputs(); ++i) {
    c->set_output(i, c->Scalar());
  }
  return Status::OK();
}

Status TwoElementOutput(InferenceContext* c) {
  c->set_output(0, c->Vector(2));
  return Status::OK();
}

}  // namespace

REGISTER_OP("SaveV2")
    .Input("prefix: string")
    .Input("tensor_names: string")
    .Input("shape_and_slices: string")
    .Input("tensors: dtypes")
    .Attr("dtypes: list(type)")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      ShapeHandle s;
      DimensionHandle unused_dim;

      // Validate prefix.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));

      // Validate tensor_names and shapes_and_slices.
      for (int i = 1; i <= 2; ++i) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &s));
        TF_RETURN_IF_ERROR(
            c->WithValue(c->Dim(s, 0), c->num_inputs() - 3, &unused_dim));
      }
      // TODO(mrry): Attempt to parse the shapes_and_slices values and use
      // them to constrain the shape of the remaining inputs.
      return Status::OK();
    });

REGISTER_OP("RestoreV2")
    .Input("prefix: string")
    .Input("tensor_names: string")
    .Input("shape_and_slices: string")
    .Output("tensors: dtypes")
    .Attr("dtypes: list(type)")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle shape0, shape1, shape2;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &shape0));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &shape1));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &shape2));
      TF_RETURN_IF_ERROR(c->Merge(shape1, shape2, &shape0));

      // Attempt to infer output shapes from its shape_and_slice input.
      const Tensor* shape_and_slices_tensor = c->input_tensor(2);
      if (shape_and_slices_tensor) {
        const auto& shape_and_slices_flat =
            shape_and_slices_tensor->flat<tstring>();
        if (shape_and_slices_flat.size() != c->num_outputs()) {
          return errors::InvalidArgument(
              "The number of shape_and_slice doesn't match tensor outputs.");
        }
        for (int i = 0; i < shape_and_slices_flat.size(); ++i) {
          const string& shape_and_slice = shape_and_slices_flat(i);
          if (shape_and_slice.empty()) {
            c->set_output(i, c->UnknownShape());
            continue;
          }
          TensorShape parsed_full_shape;
          TensorSlice parsed_slice;
          TensorShape parsed_slice_shape;
          TF_RETURN_IF_ERROR(checkpoint::ParseShapeAndSlice(
              shape_and_slice, &parsed_full_shape, &parsed_slice,
              &parsed_slice_shape));
          ShapeHandle shape_handle;
          TF_RETURN_IF_ERROR(
              c->MakeShapeFromTensorShape(parsed_slice_shape, &shape_handle));
          c->set_output(i, shape_handle);
        }
        return Status::OK();
      } else {
        return UnknownShape(c);
      }
    });

REGISTER_OP("MergeV2Checkpoints")
    .Input("checkpoint_prefixes: string")
    .Input("destination_prefix: string")
    .Attr("delete_old_dirs: bool = true")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return Status::OK();
    });

=======
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/platform/port.h"

namespace tensorflow {

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
REGISTER_OP("Save")
    .Input("filename: string")
    .Input("tensor_names: string")
    .Input("data: T")
<<<<<<< HEAD
    .Attr("T: list(type)")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      ShapeHandle s;
      DimensionHandle unused_dim;

      // Validate filename.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));

      // Validate tensor_names.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &s));
      TF_RETURN_IF_ERROR(
          c->WithValue(c->Dim(s, 0), c->num_inputs() - 2, &unused_dim));

      return Status::OK();
    });
=======
    .Attr("T: list({float, double, int32, int64, quint8, qint8, qint32})")
    .Doc(R"doc(
Saves the input tensors to disk.

The size of `tensor_names` must match the number of tensors in `data`. `data[i]`
is written to `filename` with name `tensor_names[i]`.

See also `SaveSlices`.

filename: Must have a single element. The name of the file to which we write
the tensor.
tensor_names: Shape `[N]`. The names of the tensors to be saved.
data: `N` tensors to save.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("SaveSlices")
    .Input("filename: string")
    .Input("tensor_names: string")
    .Input("shapes_and_slices: string")
    .Input("data: T")
<<<<<<< HEAD
    .Attr("T: list(type)")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      ShapeHandle s;
      DimensionHandle unused_dim;

      // Validate filename.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));

      // Validate tensor_names and unused_shapes_and_slices.
      for (int i = 1; i <= 2; ++i) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &s));
        TF_RETURN_IF_ERROR(
            c->WithValue(c->Dim(s, 0), c->num_inputs() - 3, &unused_dim));
      }
      // TODO(mrry): Attempt to parse the shapes_and_slices values and use
      // them to constrain the shape of the remaining inputs.
      return Status::OK();
    });
=======
    .Attr("T: list({float, double, int32, int64, quint8, qint8, qint32})")
    .Doc(R"doc(
Saves input tensors slices to disk.

This is like `Save` except that tensors can be listed in the saved file as being
a slice of a larger tensor.  `shapes_and_slices` specifies the shape of the
larger tensor and the slice that this tensor covers. `shapes_and_slices` must
have as many elements as `tensor_names`.

Elements of the `shapes_and_slices` input must either be:

*  The empty string, in which case the corresponding tensor is
   saved normally.
*  A string of the form `dim0 dim1 ... dimN-1 slice-spec` where the
   `dimI` are the dimensions of the larger tensor and `slice-spec`
   specifies what part is covered by the tensor to save.

`slice-spec` itself is a `:`-separated list: `slice0:slice1:...:sliceN-1`
where each `sliceI` is either:

*  The string `-` meaning that the slice covers all indices of this dimension
*  `start,length` where `start` and `length` are integers.  In that
   case the slice covers `length` indices starting at `start`.

See also `Save`.

filename: Must have a single element. The name of the file to which we write the
tensor.
tensor_names: Shape `[N]`. The names of the tensors to be saved.
shapes_and_slices: Shape `[N]`.  The shapes and slice specifications to use when
saving the tensors.
data: `N` tensors to save.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("Restore")
    .Input("file_pattern: string")
    .Input("tensor_name: string")
    .Output("tensor: dt")
    .Attr("dt: type")
    .Attr("preferred_shard: int = -1")
<<<<<<< HEAD
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      c->set_output(0, c->UnknownShape());
      return Status::OK();
    });
=======
    .Doc(R"doc(
Restores a tensor from checkpoint files.

Reads a tensor stored in one or several files. If there are several files (for
instance because a tensor was saved as slices), `file_pattern` may contain
wildcard symbols (`*` and `?`) in the filename portion only, not in the
directory portion.

If a `file_pattern` matches several files, `preferred_shard` can be used to hint
in which file the requested tensor is likely to be found. This op will first
open the file at index `preferred_shard` in the list of matching files and try
to restore tensors from that file.  Only if some tensors or tensor slices are
not found in that first file, then the Op opens all the files. Setting
`preferred_shard` to match the value passed as the `shard` input
of a matching `Save` Op may speed up Restore.  This attribute only affects
performance, not correctness.  The default value -1 means files are processed in
order.

See also `RestoreSlice`.

file_pattern: Must have a single element. The pattern of the files from
  which we read the tensor.
tensor_name: Must have a single element. The name of the tensor to be
  restored.
tensor: The restored tensor.
dt: The type of the tensor to be restored.
preferred_shard: Index of file to open first if multiple files match
  `file_pattern`.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("RestoreSlice")
    .Input("file_pattern: string")
    .Input("tensor_name: string")
    .Input("shape_and_slice: string")
    .Output("tensor: dt")
    .Attr("dt: type")
    .Attr("preferred_shard: int = -1")
<<<<<<< HEAD
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));

      // Attempt to infer output shapes from its shape_and_slice input.
      const Tensor* shape_and_slices_tensor = c->input_tensor(2);
      if (shape_and_slices_tensor) {
        const auto& shape_and_slice =
            shape_and_slices_tensor->flat<tstring>()(0);
        if (shape_and_slice.empty()) {
          c->set_output(0, c->UnknownShape());
        } else {
          TensorShape parsed_full_shape;
          TensorSlice parsed_slice;
          TensorShape parsed_slice_shape;
          TF_RETURN_IF_ERROR(checkpoint::ParseShapeAndSlice(
              shape_and_slice, &parsed_full_shape, &parsed_slice,
              &parsed_slice_shape));
          ShapeHandle shape_handle;
          TF_RETURN_IF_ERROR(
              c->MakeShapeFromTensorShape(parsed_slice_shape, &shape_handle));
          c->set_output(0, shape_handle);
        }
      } else {
        c->set_output(0, c->UnknownShape());
      }
      return Status::OK();
    });
=======
    .Doc(R"doc(
Restores a tensor from checkpoint files.

This is like `Restore` except that restored tensor can be listed as filling
only a slice of a larger tensor.  `shape_and_slice` specifies the shape of the
larger tensor and the slice that the restored tensor covers.

The `shape_and_slice` input has the same format as the
elements of the `shapes_and_slices` input of the `SaveSlices` op.

file_pattern: Must have a single element. The pattern of the files from
  which we read the tensor.
tensor_name: Must have a single element. The name of the tensor to be
  restored.
shape_and_slice: Scalar. The shapes and slice specifications to use when
  restoring a tensors.
tensor: The restored tensor.
dt: The type of the tensor to be restored.
preferred_shard: Index of file to open first if multiple files match
  `file_pattern`. See the documentation for `Restore`.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("ShardedFilename")
    .Input("basename: string")
    .Input("shard: int32")
    .Input("num_shards: int32")
    .Output("filename: string")
<<<<<<< HEAD
    .SetShapeFn(ScalarInputsAndOutputs);
=======
    .Doc(R"doc(
Generate a sharded filename. The filename is printf formated as
   %s-%05d-of-%05d, basename, shard, num_shards.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("ShardedFilespec")
    .Input("basename: string")
    .Input("num_shards: int32")
    .Output("filename: string")
<<<<<<< HEAD
    .SetShapeFn(ScalarInputsAndOutputs);
=======
    .Doc(R"doc(
Generate a glob pattern matching all sharded file names.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

// Reader source ops ----------------------------------------------------------

REGISTER_OP("WholeFileReader")
    .Output("reader_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
<<<<<<< HEAD
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("WholeFileReaderV2")
    .Output("reader_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);
=======
    .Doc(R"doc(
A Reader that outputs the entire contents of a file as a value.

To use, enqueue filenames in a Queue.  The output of ReaderRead will
be a filename (key) and the contents of that file (value).

reader_handle: The handle to reference the Reader.
container: If non-empty, this reader is placed in the given container.
        Otherwise, a default container is used.
shared_name: If non-empty, this reader is named in the given bucket
             with this shared_name. Otherwise, the node name is used instead.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("TextLineReader")
    .Output("reader_handle: Ref(string)")
    .Attr("skip_header_lines: int = 0")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
<<<<<<< HEAD
    .SetShapeFn(TwoElementOutput)
    .Deprecated(26, "Use TextLineReaderV2");

REGISTER_OP("TextLineReaderV2")
    .Output("reader_handle: resource")
    .Attr("skip_header_lines: int = 0")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);
=======
    .Doc(R"doc(
A Reader that outputs the lines of a file delimited by '\n'.

reader_handle: The handle to reference the Reader.
skip_header_lines: Number of lines to skip from the beginning of every file.
container: If non-empty, this reader is placed in the given container.
        Otherwise, a default container is used.
shared_name: If non-empty, this reader is named in the given bucket
             with this shared_name. Otherwise, the node name is used instead.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("FixedLengthRecordReader")
    .Output("reader_handle: Ref(string)")
    .Attr("header_bytes: int = 0")
    .Attr("record_bytes: int")
    .Attr("footer_bytes: int = 0")
<<<<<<< HEAD
    .Attr("hop_bytes: int = 0")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput)
    .Deprecated(26, "Use FixedLengthRecordReaderV2");

REGISTER_OP("FixedLengthRecordReaderV2")
    .Output("reader_handle: resource")
    .Attr("header_bytes: int = 0")
    .Attr("record_bytes: int")
    .Attr("footer_bytes: int = 0")
    .Attr("hop_bytes: int = 0")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("encoding: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);
=======
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .Doc(R"doc(
A Reader that outputs fixed-length records from a file.

reader_handle: The handle to reference the Reader.
container: If non-empty, this reader is placed in the given container.
        Otherwise, a default container is used.
shared_name: If non-empty, this reader is named in the given bucket
             with this shared_name. Otherwise, the node name is used instead.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("TFRecordReader")
    .Output("reader_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
<<<<<<< HEAD
    .Attr("compression_type: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput)
    .Deprecated(26, "Use TFRecordReaderV2");

REGISTER_OP("TFRecordReaderV2")
    .Output("reader_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("compression_type: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("LMDBReader")
    .Output("reader_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);
=======
    .SetIsStateful()
    .Doc(R"doc(
A Reader that outputs the records from a TensorFlow Records file.

reader_handle: The handle to reference the Reader.
container: If non-empty, this reader is placed in the given container.
        Otherwise, a default container is used.
shared_name: If non-empty, this reader is named in the given bucket
             with this shared_name. Otherwise, the node name is used instead.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("IdentityReader")
    .Output("reader_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
<<<<<<< HEAD
    .SetShapeFn(TwoElementOutput)
    .Deprecated(26, "Use IdentityReaderV2");

REGISTER_OP("IdentityReaderV2")
    .Output("reader_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);
=======
    .Doc(R"doc(
A Reader that outputs the queued work as both the key and value.

To use, enqueue strings in a Queue.  ReaderRead will take the front
work string and output (work, work).

reader_handle: The handle to reference the Reader.
container: If non-empty, this reader is placed in the given container.
        Otherwise, a default container is used.
shared_name: If non-empty, this reader is named in the given bucket
             with this shared_name. Otherwise, the node name is used instead.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

// Ops that operate on Readers ------------------------------------------------

REGISTER_OP("ReaderRead")
    .Input("reader_handle: Ref(string)")
    .Input("queue_handle: Ref(string)")
    .Output("key: string")
    .Output("value: string")
<<<<<<< HEAD
    .SetShapeFn(TwoElementVectorAndScalarOutputs);

REGISTER_OP("ReaderReadV2")
    .Input("reader_handle: resource")
    .Input("queue_handle: resource")
    .Output("key: string")
    .Output("value: string")
    .SetShapeFn(ScalarInputsAndOutputs);

REGISTER_OP("ReaderReadUpTo")
    .Input("reader_handle: Ref(string)")
    .Input("queue_handle: Ref(string)")
    .Input("num_records: int64")
    .Output("keys: string")
    .Output("values: string")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      ShapeHandle out = c->Vector(InferenceContext::kUnknownDim);
      c->set_output(0, out);
      c->set_output(1, out);
      return Status::OK();
    });

REGISTER_OP("ReaderReadUpToV2")
    .Input("reader_handle: resource")
    .Input("queue_handle: resource")
    .Input("num_records: int64")
    .Output("keys: string")
    .Output("values: string")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      ShapeHandle out = c->Vector(InferenceContext::kUnknownDim);
      c->set_output(0, out);
      c->set_output(1, out);
      return Status::OK();
    });
=======
    .Doc(R"doc(
Returns the next record (key, value pair) produced by a Reader.

Will dequeue from the input queue if necessary (e.g. when the
Reader needs to start reading from a new file since it has finished
with the previous file).

reader_handle: Handle to a Reader.
queue_handle: Handle to a Queue, with string work items.
key: A scalar.
value: A scalar.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("ReaderNumRecordsProduced")
    .Input("reader_handle: Ref(string)")
    .Output("records_produced: int64")
<<<<<<< HEAD
    .SetShapeFn(TwoElementVectorAndScalarOutputs);

REGISTER_OP("ReaderNumRecordsProducedV2")
    .Input("reader_handle: resource")
    .Output("records_produced: int64")
    .SetShapeFn(ScalarInputsAndOutputs);
=======
    .Doc(R"doc(
Returns the number of records this Reader has produced.

This is the same as the number of ReaderRead executions that have
succeeded.

reader_handle: Handle to a Reader.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("ReaderNumWorkUnitsCompleted")
    .Input("reader_handle: Ref(string)")
    .Output("units_completed: int64")
<<<<<<< HEAD
    .SetShapeFn(TwoElementVectorAndScalarOutputs);

REGISTER_OP("ReaderNumWorkUnitsCompletedV2")
    .Input("reader_handle: resource")
    .Output("units_completed: int64")
    .SetShapeFn(ScalarInputsAndOutputs);
=======
    .Doc(R"doc(
Returns the number of work units this Reader has finished processing.

reader_handle: Handle to a Reader.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("ReaderSerializeState")
    .Input("reader_handle: Ref(string)")
    .Output("state: string")
<<<<<<< HEAD
    .SetShapeFn(TwoElementVectorAndScalarOutputs);

REGISTER_OP("ReaderSerializeStateV2")
    .Input("reader_handle: resource")
    .Output("state: string")
    .SetShapeFn(ScalarInputsAndOutputs);
=======
    .Doc(R"doc(
Produce a string tensor that encodes the state of a Reader.

Not all Readers support being serialized, so this can produce an
Unimplemented error.

reader_handle: Handle to a Reader.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("ReaderRestoreState")
    .Input("reader_handle: Ref(string)")
    .Input("state: string")
<<<<<<< HEAD
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
      DimensionHandle unused_handle;
      TF_RETURN_IF_ERROR(
          c->WithValue(c->Dim(c->input(0), 0), 2, &unused_handle));

      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return Status::OK();
    });

REGISTER_OP("ReaderRestoreStateV2")
    .Input("reader_handle: resource")
    .Input("state: string")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return Status::OK();
    });

REGISTER_OP("ReaderReset")
    .Input("reader_handle: Ref(string)")
    .SetShapeFn(TwoElementVectorAndScalarOutputs);

REGISTER_OP("ReaderResetV2")
    .Input("reader_handle: resource")
    .SetShapeFn(ScalarInputsAndOutputs);
=======
    .Doc(R"doc(
Restore a reader to a previously saved state.

Not all Readers support being restored, so this can produce an
Unimplemented error.

reader_handle: Handle to a Reader.
state: Result of a ReaderSerializeState of a Reader with type
  matching reader_handle.
)doc");

REGISTER_OP("ReaderReset")
    .Input("reader_handle: Ref(string)")
    .Doc(R"doc(
Restore a Reader to its initial clean state.

reader_handle: Handle to a Reader.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

// Other input Ops ----------------------------------------------------------

REGISTER_OP("ReadFile")
    .Input("filename: string")
    .Output("contents: string")
<<<<<<< HEAD
    .SetShapeFn(ScalarInputsAndOutputs);

REGISTER_OP("WriteFile")
    .Input("filename: string")
    .Input("contents: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return Status::OK();
    });
=======
    .Doc(R"doc(
Reads and outputs the entire contents of the input filename.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

REGISTER_OP("MatchingFiles")
    .Input("pattern: string")
    .Output("filenames: string")
<<<<<<< HEAD
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 1, &unused));
      c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
      return Status::OK();
    });
=======
    .Doc(R"doc(
Returns the set of files matching a pattern.

Note that this routine only supports wildcard characters in the
basename portion of the pattern, not in the directory portion.

pattern: A (scalar) shell wildcard pattern.
filenames: A vector of matching filenames.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

}  // namespace tensorflow
