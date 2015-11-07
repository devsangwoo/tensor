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

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {

// Sets output[0] to shape [batch_dim,height,width,channel_dim], where
// height and width come from the size_tensor.
Status SetOutputToSizedImage(InferenceContext* c, DimensionHandle batch_dim,
                             int size_input_idx, DimensionHandle channel_dim) {
  // Verify shape of size input.
  ShapeHandle size;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(size_input_idx), 1, &size));
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(c->WithValue(c->Dim(size, 0), 2, &unused));

  // Get size values from the size tensor.
  const Tensor* size_tensor = c->input_tensor(size_input_idx);
  DimensionHandle width;
  DimensionHandle height;
  if (size_tensor == nullptr) {
    width = c->UnknownDim();
    height = c->UnknownDim();
  } else {
    // TODO(petewarden) - Remove once we have constant evaluation in C++ only.
    if (size_tensor->dtype() != DT_INT32) {
      return errors::InvalidArgument(
          "Bad size input type for SetOutputToSizedImage: Expected DT_INT32 "
          "but got ",
          DataTypeString(size_tensor->dtype()), " for input #", size_input_idx,
          " in ", c->DebugString());
    }
    auto vec = size_tensor->vec<int32>();
    height = c->MakeDim(vec(0));
    width = c->MakeDim(vec(1));
  }
  c->set_output(0, c->MakeShape({batch_dim, height, width, channel_dim}));
  return Status::OK();
}

Status ResizeShapeFn(InferenceContext* c) {
  ShapeHandle input;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));
  return SetOutputToSizedImage(c, c->Dim(input, 0), 1 /* size_input_idx */,
                               c->Dim(input, 3));
}

Status DecodeImageShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
  DimensionHandle channels_dim;
  int32 channels;
  TF_RETURN_IF_ERROR(c->GetAttr("channels", &channels));
  if (channels == 0) {
    channels_dim = c->UnknownDim();
  } else {
    if (channels < 0) {
      return errors::InvalidArgument("channels must be non-negative, got ",
                                     channels);
    }
    channels_dim = c->MakeDim(channels);
  }

  c->set_output(0, c->MakeShape({InferenceContext::kUnknownDim,
                                 InferenceContext::kUnknownDim, channels_dim}));
  return Status::OK();
}

Status EncodeImageShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &unused));
  c->set_output(0, c->Scalar());
  return Status::OK();
}

Status ColorspaceShapeFn(InferenceContext* c) {
  ShapeHandle input;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &input));

  // The last dimension value is always 3.
  DimensionHandle last_dim;
  TF_RETURN_IF_ERROR(c->WithValue(c->Dim(input, -1), 3, &last_dim));
  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->ReplaceDim(input, -1, last_dim, &out));
  c->set_output(0, out);

  return Status::OK();
}

Status NMSShapeFn(InferenceContext* c) {
  // Get inputs and validate ranks.
  ShapeHandle boxes;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &boxes));
  ShapeHandle scores;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &scores));
  ShapeHandle max_output_size;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &max_output_size));
  ShapeHandle iou_threshold;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &iou_threshold));
  ShapeHandle score_threshold;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &score_threshold));
  // The boxes is a 2-D float Tensor of shape [num_boxes, 4].
  DimensionHandle unused;
  // The boxes[0] and scores[0] are both num_boxes.
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(boxes, 0), c->Dim(scores, 0), &unused));
  // The boxes[1] is 4.
  TF_RETURN_IF_ERROR(c->WithValue(c->Dim(boxes, 1), 4, &unused));

  c->set_output(0, c->Vector(c->UnknownDim()));
  return Status::OK();
}

Status SoftNMSShapeFn(InferenceContext* c) {
  // Get inputs and validate ranks.
  ShapeHandle boxes;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &boxes));
  ShapeHandle scores;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &scores));
  ShapeHandle max_output_size;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &max_output_size));
  ShapeHandle iou_threshold;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &iou_threshold));
  ShapeHandle score_threshold;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &score_threshold));
  ShapeHandle soft_nms_sigma;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &soft_nms_sigma));
  // The boxes is a 2-D float Tensor of shape [num_boxes, 4].
  DimensionHandle unused;
  // The boxes[0] and scores[0] are both num_boxes.
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(boxes, 0), c->Dim(scores, 0), &unused));
  // The boxes[1] is 4.
  TF_RETURN_IF_ERROR(c->WithValue(c->Dim(boxes, 1), 4, &unused));

  c->set_output(0, c->Vector(c->UnknownDim()));
  c->set_output(1, c->Vector(c->UnknownDim()));
  return Status::OK();
}

Status CombinedNMSShapeFn(InferenceContext* c) {
  // Get inputs and validate ranks
  ShapeHandle boxes;
  // boxes is a tensor of Dimensions [batch_size, num_anchors, q, 4]
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &boxes));
  ShapeHandle scores;
  // scores is a tensor of Dimensions [batch_size, num_anchors, num_classes]
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &scores));
  ShapeHandle max_output_size_per_class;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &max_output_size_per_class));
  ShapeHandle max_total_size;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &max_total_size));
  ShapeHandle unused_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused_shape));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused_shape));

  DimensionHandle unused;
  // boxes[0] and scores[0] are both batch_size
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(boxes, 0), c->Dim(scores, 0), &unused));
  // boxes[1] and scores[1] are both num_anchors
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(boxes, 1), c->Dim(scores, 1), &unused));
  // The boxes[3] is 4.
  TF_RETURN_IF_ERROR(c->WithValue(c->Dim(boxes, 3), 4, &unused));

  DimensionHandle d = c->Dim(boxes, 2);
  DimensionHandle class_dim = c->Dim(scores, 2);
  if (c->ValueKnown(d) && c->ValueKnown(class_dim)) {
    if (c->Value(d) != 1 && c->Value(d) != c->Value(class_dim)) {
      return errors::InvalidArgument(
          "third dimension of boxes must be either "
          "1 or equal to the third dimension of scores");
    }
  }
  DimensionHandle output_dim;
  DimensionHandle batch_dim = c->Dim(boxes, 0);

  TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(3, &output_dim));
  if (c->ValueKnown(output_dim) && c->Value(output_dim) <= 0) {
    return errors::InvalidArgument("max_total_size should be > 0 ");
  }
  DimensionHandle size_per_class;
  TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(2, &size_per_class));

  int64 output_size;
  bool pad_per_class;
  TF_RETURN_IF_ERROR(c->GetAttr("pad_per_class", &pad_per_class));
  if (!pad_per_class) {
    output_size = c->Value(output_dim);
  } else {
    if (c->ValueKnown(size_per_class) && c->Value(size_per_class) <= 0) {
      return errors::InvalidArgument(
          "max_output_size_per_class must be > 0 "
          "if pad_per_class is set to true ");
    }
    output_size = std::min(c->Value(output_dim),
                           c->Value(size_per_class) * c->Value(class_dim));
  }
  c->set_output(0, c->MakeShape({batch_dim, output_size, 4}));
  c->set_output(1, c->MakeShape({batch_dim, output_size}));
  c->set_output(2, c->MakeShape({batch_dim, output_size}));
  c->set_output(3, c->Vector(batch_dim));
  return Status::OK();
}

}  // namespace

=======
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
// --------------------------------------------------------------------------
REGISTER_OP("ResizeArea")
    .Input("images: T")
    .Input("size: int32")
    .Output("resized_images: float")
<<<<<<< HEAD
    .Attr("T: {int8, uint8, int16, uint16, int32, int64, half, float, double}")
    .Attr("align_corners: bool = false")
    .SetShapeFn(ResizeShapeFn);
=======
    .Attr("T: {uint8, int8, int32, float, double}")
    .Doc(R"doc(
Resize `images` to `size` using area interpolation.

Input images can be of different types but output images are always float.

images: 4-D with shape `[batch, height, width, channels]`.
size:= A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
  new size for the images.
resized_images:  4-D with shape
  `[batch, new_height, new_width, channels]`.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

// --------------------------------------------------------------------------
REGISTER_OP("ResizeBicubic")
    .Input("images: T")
    .Input("size: int32")
    .Output("resized_images: float")
<<<<<<< HEAD
    .Attr("T: {int8, uint8, int16, uint16, int32, int64, half, float, double}")
    .Attr("align_corners: bool = false")
    .Attr("half_pixel_centers: bool = false")
    .SetShapeFn(ResizeShapeFn);

// --------------------------------------------------------------------------
REGISTER_OP("ResizeBicubicGrad")
    .Input("grads: float")
    .Input("original_image: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Attr("align_corners: bool = false")
    .Attr("half_pixel_centers: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    });
=======
    .Attr("T: {uint8, int8, int32, float, double}")
    .Doc(R"doc(
Resize `images` to `size` using bicubic interpolation.

Input images can be of different types but output images are always float.

images: 4-D with shape `[batch, height, width, channels]`.
size:= A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
  new size for the images.
resized_images:  4-D with shape
  `[batch, new_height, new_width, channels]`.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

// --------------------------------------------------------------------------
REGISTER_OP("ResizeBilinear")
    .Input("images: T")
    .Input("size: int32")
    .Output("resized_images: float")
<<<<<<< HEAD
    .Attr(
        "T: {int8, uint8, int16, uint16, int32, int64, bfloat16, half, "
        "float, double}")
    .Attr("align_corners: bool = false")
    .Attr("half_pixel_centers: bool = false")
    .SetShapeFn(ResizeShapeFn);

// --------------------------------------------------------------------------
REGISTER_OP("ScaleAndTranslate")
    .Input("images: T")
    .Input("size: int32")
    .Input("scale: float")
    .Input("translation: float")
    .Output("resized_images: float")
    .Attr(
        "T: {int8, uint8, int16, uint16, int32, int64, bfloat16, half, "
        "float, double}")
    .Attr("kernel_type: string = 'lanczos3'")
    .Attr("antialias: bool = true")
    .SetShapeFn(ResizeShapeFn);

// --------------------------------------------------------------------------
REGISTER_OP("QuantizedResizeBilinear")
    .Input("images: T")
    .Input("size: int32")
    .Input("min: float")
    .Input("max: float")
    .Output("resized_images: T")
    .Output("out_min: float")
    .Output("out_max: float")
    .Attr("T: {quint8, qint32, float}")
    .Attr("align_corners: bool = false")
    .Attr("half_pixel_centers: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(ResizeShapeFn(c));
      ShapeHandle min_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &min_shape));
      ShapeHandle max_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &max_shape));
      c->set_output(1, c->MakeShape({}));
      c->set_output(2, c->MakeShape({}));
      return Status::OK();
    });

// --------------------------------------------------------------------------
REGISTER_OP("ResizeBilinearGrad")
    .Input("grads: float")
    .Input("original_image: T")
    .Output("output: T")
    .Attr("T: {float, bfloat16, half, double}")
    .Attr("align_corners: bool = false")
    .Attr("half_pixel_centers: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    });

// --------------------------------------------------------------------------
REGISTER_OP("ScaleAndTranslateGrad")
    .Input("grads: T")
    .Input("original_image: T")
    .Input("scale: float")
    .Input("translation: float")
    .Output("output: T")
    .Attr("T: {float}")
    .Attr("kernel_type: string = 'lanczos3'")
    .Attr("antialias: bool = true")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    });
=======
    .Attr("T: {uint8, int8, int32, float, double}")
    .Doc(R"doc(
Resize `images` to `size` using bilinear interpolation.

Input images can be of different types but output images are always float.

images: 4-D with shape `[batch, height, width, channels]`.
size:= A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
  new size for the images.
resized_images:  4-D with shape
  `[batch, new_height, new_width, channels]`.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

// --------------------------------------------------------------------------
REGISTER_OP("ResizeNearestNeighbor")
    .Input("images: T")
    .Input("size: int32")
    .Output("resized_images: T")
<<<<<<< HEAD
    .Attr("T: {int8, uint8, int16, uint16, int32, int64, half, float, double}")
    .Attr("align_corners: bool = false")
    .Attr("half_pixel_centers: bool = false")
    .SetShapeFn(ResizeShapeFn);

// --------------------------------------------------------------------------
REGISTER_OP("ResizeNearestNeighborGrad")
    .Input("grads: T")
    .Input("size: int32")
    .Output("output: T")
    .Attr("T: {uint8, int8, int32, half, float, double}")
    .Attr("align_corners: bool = false")
    .Attr("half_pixel_centers: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));
      ShapeHandle unused;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(unused, 0), 2, &unused_dim));
      const Tensor* size = c->input_tensor(1);
      if (size == nullptr) {
        TF_RETURN_IF_ERROR(c->ReplaceDim(input, 1, c->UnknownDim(), &input));
        TF_RETURN_IF_ERROR(c->ReplaceDim(input, 2, c->UnknownDim(), &input));
      } else {
        auto size_vec = size->vec<int32>();
        TF_RETURN_IF_ERROR(
            c->ReplaceDim(input, 1, c->MakeDim(size_vec(0)), &input));
        TF_RETURN_IF_ERROR(
            c->ReplaceDim(input, 2, c->MakeDim(size_vec(1)), &input));
      }
      c->set_output(0, input);
      return Status::OK();
    });
=======
    .Attr("T: {uint8, int8, int32, float, double}")
    .Doc(R"doc(
Resize `images` to `size` using nearest neighbor interpolation.

Input images can be of different types but output images are always float.

images: 4-D with shape `[batch, height, width, channels]`.
size:= A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
  new size for the images.
resized_images:  4-D with shape
  `[batch, new_height, new_width, channels]`.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

// --------------------------------------------------------------------------
REGISTER_OP("RandomCrop")
    .Input("image: T")
    .Input("size: int64")
    .Output("output: T")
    .Attr("T: {uint8, int8, int16, int32, int64, float, double}")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .SetIsStateful()
<<<<<<< HEAD
    .Deprecated(8, "Random crop is now pure Python")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle image;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &image));
      DimensionHandle channels = c->Dim(image, -1);

      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->Merge(c->input(1), c->Vector(2), &unused));

      const Tensor* size = c->input_tensor(1);
      DimensionHandle h;
      DimensionHandle w;
      if (size == nullptr) {
        h = c->UnknownDim();
        w = c->UnknownDim();
      } else {
        auto size_vec = size->vec<int64>();
        h = c->MakeDim(size_vec(0));
        w = c->MakeDim(size_vec(1));
      }
      c->set_output(0, c->MakeShape({h, w, channels}));
      return Status::OK();
    });
=======
    .Doc(R"doc(
Randomly crop `image`.

`size` is a 1-D int64 tensor with 2 elements representing the crop height and
width.  The values must be non negative.

This Op picks a random location in `image` and crops a `height` by `width`
rectangle from that location.  The random location is picked so the cropped
area will fit inside the original image.

image: 3-D of shape `[height, width, channels]`.
size: 1-D of length 2 containing: `crop_height`, `crop_width`..
seed: If either seed or seed2 are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: An second seed to avoid seed collision.
output: 3-D of shape `[crop_height, crop_width, channels].`
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
// TODO(shlens): Support variable rank in RandomCrop.

// --------------------------------------------------------------------------
REGISTER_OP("DecodeJpeg")
    .Input("contents: string")
    .Attr("channels: int = 0")
    .Attr("ratio: int = 1")
    .Attr("fancy_upscaling: bool = true")
    .Attr("try_recover_truncated: bool = false")
    .Attr("acceptable_fraction: float = 1.0")
<<<<<<< HEAD
    .Attr("dct_method: string = ''")
    .Output("image: uint8")
    .SetShapeFn(DecodeImageShapeFn);

// --------------------------------------------------------------------------
REGISTER_OP("DecodeAndCropJpeg")
    .Input("contents: string")
    .Input("crop_window: int32")
    .Attr("channels: int = 0")
    .Attr("ratio: int = 1")
    .Attr("fancy_upscaling: bool = true")
    .Attr("try_recover_truncated: bool = false")
    .Attr("acceptable_fraction: float = 1.0")
    .Attr("dct_method: string = ''")
    .Output("image: uint8")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      DimensionHandle channels_dim = c->UnknownDim();
      DimensionHandle h = c->UnknownDim();
      DimensionHandle w = c->UnknownDim();

      int32 channels;
      TF_RETURN_IF_ERROR(c->GetAttr("channels", &channels));
      if (channels != 0) {
        if (channels < 0) {
          return errors::InvalidArgument("channels must be non-negative, got ",
                                         channels);
        }
        channels_dim = c->MakeDim(channels);
      }

      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(unused, 0), 4, &unused_dim));

      const Tensor* crop_window = c->input_tensor(1);
      if (crop_window != nullptr) {
        auto crop_window_vec = crop_window->vec<int32>();
        h = c->MakeDim(crop_window_vec(2));
        w = c->MakeDim(crop_window_vec(3));
      }
      c->set_output(0, c->MakeShape({h, w, channels_dim}));
      return Status::OK();
    });
=======
    .Output("image: uint8")
    .Doc(R"doc(
Decode a JPEG-encoded image to a uint8 tensor.

The attr `channels` indicates the desired number of color channels for the
decoded image.

Accepted values are:

*   0: Use the number of channels in the JPEG-encoded image.
*   1: output a grayscale image.
*   3: output an RGB image.

If needed, the JPEG-encoded image is transformed to match the requested number
of color channels.

The attr `ratio` allows downscaling the image by an integer factor during
decoding.  Allowed values are: 1, 2, 4, and 8.  This is much faster than
downscaling the image later.

contents: 0-D.  The JPEG-encoded image.
channels: Number of color channels for the decoded image.
ratio: Downscaling ratio.
fancy_upscaling: If true use a slower but nicer upscaling of the
  chroma planes (yuv420/422 only).
try_recover_truncated:  If true try to recover an image from truncated input.
acceptable_fraction: The minimum required fraction of lines before a truncated
  input is accepted.
image: 3-D with shape `[height, width, channels]`..
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

// --------------------------------------------------------------------------
REGISTER_OP("EncodeJpeg")
    .Input("image: uint8")
    .Attr("format: {'', 'grayscale', 'rgb'} = ''")
    .Attr("quality: int = 95")
    .Attr("progressive: bool = false")
    .Attr("optimize_size: bool = false")
    .Attr("chroma_downsampling: bool = true")
    .Attr("density_unit: {'in', 'cm'} = 'in'")
    .Attr("x_density: int = 300")
    .Attr("y_density: int = 300")
    .Attr("xmp_metadata: string = ''")
    .Output("contents: string")
<<<<<<< HEAD
    .SetShapeFn(EncodeImageShapeFn);

// --------------------------------------------------------------------------
REGISTER_OP("EncodeJpegVariableQuality")
    .Input("images: uint8")
    .Input("quality: int32")
    .Output("contents: string")
    .SetShapeFn(EncodeImageShapeFn);

// --------------------------------------------------------------------------
REGISTER_OP("ExtractJpegShape")
    .Input("contents: string")
    .Output("image_shape: output_type")
    .Attr("output_type: {int32, int64} = DT_INT32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      c->set_output(0, c->Vector(3));
      return Status::OK();
    });
=======
    .Doc(R"doc(
JPEG-encode an image.

`image` is a 3-D uint8 Tensor of shape `[height, width, channels]`.

The attr `format` can be used to override the color format of the encoded
output.  Values can be:

*   `''`: Use a default format based on the number of channels in the image.
*   `grayscale`: Output a grayscale JPEG image.  The `channels` dimension
    of `image` must be 1.
*   `rgb`: Output an RGB JPEG image. The `channels` dimension
    of `image` must be 3.

If `format` is not specified or is the empty string, a default format is picked
in function of the number of channels in `image`:

*   1: Output a grayscale image.
*   3: Output an RGB image.

image: 3-D with shape `[height, width, channels]`.
format: Per pixel image format.
quality: Quality of the compression from 0 to 100 (higher is better and slower).
progressive: If True, create a JPEG that loads progressively (coarse to fine).
optimize_size: If True, spend CPU/RAM to reduce size with no quality change.
chroma_downsampling: See http://en.wikipedia.org/wiki/Chroma_subsampling.
density_unit: Unit used to specify `x_density` and `y_density`:
   pixels per inch (`'in'`) or centimeter (`'cm'`).
x_density: Horizontal pixels per density unit.
y_density: Vertical pixels per density unit.
xmp_metadata: If not empty, embed this XMP metadata in the image header.
contents: 0-D. JPEG-encoded image.
)doc");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

// --------------------------------------------------------------------------
REGISTER_OP("AdjustContrast")
    .Input("images: T")
    .Input("contrast_factor: float")
    .Input("min_value: float")
    .Input("max_value: float")
    .Output("output: float")
    .Attr("T: {uint8, int8, int16, int32, int64, float, double}")
<<<<<<< HEAD
    .Deprecated(2, "Use AdjustContrastv2 instead")
    .SetShapeFn([](InferenceContext* c) {
      // The contrast_factor, min_value, max_value should be scalar only.
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 3);
    });

// --------------------------------------------------------------------------
REGISTER_OP("AdjustContrastv2")
    .Input("images: T")
    .Input("contrast_factor: float")
    .Output("output: T")
    .Attr("T: {half, float} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      // The contrast_factor should be scalar only.
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 3);
    });

// --------------------------------------------------------------------------
REGISTER_OP("AdjustHue")
    .Input("images: T")
    .Input("delta: float")
    .Output("output: T")
    .Attr("T: {half, float} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 3);
    });

// --------------------------------------------------------------------------
REGISTER_OP("AdjustSaturation")
    .Input("images: T")
    .Input("scale: float")
    .Output("output: T")
    .Attr("T: {half, float} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 3);
    });

// --------------------------------------------------------------------------
REGISTER_OP("DecodePng")
    .Input("contents: string")
    .Attr("channels: int = 0")
    .Attr("dtype: {uint8, uint16} = DT_UINT8")
    .Output("image: dtype")
    .SetShapeFn(DecodeImageShapeFn);

// --------------------------------------------------------------------------
REGISTER_OP("EncodePng")
    .Attr("compression: int = -1")
    .Attr("T: {uint8, uint16} = DT_UINT8")
    .Input("image: T")
    .Output("contents: string")
    .SetShapeFn(EncodeImageShapeFn);

// --------------------------------------------------------------------------
REGISTER_OP("DecodeBmp")
    .Input("contents: string")
    .Output("image: uint8")
    .Attr("channels: int = 0")
    .SetShapeFn(DecodeImageShapeFn);

// --------------------------------------------------------------------------
REGISTER_OP("DecodeGif")
    .Input("contents: string")
    .Output("image: uint8")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      c->set_output(0, c->MakeShape({InferenceContext::kUnknownDim,
                                     InferenceContext::kUnknownDim,
                                     InferenceContext::kUnknownDim, 3}));
      return Status::OK();
    });

// --------------------------------------------------------------------------
REGISTER_OP("RGBToHSV")
    .Input("images: T")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, double} = DT_FLOAT")
    .SetShapeFn(ColorspaceShapeFn);

// --------------------------------------------------------------------------
REGISTER_OP("HSVToRGB")
    .Input("images: T")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, double} = DT_FLOAT")
    .SetShapeFn(ColorspaceShapeFn);

// --------------------------------------------------------------------------
REGISTER_OP("DrawBoundingBoxes")
    .Input("images: T")
    .Input("boxes: float")
    .Output("output: T")
    .Attr("T: {float, half} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      // The rank of images should be 4.
      ShapeHandle images;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &images));
      // Channel depth should be either 1 (GRY), 3 (RGB), or 4 (RGBA).
      if (c->ValueKnown(c->Dim(images, 3))) {
        int64 depth = c->Value(c->Dim(images, 3));
        if (!(depth == 1 || depth == 3 || depth == 4)) {
          return errors::InvalidArgument(
              "Channel depth should be either 1 (GRY), "
              "3 (RGB), or 4 (RGBA)");
        }
      }

      // The rank of boxes is 3: [batch, num_bounding_boxes, 4].
      ShapeHandle boxes;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &boxes));
      // The last value of boxes shape is 4.
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(boxes, 2), 4, &unused));

      // The rank of the input image (rank = 4) has already been restricted
      // above, and the output is of the same shape as the input.
      return shape_inference::UnchangedShape(c);
    });

// --------------------------------------------------------------------------
REGISTER_OP("DrawBoundingBoxesV2")
    .Input("images: T")
    .Input("boxes: float")
    .Input("colors: float")
    .Output("output: T")
    .Attr("T: {float, half} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 3);
    });

// --------------------------------------------------------------------------
REGISTER_OP("SampleDistortedBoundingBox")
    .Input("image_size: T")
    .Input("bounding_boxes: float")
    .Output("begin: T")
    .Output("size: T")
    .Output("bboxes: float")
    .Attr("T: {uint8, int8, int16, int32, int64}")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("min_object_covered: float = 0.1")
    .Attr("aspect_ratio_range: list(float) = [0.75, 1.33]")
    .Attr("area_range: list(float) = [0.05, 1.0]")
    .Attr("max_attempts: int = 100")
    .Attr("use_image_if_no_bounding_boxes: bool = false")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      // Get inputs and validate ranks.
      ShapeHandle image_size;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &image_size));
      ShapeHandle bounding_boxes;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &bounding_boxes));
      // image_size: 1-D with [height, width, channels]
      // bounding_boxes: 3-D with shape [batch, N, 4]
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(image_size, 0), 3, &unused));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(bounding_boxes, 2), 4, &unused));

      c->set_output(0, c->Vector(3));
      c->set_output(1, c->Vector(3));
      c->set_output(2, c->MakeShape({1, 1, 4}));
      return Status::OK();
    });

REGISTER_OP("SampleDistortedBoundingBoxV2")
    .Input("image_size: T")
    .Input("bounding_boxes: float")
    .Input("min_object_covered: float")
    .Output("begin: T")
    .Output("size: T")
    .Output("bboxes: float")
    .Attr("T: {uint8, int8, int16, int32, int64}")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("aspect_ratio_range: list(float) = [0.75, 1.33]")
    .Attr("area_range: list(float) = [0.05, 1.0]")
    .Attr("max_attempts: int = 100")
    .Attr("use_image_if_no_bounding_boxes: bool = false")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      // Get inputs and validate ranks.
      ShapeHandle image_size;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &image_size));
      ShapeHandle bounding_boxes;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &bounding_boxes));
      ShapeHandle min_object_covered;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &min_object_covered));
      // image_size: 1-D with [height, width, channels]
      // bounding_boxes: 3-D with shape [batch, N, 4]
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(image_size, 0), 3, &unused));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(bounding_boxes, 2), 4, &unused));

      c->set_output(0, c->Vector(3));
      c->set_output(1, c->Vector(3));
      c->set_output(2, c->MakeShape({1, 1, 4}));
      return Status::OK();
    });

// --------------------------------------------------------------------------

// glimpse = extract_glimpse(input, size, offsets) extract the glimpse
// of size `size` centered at location `offsets` from the input tensor
// `input`.
//
// REQUIRES: input.dims() == 4
//
REGISTER_OP("ExtractGlimpse")
    .Input("input: float")
    .Input("size: int32")
    .Input("offsets: float")
    .Output("glimpse: float")
    .Attr("centered: bool = true")
    .Attr("normalized: bool = true")
    .Attr("uniform_noise: bool = true")
    .Attr("noise: string = 'uniform'")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));
      ShapeHandle offsets;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &offsets));

      DimensionHandle batch_dim;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(input, 0), c->Dim(offsets, 0), &batch_dim));
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(offsets, 1), 2, &unused));

      bool uniform_noise = false;
      TF_RETURN_IF_ERROR(c->GetAttr("uniform_noise", &uniform_noise));
      string noise;
      TF_RETURN_IF_ERROR(c->GetAttr("noise", &noise));
      if (uniform_noise && (!noise.empty() && noise != "uniform")) {
        return errors::InvalidArgument(
            "The uniform_noise and noise should not be specified at the same "
            "time");
      }

      return SetOutputToSizedImage(c, batch_dim, 1 /* size_input_idx */,
                                   c->Dim(input, 3));
    });

// --------------------------------------------------------------------------

REGISTER_OP("CropAndResize")
    .Input("image: T")
    .Input("boxes: float")
    .Input("box_ind: int32")
    .Input("crop_size: int32")
    .Output("crops: float")
    .Attr("T: {uint8, uint16, int8, int16, int32, int64, half, float, double}")
    .Attr("method: {'bilinear', 'nearest'} = 'bilinear'")
    .Attr("extrapolation_value: float = 0")
    .SetShapeFn([](InferenceContext* c) {
      // Get inputs and validate ranks.
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));
      ShapeHandle boxes;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &boxes));
      ShapeHandle box_ind;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &box_ind));

      // boxes[0] and box_ind[0] are both num_boxes.
      DimensionHandle num_boxes_dim;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(boxes, 0), c->Dim(box_ind, 0), &num_boxes_dim));

      // boxes.dim(1) is 4.
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(boxes, 1), 4, &unused));

      return SetOutputToSizedImage(c, num_boxes_dim, 3 /* size_input_idx */,
                                   c->Dim(input, 3));
    });

REGISTER_OP("CropAndResizeGradImage")
    .Input("grads: float")
    .Input("boxes: float")
    .Input("box_ind: int32")
    .Input("image_size: int32")
    .Output("output: T")
    .Attr("T: {float, half, double}")
    .Attr("method: {'bilinear', 'nearest'} = 'bilinear'")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(3, &out));
      TF_RETURN_IF_ERROR(c->WithRank(out, 4, &out));
      c->set_output(0, out);
      return Status::OK();
    });

REGISTER_OP("CropAndResizeGradBoxes")
    .Input("grads: float")
    .Input("image: T")
    .Input("boxes: float")
    .Input("box_ind: int32")
    .Output("output: float")
    .Attr("T: {uint8, uint16, int8, int16, int32, int64, half, float, double}")
    .Attr("method: {'bilinear'} = 'bilinear'")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(2));
      return Status::OK();
    });

// --------------------------------------------------------------------------

REGISTER_OP("NonMaxSuppression")
    .Input("boxes: float")
    .Input("scores: float")
    .Input("max_output_size: int32")
    .Output("selected_indices: int32")
    .Attr("iou_threshold: float = 0.5")
    .SetShapeFn([](InferenceContext* c) {
      // Get inputs and validate ranks.
      ShapeHandle boxes;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &boxes));
      ShapeHandle scores;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &scores));
      ShapeHandle max_output_size;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &max_output_size));
      // The boxes is a 2-D float Tensor of shape [num_boxes, 4].
      DimensionHandle unused;
      // The boxes[0] and scores[0] are both num_boxes.
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(boxes, 0), c->Dim(scores, 0), &unused));
      // The boxes[1] is 4.
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(boxes, 1), 4, &unused));

      c->set_output(0, c->Vector(c->UnknownDim()));
      return Status::OK();
    });

REGISTER_OP("NonMaxSuppressionV2")
    .Input("boxes: T")
    .Input("scores: T")
    .Input("max_output_size: int32")
    .Input("iou_threshold: T_threshold")
    .Output("selected_indices: int32")
    .Attr("T: {half, float} = DT_FLOAT")
    .Attr("T_threshold: {half, float} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      // Get inputs and validate ranks.
      ShapeHandle boxes;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &boxes));
      ShapeHandle scores;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &scores));
      ShapeHandle max_output_size;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &max_output_size));
      ShapeHandle iou_threshold;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &iou_threshold));
      // The boxes is a 2-D float Tensor of shape [num_boxes, 4].
      DimensionHandle unused;
      // The boxes[0] and scores[0] are both num_boxes.
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(boxes, 0), c->Dim(scores, 0), &unused));
      // The boxes[1] is 4.
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(boxes, 1), 4, &unused));

      c->set_output(0, c->Vector(c->UnknownDim()));
      return Status::OK();
    });

REGISTER_OP("NonMaxSuppressionV3")
    .Input("boxes: T")
    .Input("scores: T")
    .Input("max_output_size: int32")
    .Input("iou_threshold: T_threshold")
    .Input("score_threshold: T_threshold")
    .Output("selected_indices: int32")
    .Attr("T: {half, float} = DT_FLOAT")
    .Attr("T_threshold: {half, float} = DT_FLOAT")
    .SetShapeFn(NMSShapeFn);

REGISTER_OP("NonMaxSuppressionV4")
    .Input("boxes: T")
    .Input("scores: T")
    .Input("max_output_size: int32")
    .Input("iou_threshold: T_threshold")
    .Input("score_threshold: T_threshold")
    .Output("selected_indices: int32")
    .Output("valid_outputs: int32")
    .Attr("T: {half, float} = DT_FLOAT")
    .Attr("T_threshold: {half, float} = DT_FLOAT")
    .Attr("pad_to_max_output_size: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(NMSShapeFn(c));

      bool pad_to_max;
      TF_RETURN_IF_ERROR(c->GetAttr("pad_to_max_output_size", &pad_to_max));
      if (pad_to_max) {
        // If padded, overwrite the shape of the output to be static.
        DimensionHandle output_dim;
        TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(2, &output_dim));
        c->set_output(0, c->MakeShape({output_dim}));
      }
      c->set_output(1, c->MakeShape({}));
      return Status::OK();
    });

REGISTER_OP("NonMaxSuppressionV5")
    .Input("boxes: T")
    .Input("scores: T")
    .Input("max_output_size: int32")
    .Input("iou_threshold: T")
    .Input("score_threshold: T")
    .Input("soft_nms_sigma: T")
    .Output("selected_indices: int32")
    .Output("selected_scores: T")
    .Output("valid_outputs: int32")
    .Attr("T: {half, float} = DT_FLOAT")
    .Attr("pad_to_max_output_size: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(SoftNMSShapeFn(c));

      bool pad_to_max;
      TF_RETURN_IF_ERROR(c->GetAttr("pad_to_max_output_size", &pad_to_max));
      if (pad_to_max) {
        // If padded, overwrite the shape of the output to be static.
        DimensionHandle output_dim;
        TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(2, &output_dim));
        c->set_output(0, c->MakeShape({output_dim}));
        c->set_output(1, c->MakeShape({output_dim}));
      }

      c->set_output(2, c->MakeShape({}));
      return Status::OK();
    });

REGISTER_OP("NonMaxSuppressionWithOverlaps")
    .Input("overlaps: float")
    .Input("scores: float")
    .Input("max_output_size: int32")
    .Input("overlap_threshold: float")
    .Input("score_threshold: float")
    .Output("selected_indices: int32")
    .SetShapeFn([](InferenceContext* c) {
      // Get inputs and validate ranks.
      ShapeHandle overlaps;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &overlaps));
      ShapeHandle scores;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &scores));
      ShapeHandle max_output_size;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &max_output_size));
      ShapeHandle overlap_threshold;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &overlap_threshold));
      ShapeHandle score_threshold;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &score_threshold));
      // The boxes is a 2-D float Tensor of shape [num_boxes, 4].
      DimensionHandle unused;
      // The boxes[0] and scores[0] are both num_boxes.
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(overlaps, 0), c->Dim(scores, 0), &unused));
      // The boxes[1] is 4.
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(overlaps, 0), c->Dim(overlaps, 1), &unused));

      c->set_output(0, c->Vector(c->UnknownDim()));
      return Status::OK();
    });

REGISTER_OP("CombinedNonMaxSuppression")
    .Input("boxes: float")
    .Input("scores: float")
    .Input("max_output_size_per_class: int32")
    .Input("max_total_size: int32")
    .Input("iou_threshold: float")
    .Input("score_threshold: float")
    .Output("nmsed_boxes: float")
    .Output("nmsed_scores: float")
    .Output("nmsed_classes: float")
    .Output("valid_detections: int32")
    .Attr("pad_per_class: bool = false")
    .Attr("clip_boxes: bool = true")
    .SetShapeFn(CombinedNMSShapeFn);

REGISTER_OP("GenerateBoundingBoxProposals")
    .Input("scores: float")
    .Input("bbox_deltas: float")
    .Input("image_info: float")
    .Input("anchors: float")
    .Input("nms_threshold: float")
    .Input("pre_nms_topn: int32")
    .Input("min_size: float")
    .Output("rois: float")
    .Output("roi_probabilities: float")
    .Attr("post_nms_topn: int = 300")
    .SetShapeFn([](InferenceContext* c) -> Status {
      // make sure input tensors have are correct rank
      ShapeHandle scores, images, bounding_boxes, anchors, nms_threshold,
          n_pre_nms, min_box_size;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &scores));  //(N, H, W, A)
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(1), 4, &bounding_boxes));         //(N,H,W,A4)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &images));  // (N,5)
      auto im_info = c->Dim(images, 1);
      TF_RETURN_IF_ERROR(c->WithValue(im_info, 5, &im_info));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 3, &anchors));  // (A4)
      // check scalar tensors
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &nms_threshold));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &n_pre_nms));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &min_box_size));

      // TODO(skama): verify that the inputs are compatible
      int post_nms_top_n;
      TF_RETURN_IF_ERROR(c->GetAttr("post_nms_topn", &post_nms_top_n));
      auto roi_shape = c->MakeShape(
          {c->Dim(scores, 0), post_nms_top_n, 4});  //(N,post_nms_top_n,4)
      auto prob_shape = c->MakeShape(
          {c->Dim(scores, 0), post_nms_top_n});  // (N,post_nms_top_n)
      c->set_output(0, roi_shape);
      c->set_output(1, prob_shape);
      return Status::OK();
    });
=======
    .Doc(R"Doc(
Adjust the contrast of one or more images.

`images` is a tensor of at least 3 dimensions.  The last 3 dimensions are
interpreted as `[height, width, channels]`.  The other dimensions only
represent a collection of images, such as `[batch, height, width, channels].`

Contrast is adjusted independently for each channel of each image.

For each channel, the Op first computes the mean of the image pixels in the
channel and then adjusts each component of each pixel to
`(x - mean) * contrast_factor + mean`.

These adjusted values are then clipped to fit in the `[min_value, max_value]`
interval.

`images: Images to adjust.  At least 3-D.
contrast_factor: A float multiplier for adjusting contrast.
min_value: Minimum value for clipping the adjusted pixels.
max_value: Maximum value for clipping the adjusted pixels.
output: The constrast-adjusted image or images.
)Doc");

// --------------------------------------------------------------------------
REGISTER_OP("DecodePng")
    .Input("contents: string")
    .Attr("channels: int = 0")
    .Output("image: uint8")
    .Doc(R"doc(
Decode a PNG-encoded image to a uint8 tensor.

The attr `channels` indicates the desired number of color channels for the
decoded image.

Accepted values are:

*   0: Use the number of channels in the PNG-encoded image.
*   1: output a grayscale image.
*   3: output an RGB image.
*   4: output an RGBA image.

If needed, the PNG-encoded image is transformed to match the requested number
of color channels.

contents: 0-D.  The PNG-encoded image.
channels: Number of color channels for the decoded image.
image: 3-D with shape `[height, width, channels]`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("EncodePng")
    .Input("image: uint8")
    .Attr("compression: int = -1")
    .Output("contents: string")
    .Doc(R"doc(
PNG-encode an image.

`image` is a 3-D uint8 Tensor of shape `[height, width, channels]` where
`channels` is:

*   1: for grayscale.
*   3: for RGB.
*   4: for RGBA.

The ZLIB compression level, `compression`, can be -1 for the PNG-encoder
default or a value from 0 to 9.  9 is the highest compression level, generating
the smallest output, but is slower.

image: 3-D with shape `[height, width, channels]`.
compression: Compression level.
contents: 0-D. PNG-encoded image.
)doc");

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}  // namespace tensorflow
