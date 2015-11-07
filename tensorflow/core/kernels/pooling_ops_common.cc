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

#include "tensorflow/core/kernels/pooling_ops_common.h"

#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cudnn/cudnn.h"
#endif  // GOOGLE_CUDA
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/gpu_utils.h"
#if TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#endif
#include "tensorflow/core/kernels/pooling_ops_common_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {

namespace {

template <typename T>
struct RawType {
  using type = T;
};

template <>
struct RawType<qint8> {
  using type = int8;
};

}  // namespace

PoolParameters::PoolParameters(OpKernelContext* context,
                               const std::vector<int32>& ksize,
                               const std::vector<int32>& stride,
                               Padding padding, TensorFormat data_format,
                               const TensorShape& tensor_in_shape) {
  // For maxpooling, tensor_in should have 2 spatial dimensions.
  // Note: the total number of dimensions could be 4 for NHWC, NCHW,
  // or 5 for NCHW_VECT_C.
  OP_REQUIRES(context,
              GetTensorSpatialDims(tensor_in_shape.dims(), data_format) == 2,
              errors::InvalidArgument(
                  "tensor_in_shape must have 2 spatial dimensions. ",
                  tensor_in_shape.dims(), " ", data_format));

  this->data_format = data_format;
  depth = GetTensorDim(tensor_in_shape, data_format, 'C') *
          (data_format == FORMAT_NCHW_VECT_C ? 4 : 1);
  tensor_in_cols = GetTensorDim(tensor_in_shape, data_format, 'W');
  tensor_in_rows = GetTensorDim(tensor_in_shape, data_format, 'H');
  tensor_in_batch = GetTensorDim(tensor_in_shape, data_format, 'N');
  window_rows = GetTensorDim(ksize, data_format, 'H');
  window_cols = GetTensorDim(ksize, data_format, 'W');
  depth_window = GetTensorDim(ksize, data_format, 'C');
  row_stride = GetTensorDim(stride, data_format, 'H');
  col_stride = GetTensorDim(stride, data_format, 'W');
  depth_stride = GetTensorDim(stride, data_format, 'C');
=======
#include "tensorflow/core/kernels/pooling_ops_common.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/public/tensor.h"

#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/maxpooling_op_gpu.h"
#include "tensorflow/core/kernels/pooling_ops_common_gpu.h"
#include "tensorflow/stream_executor/dnn.h"
#include "tensorflow/stream_executor/stream.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

PoolParameters::PoolParameters(OpKernelContext* context,
                               const std::vector<int32>& ksize,
                               const std::vector<int32>& stride,
                               Padding padding,
                               const TensorShape& tensor_in_shape) {
  // For maxpooling, tensor_in should have 4 dimensions.
  OP_REQUIRES(context, tensor_in_shape.dims() == 4,
              errors::InvalidArgument("tensor_in must be 4-dimensional"));

  depth = tensor_in_shape.dim_size(3);
  tensor_in_cols = tensor_in_shape.dim_size(2);
  tensor_in_rows = tensor_in_shape.dim_size(1);
  tensor_in_batch = tensor_in_shape.dim_size(0);
  window_rows = ksize[1];
  window_cols = ksize[2];
  depth_window = ksize[3];
  row_stride = stride[1];
  col_stride = stride[2];
  depth_stride = stride[3];
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  // We only support 2D pooling across width/height and depthwise
  // pooling, not a combination.
  OP_REQUIRES(context,
              (depth_window == 1 || (window_rows == 1 && window_cols == 1)),
              errors::Unimplemented(
                  "MaxPooling supports exactly one of pooling across depth "
                  "or pooling across width/height."));

  if (depth_window == 1) {
<<<<<<< HEAD
    OP_REQUIRES_OK(
        context, GetWindowedOutputSize(tensor_in_rows, window_rows, row_stride,
                                       padding, &out_height, &pad_rows));
    OP_REQUIRES_OK(
        context, GetWindowedOutputSize(tensor_in_cols, window_cols, col_stride,
                                       padding, &out_width, &pad_cols));
    pad_depth = 0;
    out_depth = depth;
=======
    OP_REQUIRES_OK(context, Get2dOutputSize(
                                tensor_in_rows, tensor_in_cols, window_rows,
                                window_cols, row_stride, col_stride, padding,
                                &out_height, &out_width, &pad_rows, &pad_cols));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  } else {
    // Our current version of depthwise max pooling does not support
    // any padding, and expects the depth_window to equal the
    // depth_stride (no overlapping).
    OP_REQUIRES(
        context, depth % depth_window == 0,
        errors::Unimplemented("Depthwise max pooling requires the depth "
                              "window to evenly divide the input depth"));
    OP_REQUIRES(
        context, depth_stride == depth_window,
        errors::Unimplemented("Depthwise max pooling requires the depth "
                              "window to equal the depth stride"));

    // The current version of depthwise max is only implemented on CPU.
    OP_REQUIRES(context,
                (DeviceType(static_cast<Device*>(context->device())
                                ->attributes()
                                .device_type()) == DeviceType(DEVICE_CPU)),
                errors::Unimplemented("Depthwise max pooling is currently "
                                      "only implemented for CPU devices."));

    pad_depth = 0;
    out_depth = depth / depth_window;
  }
}

TensorShape PoolParameters::forward_output_shape() {
  if (depth_window == 1) {
    // Spatial pooling
<<<<<<< HEAD
    return ShapeFromFormat(data_format, tensor_in_batch, out_height, out_width,
                           depth);
=======
    return TensorShape({tensor_in_batch, out_height, out_width, depth});
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  } else {
    // Depthwise pooling
    return TensorShape(
        {tensor_in_batch, tensor_in_rows, tensor_in_cols, out_depth});
  }
}

<<<<<<< HEAD
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename T>
void DnnPoolingOp<T>::Compute(OpKernelContext* context,
                              se::dnn::PoolingMode pooling_mode,
                              const std::vector<int32>& size,
                              const std::vector<int32>& stride, Padding padding,
                              TensorFormat data_format, const Tensor& tensor_in,
                              const TensorShape& tensor_out_shape,
                              bool propagate_nans) {
  Tensor* tensor_out = nullptr;
  OP_REQUIRES_OK(context,
                 context->allocate_output(0, tensor_out_shape, &tensor_out));
  if (tensor_in.shape().num_elements() == 0) {
    return;
  }

  PoolParameters params{context, size,        stride,
                        padding, data_format, tensor_in.shape()};
  if (!context->status().ok()) {
    return;
  }

  int batch_size = params.tensor_in_batch;
  int depth = params.depth;
#if CUDNN_VERSION < 7300
  /// Earlier versions do not support NHWC format, so we need to convert it
  /// to NCHW before calling cudnn. We need to get rid of this once it is done
  Tensor transformed_input;
  if (data_format == FORMAT_NHWC) {
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<T>::value,
                                ShapeFromFormat(FORMAT_NCHW, tensor_in.shape(),
                                                data_format),
                                &transformed_input));
    functor::NHWCToNCHW<GPUDevice, T, 4>()(context->eigen_device<Device>(),
                                           tensor_in.tensor<T, 4>(),
                                           transformed_input.tensor<T, 4>());
  } else {
    transformed_input = tensor_in;
  }
  Tensor transformed_output;
  if (data_format == FORMAT_NHWC) {
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<T>::value,
                                ShapeFromFormat(FORMAT_NCHW, tensor_out_shape,
                                                data_format),
                                &transformed_output));
  } else {
    transformed_output = *tensor_out;
  }
  se::dnn::DataLayout data_layout = se::dnn::DataLayout::kBatchDepthYX;
#else
  auto& transformed_input = tensor_in;
  auto& transformed_output = *tensor_out;
  se::dnn::DataLayout data_layout;
  switch (data_format) {
    case FORMAT_NHWC:
      data_layout = se::dnn::DataLayout::kBatchYXDepth;
      break;
    case FORMAT_NCHW:
      data_layout = se::dnn::DataLayout::kBatchDepthYX;
      break;
    case FORMAT_NCHW_VECT_C:
      // NCHW_VECT_C is not supported by cudnnPoolingForward(), but can be
      // emulated via NHWC.
      data_layout = se::dnn::DataLayout::kBatchYXDepth;
      batch_size *= depth / 4;
      depth = 4;
      break;
    default:
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("Unsupported format: ",
                                          ToString(data_format)));
  }
#endif
  /// Get ready to call cudnn
  se::dnn::PoolingDescriptor pooling_desc;
  pooling_desc.set_pooling_mode(pooling_mode)
      .set_window_height(params.window_rows)
      .set_window_width(params.window_cols)
      .set_vertical_stride(params.row_stride)
      .set_horizontal_stride(params.col_stride)
      .set_vertical_padding(params.pad_rows)
      .set_horizontal_padding(params.pad_cols)
      .set_propagate_nans(propagate_nans);

  se::dnn::BatchDescriptor input_desc;
  input_desc.set_count(batch_size)
      .set_height(params.tensor_in_rows)
      .set_width(params.tensor_in_cols)
      .set_feature_map_count(depth)
      .set_layout(data_layout);

  se::dnn::BatchDescriptor output_desc;
  output_desc.set_count(batch_size)
      .set_height(params.out_height)
      .set_width(params.out_width)
      .set_feature_map_count(depth)
      .set_layout(data_layout);

  auto input_data =
      AsDeviceMemory(reinterpret_cast<const typename RawType<T>::type*>(
                         transformed_input.template flat<T>().data()),
                     transformed_input.template flat<T>().size());

  auto output_data =
      AsDeviceMemory(reinterpret_cast<const typename RawType<T>::type*>(
                         transformed_output.template flat<T>().data()),
                     transformed_output.template flat<T>().size());

  auto* stream = context->op_device_context()->stream();
  OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

#if TENSORFLOW_USE_ROCM
  static int64 PoolingScratchSize = GetDnnWorkspaceLimit(
      // default value is in bytes despite the name of the environment variable
      "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32  // 4GB
  );

  DnnScratchAllocator scratch_allocator(PoolingScratchSize, context);
  bool status =
      stream
          ->ThenPoolForward(pooling_desc, input_desc, input_data, output_desc,
                            &output_data, &scratch_allocator)
          .ok();
#else
  bool status = stream
                    ->ThenPoolForward(pooling_desc, input_desc, input_data,
                                      output_desc, &output_data)
                    .ok();
#endif
  OP_REQUIRES(context, status,
              errors::Internal("dnn PoolForward launch failed"));
#if CUDNN_VERSION < 7300
  if (data_format == FORMAT_NHWC) {
    /// Transform the output data from NCHW back to NHWC
    auto toConstTensor = [](const Tensor& x) -> const Tensor { return x; };
    using RT = typename RawType<T>::type;
    functor::NCHWToNHWC<GPUDevice, RT, 4>()(
        context->eigen_device<Device>(),
        toConstTensor(transformed_output).template tensor<RT, 4>(),
        tensor_out->tensor<RT, 4>());
  }
#endif
}

template <typename T>
void DnnPoolingGradOp<T>::Compute(
    OpKernelContext* context, se::dnn::PoolingMode pooling_mode,
    const std::vector<int32>& size, const std::vector<int32>& stride,
    Padding padding, TensorFormat data_format, const Tensor* tensor_in,
    const Tensor* tensor_out, const Tensor& out_backprop,
    const TensorShape& tensor_in_shape, bool propagate_nans) {
  CHECK((pooling_mode != se::dnn::PoolingMode::kMaximum) ||
=======
#ifdef GOOGLE_CUDA

namespace {
template <typename T>
perftools::gputools::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory,
                                                    uint64 size) {
  perftools::gputools::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory),
                                                size * sizeof(T));
  perftools::gputools::DeviceMemory<T> typed(wrapped);
  return typed;
}
}  // namespace

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                      \
  template <>                                                    \
  void TransformDepth<GPUDevice, T>::operator()(                 \
      const GPUDevice& d, typename TTypes<T, 4>::ConstTensor in, \
      const Eigen::DSizes<Eigen::DenseIndex, 4>& shuffle,        \
      typename TTypes<T, 4>::Tensor out);                        \
  extern template struct TransformDepth<GPUDevice, T>;

DECLARE_GPU_SPEC(float);
#undef DECLARE_GPU_SPEC
}  // namespace functor

template <typename T>
void DnnPoolingGradOp<T>::Compute(
    OpKernelContext* context,
    perftools::gputools::dnn::PoolingMode pooling_mode,
    const std::vector<int32>& size, const std::vector<int32>& stride,
    Padding padding, const Tensor* tensor_in, const Tensor* tensor_out,
    const Tensor& out_backprop, const TensorShape& tensor_in_shape) {
  CHECK((pooling_mode == perftools::gputools::dnn::PoolingMode::kMaximum) ||
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
        (tensor_in && tensor_out))
      << "For MaxPoolGrad, both tensor_in and tensor_out needs to be "
         "specified";

<<<<<<< HEAD
  Tensor* input_backprop = nullptr;
  OP_REQUIRES_OK(context,
                 context->allocate_output(0, tensor_in_shape, &input_backprop));
  if (tensor_in_shape.num_elements() == 0) {
    return;
  }

  PoolParameters params{context, size,        stride,
                        padding, data_format, tensor_in_shape};
=======
  Tensor* output = nullptr;
  OP_REQUIRES_OK(context,
                 context->allocate_output(0, tensor_in_shape, &output));

  PoolParameters params{context, size, stride, padding, tensor_in_shape};
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  if (!context->status().ok()) {
    return;
  }

<<<<<<< HEAD
#if CUDNN_VERSION < 7300
  /// For now, cudnn does not support NHWC format, so we need to convert it
  /// to NCHW before calling cudnn. We need to get rid of this once it is done
  Tensor transformed_input;
  TensorShape transformed_input_shape;
  if (data_format == FORMAT_NHWC || !tensor_in) {
    transformed_input_shape =
        ShapeFromFormat(FORMAT_NCHW, tensor_in_shape, data_format);
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   transformed_input_shape,
                                                   &transformed_input));
  } else {
    transformed_input = *tensor_in;
  }
  Tensor transformed_output;
  TensorShape transformed_output_shape;
  if (data_format == FORMAT_NHWC || !tensor_out) {
    transformed_output_shape =
        ShapeFromFormat(FORMAT_NCHW, out_backprop.shape(), data_format);
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   transformed_output_shape,
                                                   &transformed_output));
  } else {
    transformed_output = *tensor_out;
  }
  Tensor transformed_input_backprop;
  if (data_format == FORMAT_NHWC) {
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<T>::value,
                                          transformed_input_shape,
                                          &transformed_input_backprop));
  } else {
    transformed_input_backprop = *input_backprop;
  }
  Tensor transformed_output_backprop;
  if (data_format == FORMAT_NHWC) {
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<T>::value,
                                          transformed_output_shape,
                                          &transformed_output_backprop));
  } else {
    transformed_output_backprop = out_backprop;
  }

  if (data_format == FORMAT_NHWC) {
    /// Convert the data from NHWC to NCHW if necessary.
    if (tensor_in) {
      // For AvgPoolGrad, the original input tensor is not necessary. However,
      // cudnn still requires them to run, although they do not affect the
      // results.
      functor::NHWCToNCHW<GPUDevice, T, 4>()(context->eigen_device<Device>(),
                                             tensor_in->tensor<T, 4>(),
                                             transformed_input.tensor<T, 4>());
    }
    if (tensor_out) {
      // For AvgPoolGrad, the original output tensor is not necessary. However,
      // cudnn still requires them to run, although they do not affect the
      // results.
      functor::NHWCToNCHW<GPUDevice, T, 4>()(context->eigen_device<Device>(),
                                             tensor_out->tensor<T, 4>(),
                                             transformed_output.tensor<T, 4>());
    }
    functor::NHWCToNCHW<GPUDevice, T, 4>()(
        context->eigen_device<Device>(), out_backprop.tensor<T, 4>(),
        transformed_output_backprop.tensor<T, 4>());
  }
  se::dnn::DataLayout data_layout = se::dnn::DataLayout::kBatchDepthYX;
#else
  Tensor transformed_input;
  if (!tensor_in) {
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<T>::value,
                                          tensor_in_shape, &transformed_input));
  } else {
    transformed_input = *tensor_in;
  }
  Tensor transformed_output;
  if (!tensor_out) {
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   out_backprop.shape(),
                                                   &transformed_output));
  } else {
    transformed_output = *tensor_out;
  }
  Tensor transformed_input_backprop = *input_backprop;
  Tensor transformed_output_backprop = out_backprop;
  se::dnn::DataLayout data_layout;
  switch (data_format) {
    case FORMAT_NHWC:
      data_layout = se::dnn::DataLayout::kBatchYXDepth;
      break;
    case FORMAT_NCHW:
      data_layout = se::dnn::DataLayout::kBatchDepthYX;
      break;
    default:
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("Unsupported format: ",
                                          ToString(data_format)));
  }
#endif  // CUDNN_VERSION < 7300

  /// Get ready to call cudnn
  se::dnn::PoolingDescriptor pooling_desc;
=======
  /// For now, cudnn does not support NHWC format, so we need to convert it
  /// to NCHW before calling cudnn. We need to get rid of this once it is done
  Tensor transformed_input;
  OP_REQUIRES_OK(context, context->allocate_temp(
                              DataTypeToEnum<T>::value,
                              TensorShape({tensor_in_shape.dim_size(0),
                                           tensor_in_shape.dim_size(3),
                                           tensor_in_shape.dim_size(1),
                                           tensor_in_shape.dim_size(2)}),
                              &transformed_input));
  Tensor transformed_input_backprop;
  OP_REQUIRES_OK(context, context->allocate_temp(
                              DataTypeToEnum<T>::value,
                              TensorShape({tensor_in_shape.dim_size(0),
                                           tensor_in_shape.dim_size(3),
                                           tensor_in_shape.dim_size(1),
                                           tensor_in_shape.dim_size(2)}),
                              &transformed_input_backprop));
  Tensor transformed_output;
  OP_REQUIRES_OK(
      context,
      context->allocate_temp(
          DataTypeToEnum<T>::value,
          TensorShape({out_backprop.dim_size(0), out_backprop.dim_size(3),
                       out_backprop.dim_size(1), out_backprop.dim_size(2)}),
          &transformed_output));
  Tensor transformed_output_backprop;
  OP_REQUIRES_OK(
      context,
      context->allocate_temp(
          DataTypeToEnum<T>::value,
          TensorShape({out_backprop.dim_size(0), out_backprop.dim_size(3),
                       out_backprop.dim_size(1), out_backprop.dim_size(2)}),
          &transformed_output_backprop));

  auto nhwc_to_nchw = Eigen::DSizes<Eigen::DenseIndex, 4>(0, 3, 1, 2);
  if (tensor_in) {
    // For AvgPoolGrad, the original input tensor is not necessary. However,
    // cudnn still requires them to run, although they do not affect the
    // results.
    functor::TransformDepth<GPUDevice, T>()(
        context->eigen_device<Device>(), tensor_in->tensor<T, 4>(),
        nhwc_to_nchw, transformed_input.tensor<T, 4>());
  }
  if (tensor_out) {
    // For AvgPoolGrad, the original output tensor is not necessary. However,
    // cudnn still requires them to run, although they do not affect the
    // results.
    functor::TransformDepth<GPUDevice, T>()(
        context->eigen_device<Device>(), tensor_out->tensor<T, 4>(),
        nhwc_to_nchw, transformed_output.tensor<T, 4>());
  }
  functor::TransformDepth<GPUDevice, T>()(
      context->eigen_device<Device>(), out_backprop.tensor<T, 4>(),
      nhwc_to_nchw, transformed_output_backprop.tensor<T, 4>());

  /// Get ready to call cudnn
  perftools::gputools::dnn::PoolingDescriptor pooling_desc;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  pooling_desc.set_pooling_mode(pooling_mode)
      .set_window_height(params.window_rows)
      .set_window_width(params.window_cols)
      .set_vertical_stride(params.row_stride)
      .set_horizontal_stride(params.col_stride)
      .set_vertical_padding(params.pad_rows)
<<<<<<< HEAD
      .set_horizontal_padding(params.pad_cols)
      .set_propagate_nans(propagate_nans);

  se::dnn::BatchDescriptor orig_output_desc;
=======
      .set_horizontal_padding(params.pad_cols);

  perftools::gputools::dnn::BatchDescriptor orig_output_desc;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  orig_output_desc.set_count(params.tensor_in_batch)
      .set_height(params.out_height)
      .set_width(params.out_width)
      .set_feature_map_count(params.depth)
<<<<<<< HEAD
      .set_layout(data_layout);

  se::dnn::BatchDescriptor orig_input_desc;
=======
      .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);

  perftools::gputools::dnn::BatchDescriptor orig_input_desc;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  orig_input_desc.set_count(params.tensor_in_batch)
      .set_height(params.tensor_in_rows)
      .set_width(params.tensor_in_cols)
      .set_feature_map_count(params.depth)
<<<<<<< HEAD
      .set_layout(data_layout);
=======
      .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  auto orig_output_data =
      AsDeviceMemory(transformed_output.template flat<T>().data(),
                     transformed_output.template flat<T>().size());
  auto orig_input_data =
      AsDeviceMemory(transformed_input.template flat<T>().data(),
                     transformed_input.template flat<T>().size());
<<<<<<< HEAD
  auto output_backprop_data =
      AsDeviceMemory(transformed_output_backprop.template flat<T>().data(),
                     transformed_output_backprop.template flat<T>().size());
  auto input_backprop_data =
      AsDeviceMemory(transformed_input_backprop.template flat<T>().data(),
                     transformed_input_backprop.template flat<T>().size());

  auto* stream = context->op_device_context()->stream();
  OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

#if TENSORFLOW_USE_ROCM
  static int64 PoolingScratchSize = GetDnnWorkspaceLimit(
      // default value is in bytes despite the name of the environment variable
      "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32  // 4GB
  );

  DnnScratchAllocator scratch_allocator(PoolingScratchSize, context);
  bool status = stream
                    ->ThenPoolBackward(pooling_desc, orig_input_desc,
                                       orig_input_data, orig_output_desc,
                                       orig_output_data, output_backprop_data,
                                       &input_backprop_data, &scratch_allocator)
                    .ok();
#else
  bool status =
      stream
          ->ThenPoolBackward(pooling_desc, orig_input_desc, orig_input_data,
                             orig_output_desc, orig_output_data,
                             output_backprop_data, &input_backprop_data)
          .ok();
#endif

  OP_REQUIRES(context, status,
              errors::Internal("dnn PoolBackward launch failed"));

#if CUDNN_VERSION < 7300
  if (data_format == FORMAT_NHWC) {
    /// Transform the output data from NCHW back to NHWC.
    auto toConstTensor = [](const Tensor& x) -> const Tensor { return x; };
    functor::NCHWToNHWC<GPUDevice, T, 4>()(
        context->eigen_device<Device>(),
        toConstTensor(transformed_input_backprop).template tensor<T, 4>(),
        input_backprop->tensor<T, 4>());
  }
#endif  // CUDNN_VERSION < 7300
}

#define DEFINE_DNN_OPS(T)         \
  template class DnnPoolingOp<T>; \
  template class DnnPoolingGradOp<T>;
TF_CALL_GPU_NUMBER_TYPES(DEFINE_DNN_OPS)

#if CUDNN_VERSION >= 7300
template class DnnPoolingOp<qint8>;
#endif

#undef DEFINE_DNN_OPS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
=======
  auto output_backprop =
      AsDeviceMemory(transformed_output_backprop.template flat<T>().data(),
                     transformed_output_backprop.template flat<T>().size());
  auto input_backprop =
      AsDeviceMemory(transformed_input_backprop.template flat<T>().data(),
                     transformed_input_backprop.template flat<T>().size());

  auto* stream = context->op_device_context<GPUDeviceContext>()->stream();
  OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

  bool status =
      stream->ThenPoolBackward(pooling_desc, orig_input_desc, orig_input_data,
                               orig_output_desc, orig_output_data,
                               output_backprop, &input_backprop)
          .ok();
  OP_REQUIRES(context, status,
              errors::Internal("cudnn PoolBackward launch failed"));

  /// Transform the output data from NCHW back to NHWC
  auto toConstTensor = [](const Tensor& x) -> const Tensor { return x; };
  auto nchw_to_nhwc = Eigen::DSizes<Eigen::DenseIndex, 4>(0, 2, 3, 1);
  functor::TransformDepth<GPUDevice, T>()(
      context->eigen_device<Device>(),
      toConstTensor(transformed_input_backprop).template tensor<T, 4>(),
      nchw_to_nhwc, output->tensor<T, 4>());
}

template class DnnPoolingGradOp<float>;

#endif  // GOOGLE_CUDA
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

}  // namespace tensorflow
