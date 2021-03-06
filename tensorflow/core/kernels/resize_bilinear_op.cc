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

// See docs in ../ops/image_ops.cc
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/resize_bilinear_op.h"

#include <memory>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/image_resizer_state.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
=======
// See docs in ../ops/image_ops.cc
#define EIGEN_USE_THREADS

#include <memory>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
<<<<<<< HEAD
typedef Eigen::GpuDevice GPUDevice;
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

template <typename Device, typename T>
class ResizeBilinearOp : public OpKernel {
 public:
<<<<<<< HEAD
  explicit ResizeBilinearOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
    OP_REQUIRES_OK(
        context, context->GetAttr("half_pixel_centers", &half_pixel_centers_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    ImageResizerState st(align_corners_, half_pixel_centers_);
    st.ValidateAndCreateOutput(context, input);

    if (!context->status().ok()) return;

    // Return if the output is empty.
    if (st.output->NumElements() == 0) return;

    typename TTypes<T, 4>::ConstTensor image_data(input.tensor<T, 4>());
    TTypes<float, 4>::Tensor output_data = st.output->tensor<float, 4>();

    functor::ResizeBilinear<Device, T>()(
        context->eigen_device<Device>(), image_data, st.height_scale,
        st.width_scale, half_pixel_centers_, output_data);
  }

 private:
  bool align_corners_;
  bool half_pixel_centers_;
};

namespace {
// Compute the interpolation indices only once.
struct CachedInterpolation {
  int64 lower;  // Lower source index used in the interpolation
  int64 upper;  // Upper source index used in the interpolation
  // 1-D linear iterpolation scale (see:
  // https://en.wikipedia.org/wiki/Bilinear_interpolation)
  float lerp;
};

template <typename Scaler>
inline void compute_interpolation_weights(const Scaler scaler,
                                          const int64 out_size,
                                          const int64 in_size,
                                          const float scale,
                                          CachedInterpolation* interpolation) {
  interpolation[out_size].lower = 0;
  interpolation[out_size].upper = 0;
  for (int64 i = out_size - 1; i >= 0; --i) {
    const float in = scaler(i, scale);
    const float in_f = std::floor(in);
    interpolation[i].lower =
        std::max(static_cast<int64>(in_f), static_cast<int64>(0));
    interpolation[i].upper =
        std::min(static_cast<int64>(std::ceil(in)), in_size - 1);
    interpolation[i].lerp = in - in_f;
  }
}

/**
 * Computes the bilinear interpolation from the appropriate 4 float points
 * and the linear interpolation weights.
 */
inline float compute_lerp(const float top_left, const float top_right,
                          const float bottom_left, const float bottom_right,
                          const float x_lerp, const float y_lerp) {
  const float top = top_left + (top_right - top_left) * x_lerp;
  const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
  return top + (bottom - top) * y_lerp;
}

template <typename T>
void resize_image(
    typename TTypes<T, 4>::ConstTensor images, const int batch_size,
    const int64 in_height, const int64 in_width, const int64 out_height,
    const int64 out_width, const int channels,
    const std::vector<CachedInterpolation>& xs,
    const std::vector<CachedInterpolation>& ys,
    typename TTypes<float, 4>::Tensor output) TF_ATTRIBUTE_NOINLINE;
template <typename T>
void resize_image(typename TTypes<T, 4>::ConstTensor images,
                  const int batch_size, const int64 in_height,
                  const int64 in_width, const int64 out_height,
                  const int64 out_width, const int channels,
                  const std::vector<CachedInterpolation>& xs_vec,
                  const std::vector<CachedInterpolation>& ys,
                  typename TTypes<float, 4>::Tensor output) {
  const int64 in_row_size = in_width * channels;
  const int64 in_batch_num_values = in_height * in_row_size;
  const int64 out_row_size = out_width * channels;

  const T* input_b_ptr = images.data();
  const CachedInterpolation* xs = xs_vec.data();

  if (channels == 3) {
    float* output_y_ptr = output.data();
    for (int b = 0; b < batch_size; ++b) {
      for (int64 y = 0; y < out_height; ++y) {
        const T* ys_input_lower_ptr = input_b_ptr + ys[y].lower * in_row_size;
        const T* ys_input_upper_ptr = input_b_ptr + ys[y].upper * in_row_size;
        const float ys_lerp = ys[y].lerp;
        for (int64 x = 0; x < out_width; ++x) {
          const int64 xs_lower = xs[x].lower;
          const int64 xs_upper = xs[x].upper;
          const float xs_lerp = xs[x].lerp;

          // Read channel 0.
          const float top_left0(ys_input_lower_ptr[xs_lower + 0]);
          const float top_right0(ys_input_lower_ptr[xs_upper + 0]);
          const float bottom_left0(ys_input_upper_ptr[xs_lower + 0]);
          const float bottom_right0(ys_input_upper_ptr[xs_upper + 0]);

          // Read channel 1.
          const float top_left1(ys_input_lower_ptr[xs_lower + 1]);
          const float top_right1(ys_input_lower_ptr[xs_upper + 1]);
          const float bottom_left1(ys_input_upper_ptr[xs_lower + 1]);
          const float bottom_right1(ys_input_upper_ptr[xs_upper + 1]);

          // Read channel 2.
          const float top_left2(ys_input_lower_ptr[xs_lower + 2]);
          const float top_right2(ys_input_lower_ptr[xs_upper + 2]);
          const float bottom_left2(ys_input_upper_ptr[xs_lower + 2]);
          const float bottom_right2(ys_input_upper_ptr[xs_upper + 2]);

          // Compute output.
          output_y_ptr[x * channels + 0] =
              compute_lerp(top_left0, top_right0, bottom_left0, bottom_right0,
                           xs_lerp, ys_lerp);
          output_y_ptr[x * channels + 1] =
              compute_lerp(top_left1, top_right1, bottom_left1, bottom_right1,
                           xs_lerp, ys_lerp);
          output_y_ptr[x * channels + 2] =
              compute_lerp(top_left2, top_right2, bottom_left2, bottom_right2,
                           xs_lerp, ys_lerp);
        }
        output_y_ptr += out_row_size;
      }
      input_b_ptr += in_batch_num_values;
    }
  } else {
    float* output_y_ptr = output.data();
    for (int b = 0; b < batch_size; ++b) {
      for (int64 y = 0; y < out_height; ++y) {
        const T* ys_input_lower_ptr = input_b_ptr + ys[y].lower * in_row_size;
        const T* ys_input_upper_ptr = input_b_ptr + ys[y].upper * in_row_size;
        const float ys_lerp = ys[y].lerp;
        for (int64 x = 0; x < out_width; ++x) {
          auto xs_lower = xs[x].lower;
          auto xs_upper = xs[x].upper;
          auto xs_lerp = xs[x].lerp;
          for (int c = 0; c < channels; ++c) {
            const float top_left(ys_input_lower_ptr[xs_lower + c]);
            const float top_right(ys_input_lower_ptr[xs_upper + c]);
            const float bottom_left(ys_input_upper_ptr[xs_lower + c]);
            const float bottom_right(ys_input_upper_ptr[xs_upper + c]);
            output_y_ptr[x * channels + c] =
                compute_lerp(top_left, top_right, bottom_left, bottom_right,
                             xs_lerp, ys_lerp);
          }
        }
        output_y_ptr += out_row_size;
      }
      input_b_ptr += in_batch_num_values;
    }
  }
}

}  // namespace

// Partial specialization of ResizeBilinear functor for a CPUDevice.
namespace functor {
template <typename T>
struct ResizeBilinear<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T, 4>::ConstTensor images,
                  const float height_scale, const float width_scale,
                  bool half_pixel_centers,
                  typename TTypes<float, 4>::Tensor output) {
    const int batch_size = images.dimension(0);
    const int64 in_height = images.dimension(1);
    const int64 in_width = images.dimension(2);
    const int channels = images.dimension(3);

    const int64 out_height = output.dimension(1);
    const int64 out_width = output.dimension(2);

    // Handle no-op resizes efficiently.
    if (out_height == in_height && out_width == in_width) {
      output = images.template cast<float>();
      return;
    }

    std::vector<CachedInterpolation> ys(out_height + 1);
    std::vector<CachedInterpolation> xs(out_width + 1);

    // Compute the cached interpolation weights on the x and y dimensions.
    if (half_pixel_centers) {
      compute_interpolation_weights(HalfPixelScaler(), out_height, in_height,
                                    height_scale, ys.data());
      compute_interpolation_weights(HalfPixelScaler(), out_width, in_width,
                                    width_scale, xs.data());

    } else {
      compute_interpolation_weights(LegacyScaler(), out_height, in_height,
                                    height_scale, ys.data());
      compute_interpolation_weights(LegacyScaler(), out_width, in_width,
                                    width_scale, xs.data());
    }
    // Scale x interpolation weights to avoid a multiplication during iteration.
    for (int i = 0; i < xs.size(); ++i) {
      xs[i].lower *= channels;
      xs[i].upper *= channels;
    }

    resize_image<T>(images, batch_size, in_height, in_width, out_height,
                    out_width, channels, xs, ys, output);
  }
};
}  // namespace functor

template <typename Device, typename T>
class ResizeBilinearOpGrad : public OpKernel {
 public:
  explicit ResizeBilinearOpGrad(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
    OP_REQUIRES_OK(
        context, context->GetAttr("half_pixel_centers", &half_pixel_centers_));
  }

  void Compute(OpKernelContext* context) override {
    // Validate input.
    // First argument is gradient with respect to resized image.
    const Tensor& input = context->input(0);
    const Tensor& original_image = context->input(1);

    ImageResizerGradientState st(align_corners_, half_pixel_centers_);
    st.ValidateAndCreateOutput(context, input, original_image);

    if (!context->status().ok()) return;

    TTypes<float, 4>::ConstTensor input_grad = input.tensor<float, 4>();
    typename TTypes<T, 4>::Tensor output_grad(st.output->tensor<T, 4>());

    functor::ResizeBilinearGrad<Device, T>()(
        context->eigen_device<Device>(), input_grad, st.height_scale,
        st.width_scale, half_pixel_centers_, output_grad);
  }

 private:
  bool align_corners_;
  bool half_pixel_centers_;
};

// Partial specialization of ResizeBilinearGrad functor for a CPUDevice.
namespace functor {

template <typename T>
struct ResizeBilinearGrad<CPUDevice, T> {
  template <typename Scaler>
  void ResizeGradCore(const Scaler& scaler,
                      typename TTypes<float, 4>::ConstTensor input_grad,
                      const float height_scale, const float width_scale,
                      typename TTypes<T, 4>::Tensor output_grad) {
    const Eigen::Index batch = output_grad.dimension(0);
    const Eigen::Index original_height = output_grad.dimension(1);
    const Eigen::Index original_width = output_grad.dimension(2);
    const Eigen::Index channels = output_grad.dimension(3);

    const Eigen::Index resized_height = input_grad.dimension(1);
    const Eigen::Index resized_width = input_grad.dimension(2);

    output_grad.setZero();

    // Each resized output pixel was computed as a weighted average of four
    // input pixels. Here we find the four input pixel locations that
    // contributed to each output pixel and propgate the gradient at the output
    // pixel location to each of those four input pixel locations in the same
    // proportions that they originally contributed to the output pixel.
    // Here is the forward-propagation pseudo-code, for reference:
    // resized(b, y, x, c) = top_left     * (1 - y) * (1 - x)
    //                     + top_right    * (1 - y) *      x
    //                     + bottom_left  *      y  * (1 - x)
    //                     + bottom_right *      y  *      x
    for (Eigen::Index b = 0; b < batch; ++b) {
      for (Eigen::Index y = 0; y < resized_height; ++y) {
        const float in_y = scaler(y, height_scale);
        const Eigen::Index top_y_index =
            std::max(static_cast<Eigen::Index>(floorf(in_y)),
                     static_cast<Eigen::Index>(0));
        const Eigen::Index bottom_y_index = std::min(
            static_cast<Eigen::Index>(ceilf(in_y)), original_height - 1);
        const float y_lerp = in_y - floorf(in_y);
        const float inverse_y_lerp = (1.0f - y_lerp);
        for (Eigen::Index x = 0; x < resized_width; ++x) {
          const float in_x = scaler(x, width_scale);
          const Eigen::Index left_x_index =
              std::max(static_cast<Eigen::Index>(floorf(in_x)),
                       static_cast<Eigen::Index>(0));
          const Eigen::Index right_x_index = std::min(
              static_cast<Eigen::Index>(ceilf(in_x)), original_width - 1);
          const float x_lerp = in_x - floorf(in_x);
          const float inverse_x_lerp = (1.0f - x_lerp);
          for (Eigen::Index c = 0; c < channels; ++c) {
            output_grad(b, top_y_index, left_x_index, c) +=
                T(input_grad(b, y, x, c) * inverse_y_lerp * inverse_x_lerp);
            output_grad(b, top_y_index, right_x_index, c) +=
                T(input_grad(b, y, x, c) * inverse_y_lerp * x_lerp);
            output_grad(b, bottom_y_index, left_x_index, c) +=
                T(input_grad(b, y, x, c) * y_lerp * inverse_x_lerp);
            output_grad(b, bottom_y_index, right_x_index, c) +=
                T(input_grad(b, y, x, c) * y_lerp * x_lerp);
=======
  explicit ResizeBilinearOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().ShortDebugString()));
    const Tensor& shape_t = context->input(1);
    OP_REQUIRES(context, shape_t.dims() == 1,
                errors::InvalidArgument("shape_t must be 1-dimensional",
                                        shape_t.shape().ShortDebugString()));
    OP_REQUIRES(context, shape_t.NumElements() == 2,
                errors::InvalidArgument("shape_t must have two elements",
                                        shape_t.shape().ShortDebugString()));

    auto Svec = shape_t.vec<int32>();
    // Initialize shape to the batch size of the input, then add
    // the rest of the dimensions
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({input.dim_size(0), Svec(0),
                                                Svec(1), input.dim_size(3)}),
                                &output));

    const int64 batch_size = input.dim_size(0);
    const int64 in_height = input.dim_size(1);
    const int64 in_width = input.dim_size(2);
    const int64 channels = input.dim_size(3);
    const int64 out_height = output->dim_size(1);
    const int64 out_width = output->dim_size(2);

    typename TTypes<T, 4>::ConstTensor input_data = input.tensor<T, 4>();
    typename TTypes<float, 4>::Tensor output_data = output->tensor<float, 4>();

    const float height_scale = in_height / static_cast<float>(out_height);
    const float width_scale = in_width / static_cast<float>(out_width);

    for (int b = 0; b < batch_size; ++b) {
      for (int y = 0; y < out_height; ++y) {
        const float in_y = y * height_scale;
        const int top_y_index = static_cast<int>(floorf(in_y));
        const int bottom_y_index =
            std::min(static_cast<int64>(ceilf(in_y)), (in_height - 1));
        const float y_lerp = in_y - top_y_index;
        const float inverse_y_lerp = (1.0f - y_lerp);
        for (int x = 0; x < out_width; ++x) {
          const float in_x = x * width_scale;
          const int left_x_index = static_cast<int>(floorf(in_x));
          const int right_x_index =
              std::min(static_cast<int64>(ceilf(in_x)), (in_width - 1));
          const float x_lerp = in_x - left_x_index;
          const float inverse_x_lerp = (1.0f - x_lerp);
          for (int c = 0; c < channels; ++c) {
            const float top_left = input_data(b, top_y_index, left_x_index, c);
            const float top_right =
                input_data(b, top_y_index, right_x_index, c);
            const float bottom_left =
                input_data(b, bottom_y_index, left_x_index, c);
            const float bottom_right =
                input_data(b, bottom_y_index, right_x_index, c);
            const float top =
                (top_left * inverse_x_lerp) + (top_right * x_lerp);
            const float bottom =
                (bottom_left * inverse_x_lerp) + (bottom_right * x_lerp);
            output_data(b, y, x, c) =
                (top * inverse_y_lerp) + (bottom * y_lerp);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
          }
        }
      }
    }
  }
<<<<<<< HEAD
  void operator()(const CPUDevice& d,
                  typename TTypes<float, 4>::ConstTensor input_grad,
                  const float height_scale, const float width_scale,
                  const bool half_pixel_centers,
                  typename TTypes<T, 4>::Tensor output_grad) {
    if (half_pixel_centers) {
      return ResizeGradCore(HalfPixelScaler(), input_grad, height_scale,
                            width_scale, output_grad);
    } else {
      return ResizeGradCore(LegacyScaler(), input_grad, height_scale,
                            width_scale, output_grad);
    }
  }
};

}  // namespace functor

=======
};

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#define REGISTER_KERNEL(T)                            \
  REGISTER_KERNEL_BUILDER(Name("ResizeBilinear")      \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("size"),    \
                          ResizeBilinearOp<CPUDevice, T>);

<<<<<<< HEAD
TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#define REGISTER_GRAD_KERNEL(T)                                             \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("ResizeBilinearGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ResizeBilinearOpGrad<CPUDevice, T>);

TF_CALL_half(REGISTER_GRAD_KERNEL);
TF_CALL_float(REGISTER_GRAD_KERNEL);
TF_CALL_double(REGISTER_GRAD_KERNEL);

#undef REGISTER_GRAD_KERNEL

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_KERNEL(T)                            \
  REGISTER_KERNEL_BUILDER(Name("ResizeBilinear")      \
                              .Device(DEVICE_GPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("size"),    \
                          ResizeBilinearOp<GPUDevice, T>);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#define REGISTER_GRAD_KERNEL(T)                                             \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("ResizeBilinearGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ResizeBilinearOpGrad<GPUDevice, T>);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_GRAD_KERNEL);

#undef REGISTER_GRAD_KERNEL

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

=======
REGISTER_KERNEL(uint8);
REGISTER_KERNEL(int8);
REGISTER_KERNEL(int32);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
#undef REGISTER_KERNEL

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}  // namespace tensorflow
