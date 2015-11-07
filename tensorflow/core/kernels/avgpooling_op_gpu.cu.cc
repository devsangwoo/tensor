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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
=======
#if GOOGLE_CUDA
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#define EIGEN_USE_GPU

#include <stdio.h>
<<<<<<< HEAD

#include <iostream>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/avgpooling_op.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
=======
#include <iostream>

#include "tensorflow/core/kernels/avgpooling_op.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

#define DEFINE_GPU_KERNELS(T) \
  template struct functor::SpatialAvgPooling<GPUDevice, T>;

<<<<<<< HEAD
DEFINE_GPU_KERNELS(Eigen::half)
DEFINE_GPU_KERNELS(float)
DEFINE_GPU_KERNELS(double)

#undef DEFINE_GPU_KERNELS

template <typename dtype>
__global__ void AvePoolBackwardNHWC(
    const int nthreads, const dtype* const __restrict__ top_diff, const int num,
    const int height, const int width, const int channels,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_t,
    const int pad_l, dtype* const __restrict__ bottom_diff) {
  GPU_1D_KERNEL_LOOP(index, nthreads) {
=======
DEFINE_GPU_KERNELS(float)

#undef DEFINE_GPU_KERNELS

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

static const int CAFFE_CUDA_NUM_THREADS = 1024;

template <typename dtype>
__global__ void AvePoolBackwardNHWC(const int nthreads,
                                    const dtype* const top_diff, const int num,
                                    const int height, const int width,
                                    const int channels, const int pooled_height,
                                    const int pooled_width, const int kernel_h,
                                    const int kernel_w, const int stride_h,
                                    const int stride_w, const int pad_t,
                                    const int pad_l, dtype* const bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    // find out the local index
    // find out the local offset
    const int c = index % channels;
    const int w = index / channels % width + pad_l;
    const int h = (index / channels / width) % height + pad_t;
    const int n = index / channels / width / height;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
<<<<<<< HEAD
    dtype gradient(0);
=======
    dtype gradient = 0;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    const dtype* const top_diff_slice =
        top_diff + n * pooled_height * pooled_width * channels + c;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_t;
        int wstart = pw * stride_w - pad_l;
        int hend = min(hstart + kernel_h, height);
        int wend = min(wstart + kernel_w, width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        int pool_size = (hend - hstart) * (wend - wstart);
<<<<<<< HEAD
        gradient += top_diff_slice[(ph * pooled_width + pw) * channels] /
                    dtype(pool_size);
=======
        gradient +=
            top_diff_slice[(ph * pooled_width + pw) * channels] / pool_size;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename T>
bool RunAvePoolBackwardNHWC(const T* const top_diff, const int num,
                            const int height, const int width,
                            const int channels, const int pooled_height,
                            const int pooled_width, const int kernel_h,
                            const int kernel_w, const int stride_h,
                            const int stride_w, const int pad_t,
                            const int pad_l, T* const bottom_diff,
                            const GPUDevice& d) {
  int x_size = num * height * width * channels;
<<<<<<< HEAD
  GpuLaunchConfig config = GetGpuLaunchConfig(x_size, d);
  TF_CHECK_OK(GpuLaunchKernel(
      AvePoolBackwardNHWC<T>, config.block_count, config.thread_per_block, 0,
      d.stream(), config.virtual_thread_count, top_diff, num, height, width,
      channels, pooled_height, pooled_width, kernel_h, kernel_w, stride_h,
      stride_w, pad_t, pad_t, bottom_diff));
=======
  int thread_per_block =
      std::min(CAFFE_CUDA_NUM_THREADS, d.maxCudaThreadsPerMultiProcessor());
  int block_count = (x_size + thread_per_block - 1) / thread_per_block;
  AvePoolBackwardNHWC<T><<<block_count, thread_per_block, 0, d.stream()>>>(
      x_size, top_diff, num, height, width, channels, pooled_height,
      pooled_width, kernel_h, kernel_w, stride_h, stride_w, pad_t, pad_t,
      bottom_diff);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  return d.ok();
}

template bool RunAvePoolBackwardNHWC(
<<<<<<< HEAD
    const double* const top_diff, const int num, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_t, const int pad_l,
    double* const bottom_diff, const GPUDevice& d);
template bool RunAvePoolBackwardNHWC(
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    const float* const top_diff, const int num, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_t, const int pad_l,
    float* const bottom_diff, const GPUDevice& d);
<<<<<<< HEAD
template bool RunAvePoolBackwardNHWC(
    const Eigen::half* const top_diff, const int num, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_t, const int pad_l,
    Eigen::half* const bottom_diff, const GPUDevice& d);

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
=======

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
