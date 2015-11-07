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

#define EIGEN_USE_GPU

#include <assert.h>
#include <stdio.h>

#include "tensorflow/core/kernels/random_op_gpu.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
=======
#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/random_op.h"

#include <stdio.h>
#include <assert.h>

#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

namespace tensorflow {

class OpKernelContext;

namespace functor {

typedef Eigen::GpuDevice GPUDevice;

<<<<<<< HEAD
=======
template <class Distribution, bool VariableSamplesPerOutput>
struct FillPhiloxRandomKernel;

// A cuda kernel to fill the data with random numbers from the specified
// distribution. Each output takes a fixed number of samples.
template <class Distribution>
struct FillPhiloxRandomKernel<Distribution, false> {
  typedef typename Distribution::ResultElementType T;
  PHILOX_DEVICE_FUNC void Run(random::PhiloxRandom gen, T* data, int64 size) {
    Distribution dist;
    const int kGroupSize = Distribution::kResultElementCount;

    const int32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int32 total_thread_count = gridDim.x * blockDim.x;
    int32 offset = thread_id * kGroupSize;
    gen.Skip(thread_id);

    while (offset < size) {
      typename Distribution::ResultType samples = dist(&gen);

      for (int i = 0; i < kGroupSize; ++i) {
        if (offset >= size) {
          return;
        }
        data[offset] = samples[i];
        ++offset;
      }

      offset += (total_thread_count - 1) * kGroupSize;
      gen.Skip(total_thread_count - 1);
    }
  }
};

// A cuda kernel to fill the data with random numbers from the specified
// distribution. Each output takes a variable number of samples.
template <class Distribution>
struct FillPhiloxRandomKernel<Distribution, true> {
  typedef typename Distribution::ResultElementType T;
  PHILOX_DEVICE_FUNC void Run(const random::PhiloxRandom& base_gen, T* data,
                              int64 size) {
    using random::PhiloxRandom;
    using random::SingleSampleAdapter;

    const int kReservedSamplesPerOutput = 256;
    const int kGroupSize = Distribution::kResultElementCount;
    const int kGeneratorSkipPerOutputGroup = kGroupSize *
                                             kReservedSamplesPerOutput /
                                             PhiloxRandom::kResultElementCount;

    const int32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int32 total_thread_count = gridDim.x * blockDim.x;
    int64 group_index = thread_id;
    int64 offset = group_index * kGroupSize;
    Distribution dist;

    while (offset < size) {
      // Since each output takes a variable number of samples, we need to
      // realign the generator to the beginning for the current output group
      PhiloxRandom gen = base_gen;
      gen.Skip(group_index * kGeneratorSkipPerOutputGroup);
      SingleSampleAdapter<PhiloxRandom> single_samples(&gen);

      typename Distribution::ResultType samples = dist(&single_samples);

      for (int i = 0; i < kGroupSize; ++i) {
        if (offset >= size) {
          return;
        }
        data[offset] = samples[i];
        ++offset;
      }

      offset += (total_thread_count - 1) * kGroupSize;
      group_index += total_thread_count;
    }
  }
};

// A simple launch pad to call the correct function templates to fill the data
template <class Distribution>
__global__ void __launch_bounds__(1024)
    FillPhiloxRandomKernelLaunch(random::PhiloxRandom base_gen,
                                 typename Distribution::ResultElementType* data,
                                 int64 size) {
  FillPhiloxRandomKernel<Distribution,
                         Distribution::kVariableSamplesPerOutput>()
      .Run(base_gen, data, size);
}

// Partial specialization for GPU
template <class Distribution>
struct FillPhiloxRandom<GPUDevice, Distribution> {
  typedef typename Distribution::ResultElementType T;
  typedef GPUDevice Device;
  void operator()(OpKernelContext*, const Device& d, random::PhiloxRandom gen,
                  T* data, int64 size) {
    const int32 block_size = d.maxCudaThreadsPerBlock();
    const int32 num_blocks =
        (d.getNumCudaMultiProcessors() * d.maxCudaThreadsPerMultiProcessor()) /
        block_size;

    FillPhiloxRandomKernelLaunch<
        Distribution><<<num_blocks, block_size, 0, d.stream()>>>(gen, data,
                                                                 size);
  }
};

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
// Explicit instantiation of the GPU distributions functors
// clang-format off
// NVCC cannot handle ">>" properly
template struct FillPhiloxRandom<
<<<<<<< HEAD
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, Eigen::half> >;
template struct FillPhiloxRandom<
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, float> >;
template struct FillPhiloxRandom<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, double> >;
template struct FillPhiloxRandom<
<<<<<<< HEAD
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, int32> >;
template struct FillPhiloxRandom<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, int64> >;
template struct FillPhiloxRandom<
    GPUDevice, random::NormalDistribution<random::PhiloxRandom, Eigen::half> >;
template struct FillPhiloxRandom<
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    GPUDevice, random::NormalDistribution<random::PhiloxRandom, float> >;
template struct FillPhiloxRandom<
    GPUDevice, random::NormalDistribution<random::PhiloxRandom, double> >;
template struct FillPhiloxRandom<
    GPUDevice, random::TruncatedNormalDistribution<
<<<<<<< HEAD
        random::SingleSampleAdapter<random::PhiloxRandom>, Eigen::half> >;
template struct FillPhiloxRandom<
    GPUDevice, random::TruncatedNormalDistribution<
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                   random::SingleSampleAdapter<random::PhiloxRandom>, float> >;
template struct FillPhiloxRandom<
    GPUDevice, random::TruncatedNormalDistribution<
                   random::SingleSampleAdapter<random::PhiloxRandom>, double> >;
// clang-format on

}  // namespace functor
}  // namespace tensorflow

<<<<<<< HEAD
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
=======
#endif  // GOOGLE_CUDA
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
