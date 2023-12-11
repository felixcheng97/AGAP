#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t, typename bound_t>
__device__ __forceinline__ scalar_t clamp(const scalar_t v, const bound_t lo, const bound_t hi) {
  return min(max(v, lo), hi);
}

template <typename scalar_t, bool dense_mode>
__global__ void total_variation_2d_add_grad_cuda_kernel(
    const scalar_t* __restrict__ param,
    scalar_t* __restrict__ grad,
    float wx, float wy, 
    const size_t sz_i, const size_t sz_j, const size_t N) {

  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index<N && (dense_mode || grad[index]!=0)) {
    // const size_t k = index % sz_k;
    // const size_t j = index / sz_k % sz_j;
    // const size_t i = index / sz_k / sz_j % sz_i;
    const size_t j = index / sz_j;
    const size_t i = index / sz_j % sz_i;

    float grad_to_add = 0;
    // grad_to_add += (k==0      ? 0 : wz * clamp(param[index]-param[index-1], -1.f, 1.f));
    // grad_to_add += (k==sz_k-1 ? 0 : wz * clamp(param[index]-param[index+1], -1.f, 1.f));
    // grad_to_add += (j==0      ? 0 : wy * clamp(param[index]-param[index-sz_k], -1.f, 1.f));
    // grad_to_add += (j==sz_j-1 ? 0 : wy * clamp(param[index]-param[index+sz_k], -1.f, 1.f));
    // grad_to_add += (i==0      ? 0 : wz * clamp(param[index]-param[index-sz_k*sz_j], -1.f, 1.f));
    // grad_to_add += (i==sz_i-1 ? 0 : wz * clamp(param[index]-param[index+sz_k*sz_j], -1.f, 1.f));
    grad_to_add += (j==0      ? 0 : wy * clamp(param[index]-param[index-1], -1.f, 1.f));
    grad_to_add += (j==sz_j-1 ? 0 : wy * clamp(param[index]-param[index+1], -1.f, 1.f));
    grad_to_add += (i==0      ? 0 : wx * clamp(param[index]-param[index-sz_j], -1.f, 1.f));
    grad_to_add += (i==sz_i-1 ? 0 : wx * clamp(param[index]-param[index+sz_j], -1.f, 1.f));


    grad[index] += grad_to_add;
  }
}

void total_variation_2d_add_grad_cuda(torch::Tensor param, torch::Tensor grad, float wx, float wy, bool dense_mode) {
  const size_t N = param.numel();
  const size_t sz_i = param.size(2);
  const size_t sz_j = param.size(3);
  // const size_t sz_k = param.size(4);
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  wx /= 4;
  wy /= 4;
  // wz /= 6;

  if(dense_mode) {
    AT_DISPATCH_FLOATING_TYPES(param.type(), "total_variation_2d_add_grad_cuda", ([&] {
      total_variation_2d_add_grad_cuda_kernel<scalar_t,true><<<blocks, threads>>>(
          param.data<scalar_t>(),
          grad.data<scalar_t>(),
          wx, wy,
          sz_i, sz_j, N);
    }));
  }
  else {
     AT_DISPATCH_FLOATING_TYPES(param.type(), "total_variation_2d_add_grad_cuda", ([&] {
      total_variation_2d_add_grad_cuda_kernel<scalar_t,false><<<blocks, threads>>>(
          param.data<scalar_t>(),
          grad.data<scalar_t>(),
          wx, wy,
          sz_i, sz_j, N);
    }));
  }
}

