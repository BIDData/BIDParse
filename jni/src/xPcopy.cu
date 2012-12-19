#include <cuda_runtime.h>
#include "testSparse.h"
#include <stdio.h>

__global__ void __pcopy_setup(int *inds, float **src, float *target, int stride, int ncols) {
  int nx = blockDim.x * gridDim.x;
  int ix = threadIdx.x + blockDim.x * blockIdx.x;
  for (int x = ix; x < ncols; x += nx) {
    src[x] = &target[inds[x] * stride];
  }
}

// copy and transpose columns of the input matrix into the output matrix. nrows, ncols refer to input matrix

__global__ void __pcopy_transpose_in(int *iptrs, float *in, float *out, int outstride, int nrows, int ncols) {
  int nx = BLOCKDIM * gridDim.x;
  int ny = BLOCKDIM * gridDim.y;
  int ix = BLOCKDIM * blockIdx.x;
  int iy = BLOCKDIM * blockIdx.y;
  __shared__ float tile[BLOCKDIM][BLOCKDIM+1];

  for (int yb = iy; yb < ncols; yb += ny) {
    for (int xb = ix; xb < nrows; xb += nx) {
      if (xb + threadIdx.x < nrows) {
        int ylim = min(ncols, yb + BLOCKDIM);
        for (int y = threadIdx.y + yb; y < ylim; y += blockDim.y) {
          tile[threadIdx.x][y-iy] = in[iptrs[y]*nrows + threadIdx.x + xb];
        }
      }
      __syncthreads();
      if (yb + threadIdx.x < ncols) {
        int xlim = min(nrows, xb + BLOCKDIM);
        for (int x = threadIdx.y + xb; x < xlim; x += blockDim.y) {
          out[threadIdx.x + yb + x*outstride] = tile[x-ix][threadIdx.x];
        }
      }
      __syncthreads();
    }
  } 
}

int pcopy_transpose_in(int *iptrs, float *in, float *out, int stride, int nrows, int ncols) {
  const dim3 griddims(20,256,1);
  const dim3 blockdims(BLOCKDIM,INBLOCK,1);
  cudaError_t err;
  __pcopy_transpose_in<<<griddims,blockdims>>>(iptrs, in, out, stride, nrows, ncols); 
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {fprintf(stderr, "cuda error in pcopy_transpose_out"); return err;}
  return 0;
}

// copy and transpose the input matrix into columns of the output matrix. nrows, ncols refer to output matrix

__global__ void __pcopy_transpose_out(int *optrs, float *in, float *out, int instride, int nrows, int ncols) {
  int nx = BLOCKDIM * gridDim.x;
  int ny = BLOCKDIM * gridDim.y;
  int ix = BLOCKDIM * blockIdx.x;
  int iy = BLOCKDIM * blockIdx.y;
  __shared__ float tile[BLOCKDIM][BLOCKDIM+1];

  for (int yb = iy; yb < ncols; yb += ny) {
    for (int xb = ix; xb < nrows; xb += nx) {
      if (yb + threadIdx.x < ncols) {
        int xlim = min(nrows, xb + BLOCKDIM);
        for (int x = threadIdx.y + xb; x < xlim; x += blockDim.y) {
          tile[x-ix][threadIdx.x] = in[threadIdx.x + yb + x*instride];
        }
      }
      __syncthreads();
      if (xb + threadIdx.x < nrows) {
        int ylim = min(ncols, yb + BLOCKDIM);
        for (int y = threadIdx.y + yb; y < ylim; y += blockDim.y) {
          atomicAdd(&out[optrs[y]*nrows + threadIdx.x + xb], tile[threadIdx.x][y-iy]);
        }
      }
      __syncthreads();
    }
  } 
}

int pcopy_transpose_out(int *optrs, float *in, float *out, int stride, int nrows, int ncols) {
  const dim3 griddims(20,256,1);
  const dim3 blockdims(BLOCKDIM,INBLOCK,1);
  cudaError_t err;
  __pcopy_transpose_out<<<griddims,blockdims>>>(optrs, in, out, stride, nrows, ncols); 
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {fprintf(stderr, "cuda error in pcopy_transpose_out"); return err;}
  return 0;
}


__global__ void __pcopy(float **src, float *dest, int nrows, int ncols, int stride) {
  int nx = blockDim.x * gridDim.x;
  int ny = blockDim.y * gridDim.y;
  int ix = threadIdx.x + blockDim.x * blockIdx.x;
  int iy = threadIdx.y + blockDim.y * blockIdx.y;

  for (int y = iy; y < ncols ; y += ny) {
    float *p = src[y];
    int yoff = y * stride;
    for (int x = ix; x < nrows ; x += nx) {
      dest[x + yoff] = p[x];
    }
  }
}
