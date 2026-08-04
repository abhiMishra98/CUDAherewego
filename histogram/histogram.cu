#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cstdio>
#include <cstring>

#define CUDA_CHECK(call)                                                      \
    do                                                                       \
    {                                                                        \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

__global__ void histo_kernel(unsigned char *buffer, long size, int *histo)
{
    __shared__ int d_histoBuff[7]; // For privatisation, SMEM for each thread block
    int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (threadIdx.x < 7) // initialise private buffer for each thread block
        d_histoBuff[threadIdx.x] = 0;
    __syncthreads();

    while (tIdx < size)
    {
        int alpha_pos = buffer[tIdx] - 'a';
        if (alpha_pos >= 0 && alpha_pos <= 25)
        {
            atomicAdd(&(d_histoBuff[alpha_pos / 4]), 1);
        }
        tIdx += stride;
    }

    __syncthreads();

    if (threadIdx.x < 7)
    {
        atomicAdd(&(histo[threadIdx.x]), d_histoBuff[threadIdx.x]);
    }
}

int main()
{
    const char *testString = "the quick brown fox jumps over the lazy dog abcdefghijklmnopqrstuvwxyz";
    long size = (long)strlen(testString);

    unsigned char *d_buffer;
    int *d_histo;
    CUDA_CHECK(cudaMalloc(&d_buffer, size));
    CUDA_CHECK(cudaMalloc(&d_histo, 7 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_buffer, testString, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_histo, 0, 7 * sizeof(int)));

    dim3 blockDim(256);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);

    histo_kernel<<<gridDim, blockDim>>>(d_buffer, size, d_histo);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_buffer);
    cudaFree(d_histo);

    return 0;
}