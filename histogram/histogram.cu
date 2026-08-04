#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cstdio>
#include <cstring>
#include <cstdlib>

#define CUDA_CHECK(call)                                                     \
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

__global__ void histo_kernel_naive(unsigned char *buffer, long size, int *histo)
{
    int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (tIdx < size)
    {
        int alpha_pos = buffer[tIdx] - 'a';
        if (alpha_pos >= 0 && alpha_pos <= 25)
        {
            atomicAdd(&(histo[alpha_pos / 4]), 1); // every hit contends directly on global memory
        }
        tIdx += stride;
    }
}

__global__ void histo_kernel(unsigned char *buffer, long size, int *histo)
{
    __shared__ int d_histoBuff[7]; // For privatisation, SMEM for each thread block
    int reg_histoBuff[7] = {0};    // For privatisation, registers for each thread

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
            reg_histoBuff[alpha_pos / 4]++;
        }
        tIdx += stride;
    }

    for (int i = 0; i < 7; ++i) // merge thread-private counts into the block's shared buffer
    {
        if (reg_histoBuff[i] > 0)
            atomicAdd(&(d_histoBuff[i]), reg_histoBuff[i]);
    }

    __syncthreads();

    if (threadIdx.x < 7)
    {
        atomicAdd(&(histo[threadIdx.x]), d_histoBuff[threadIdx.x]);
    }
}

int main()
{
    FILE *file = fopen("input.txt", "rb");
    if (!file)
    {
        fprintf(stderr, "Failed to open input.txt\n");
        return EXIT_FAILURE;
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);

    unsigned char *h_buffer = (unsigned char *)malloc(size);
    fread(h_buffer, 1, size, file);
    fclose(file);

    unsigned char *d_buffer;
    int *d_histo, *d_histo_naive;
    CUDA_CHECK(cudaMalloc(&d_buffer, size));
    CUDA_CHECK(cudaMalloc(&d_histo, 7 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_histo_naive, 7 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_buffer, h_buffer, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_histo, 0, 7 * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_histo_naive, 0, 7 * sizeof(int)));

    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device));

    dim3 blockDim(256);

    int numBlocksPerSm = 0; // max blocks/SM this exact kernel can keep resident, given its register/SMEM footprint
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, histo_kernel, blockDim.x, 0));
    dim3 gridDim(deviceProp.multiProcessorCount * numBlocksPerSm); // fixed grid, sized to the GPU, not the input

    histo_kernel_naive<<<gridDim, blockDim>>>(d_buffer, size, d_histo_naive);
    CUDA_CHECK(cudaGetLastError());

    histo_kernel<<<gridDim, blockDim>>>(d_buffer, size, d_histo);
    CUDA_CHECK(cudaGetLastError());

    free(h_buffer);
    cudaFree(d_buffer);
    cudaFree(d_histo);
    cudaFree(d_histo_naive);

    return 0;
}