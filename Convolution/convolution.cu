#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

#define MAX_MASK_WIDTH 5

__constant__ float d_M[MAX_MASK_WIDTH * MAX_MASK_WIDTH];

__global__ void conv_1d(float *N, float *M, float *P, int maskW, int width)
{
    int tdx = blockIdx.x * blockDim.x + threadIdx.x;
    float pVal = 0.0f;
    int ip_Start = tdx - (maskW / 2);

    if (tdx < width)
    {
        for (int i = 0; i < maskW; i++)
        {
            if (ip_Start + i >= 0 && ip_Start + i < width)
            {
                pVal += M[i] * N[ip_Start + i];
            }
        }
        P[tdx] = pVal;
    }
}

__global__ void conv_1d_shared(float *N, float *M, float *P, int maskW, int width, int o_tile_width)
{
    __shared__ float d_N[1024];
    int index_o = blockIdx.x * o_tile_width + threadIdx.x;
    int index_i = index_o - (maskW / 2);
    if (index_i >= 0 && index_i < width)
    {
        d_N[threadIdx.x] = N[index_i];
    }
    else
    {
        d_N[threadIdx.x] = 0.0f; // Especially for first block
    }
    __syncthreads();

    if (threadIdx.x < o_tile_width && index_o < width)
    {
        float pVal = 0.0f;
        for (int i = 0; i < maskW; i++)
        {
            pVal += M[i] * d_N[threadIdx.x + i];
        }
        P[index_o] = pVal;
    }
}

__global__ void conv_2d(float *N, float *M, float *P, int maskW, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float pVal = 0.0f;
    if (col < width && row < height)
    {

        int ip_Start_row = row - (maskW / 2);
        int ip_Start_col = col - (maskW / 2);

        for (int i = 0; i < maskW; ++i)
        {
            for (int j = 0; j < maskW; ++j)
            {
                int curRow = ip_Start_row + i;
                int curCol = ip_Start_col + j;

                if (curRow > -1 && curRow < height && curCol > -1 && curCol < width)
                {
                    pVal += N[curRow * width + curCol] * M[i * maskW + j];
                }
            }
        }
        P[row * width + col] = pVal;
    }
}

__global__ void conv_2d_shared(float *N, float *P, int maskW, int width, int height, int o_tile_width, size_t pitchN, size_t pitchP)
{
    __shared__ float d_N[32][32];
    int index_o_col = blockIdx.x * o_tile_width + threadIdx.x;
    int index_o_row = blockIdx.y * o_tile_width + threadIdx.y;
    int index_i_col = index_o_col - (maskW / 2);
    int index_i_row = index_o_row - (maskW / 2);

    if (index_i_col >= 0 && index_i_col < width && index_i_row >= 0 && index_i_row < height)
    {
        const float *N_row = (const float *)((const char *)N + index_i_row * pitchN);
        d_N[threadIdx.y][threadIdx.x] = N_row[index_i_col];
    }
    else
    {
        d_N[threadIdx.y][threadIdx.x] = 0.0f; // Especially for first block
    }
    __syncthreads();

    if (threadIdx.x < o_tile_width && threadIdx.y < o_tile_width &&
        index_o_col < width && index_o_row < height)
    {
        float pVal = 0.0f;
        for (int i = 0; i < maskW; i++)
        {
            for (int j = 0; j < maskW; j++)
            {
                pVal += d_M[i * maskW + j] * d_N[threadIdx.y + i][threadIdx.x + j];
            }
        }
        float *P_row = (float *)((char *)P + index_o_row * pitchP);
        P_row[index_o_col] = pVal;
    }
}

int main()
{
    srand(42);

    // ==================== 1D: conv_1d vs conv_1d_shared ====================
    // Same width/mask/input for both, so the timings are a fair naive-vs-tiled
    // comparison, not just two unrelated numbers.
    int width1d = 16 * 1024 * 1024; // 16,777,216 elements
    int maskW1d = 5;
    int o_width1d = 1020; // conv_1d_shared's output tile width
    int b_width1d = 1024; // conv_1d_shared's block dim (o_width1d + maskW1d - 1)

    size_t bytes1d = (size_t)width1d * sizeof(float);
    float *h_N1d = (float *)malloc(bytes1d);
    float *h_M1d = (float *)malloc(maskW1d * sizeof(float));
    float *h_P1d = (float *)malloc(bytes1d);

    for (int i = 0; i < width1d; ++i)
    {
        h_N1d[i] = (float)(rand() % 100) / 10.0f;
    }
    for (int i = 0; i < maskW1d; ++i)
    {
        h_M1d[i] = (float)(rand() % 100) / 100.0f;
    }

    float *d_N1d, *d_M1d, *d_P1d;
    cudaMalloc(&d_N1d, bytes1d);
    cudaMalloc(&d_M1d, maskW1d * sizeof(float));
    cudaMalloc(&d_P1d, bytes1d);
    cudaMemcpy(d_N1d, h_N1d, bytes1d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M1d, h_M1d, maskW1d * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim1d(256);
    dim3 gridDim1d((width1d + blockDim1d.x - 1) / blockDim1d.x);

    cudaEvent_t startNaive1d, stopNaive1d;
    cudaEventCreate(&startNaive1d);
    cudaEventCreate(&stopNaive1d);
    cudaEventRecord(startNaive1d);
    conv_1d<<<gridDim1d, blockDim1d>>>(d_N1d, d_M1d, d_P1d, maskW1d, width1d);
    cudaEventRecord(stopNaive1d);
    cudaEventSynchronize(stopNaive1d);
    float msNaive1d = 0.0f;
    cudaEventElapsedTime(&msNaive1d, startNaive1d, stopNaive1d);
    cudaMemcpy(h_P1d, d_P1d, bytes1d, cudaMemcpyDeviceToHost);

    dim3 blockDim1dShared(b_width1d);
    dim3 gridDim1dShared(((width1d - 1) / o_width1d + 1), 1, 1);

    cudaEvent_t startShared1d, stopShared1d;
    cudaEventCreate(&startShared1d);
    cudaEventCreate(&stopShared1d);
    cudaEventRecord(startShared1d);
    conv_1d_shared<<<gridDim1dShared, blockDim1dShared>>>(d_N1d, d_M1d, d_P1d, maskW1d, width1d, o_width1d);
    cudaEventRecord(stopShared1d);
    cudaEventSynchronize(stopShared1d);
    float msShared1d = 0.0f;
    cudaEventElapsedTime(&msShared1d, startShared1d, stopShared1d);
    cudaMemcpy(h_P1d, d_P1d, bytes1d, cudaMemcpyDeviceToHost);

    printf("conv_1d:        %8.4f ms\n", msNaive1d);
    printf("conv_1d_shared: %8.4f ms  (%.2fx vs conv_1d)\n", msShared1d, msNaive1d / msShared1d);

    cudaEventDestroy(startNaive1d);
    cudaEventDestroy(stopNaive1d);
    cudaEventDestroy(startShared1d);
    cudaEventDestroy(stopShared1d);
    cudaFree(d_N1d);
    cudaFree(d_M1d);
    cudaFree(d_P1d);
    free(h_N1d);
    free(h_M1d);
    free(h_P1d);

    // ==================== 2D: conv_2d vs conv_2d_shared ====================
    // Same width/height/mask/input for both.
    int width2d = 4096;
    int height2d = 4096;
    int maskW2d = 5;
    int o_tile_width2d = 28; // conv_2d_shared's output tile width (blockDim = 28 + maskW2d - 1 = 32)

    size_t rowBytes2d = (size_t)width2d * sizeof(float);
    size_t bytes2d = rowBytes2d * height2d;
    float *h_N2d = (float *)malloc(bytes2d);
    float *h_M2d = (float *)malloc(maskW2d * maskW2d * sizeof(float));
    float *h_P2d = (float *)malloc(bytes2d);

    for (size_t i = 0; i < (size_t)width2d * height2d; ++i)
    {
        h_N2d[i] = (float)(rand() % 100) / 10.0f;
    }
    for (int i = 0; i < maskW2d * maskW2d; ++i)
    {
        h_M2d[i] = (float)(rand() % 100) / 100.0f;
    }

    // conv_2d (naive): flat, tightly packed buffers, mask as a plain pointer.
    float *d_N2d, *d_M2d, *d_P2d;
    cudaMalloc(&d_N2d, bytes2d);
    cudaMalloc(&d_M2d, maskW2d * maskW2d * sizeof(float));
    cudaMalloc(&d_P2d, bytes2d);
    cudaMemcpy(d_N2d, h_N2d, bytes2d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2d, h_M2d, maskW2d * maskW2d * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim2d(16, 16);
    dim3 gridDim2d((width2d + blockDim2d.x - 1) / blockDim2d.x, (height2d + blockDim2d.y - 1) / blockDim2d.y);

    cudaEvent_t startNaive2d, stopNaive2d;
    cudaEventCreate(&startNaive2d);
    cudaEventCreate(&stopNaive2d);
    cudaEventRecord(startNaive2d);
    conv_2d<<<gridDim2d, blockDim2d>>>(d_N2d, d_M2d, d_P2d, maskW2d, width2d, height2d);
    cudaEventRecord(stopNaive2d);
    cudaEventSynchronize(stopNaive2d);
    float msNaive2d = 0.0f;
    cudaEventElapsedTime(&msNaive2d, startNaive2d, stopNaive2d);
    cudaMemcpy(h_P2d, d_P2d, bytes2d, cudaMemcpyDeviceToHost);

    cudaEventDestroy(startNaive2d);
    cudaEventDestroy(stopNaive2d);
    cudaFree(d_N2d);
    cudaFree(d_M2d);
    cudaFree(d_P2d);

    // conv_2d_shared: __constant__ mask, pitched N/P.
    float *d_N2ds, *d_P2ds;
    size_t pitchN2ds, pitchP2ds;
    cudaMallocPitch(&d_N2ds, &pitchN2ds, rowBytes2d, height2d);
    cudaMallocPitch(&d_P2ds, &pitchP2ds, rowBytes2d, height2d);
    cudaMemcpy2D(d_N2ds, pitchN2ds, h_N2d, rowBytes2d, rowBytes2d, height2d, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_M, h_M2d, maskW2d * maskW2d * sizeof(float));

    dim3 blockDim2ds(o_tile_width2d + maskW2d - 1, o_tile_width2d + maskW2d - 1);
    dim3 gridDim2ds((width2d + o_tile_width2d - 1) / o_tile_width2d, (height2d + o_tile_width2d - 1) / o_tile_width2d);

    cudaEvent_t startShared2d, stopShared2d;
    cudaEventCreate(&startShared2d);
    cudaEventCreate(&stopShared2d);
    cudaEventRecord(startShared2d);
    conv_2d_shared<<<gridDim2ds, blockDim2ds>>>(d_N2ds, d_P2ds, maskW2d, width2d, height2d, o_tile_width2d, pitchN2ds, pitchP2ds);
    cudaEventRecord(stopShared2d);
    cudaEventSynchronize(stopShared2d);
    float msShared2d = 0.0f;
    cudaEventElapsedTime(&msShared2d, startShared2d, stopShared2d);
    cudaMemcpy2D(h_P2d, rowBytes2d, d_P2ds, pitchP2ds, rowBytes2d, height2d, cudaMemcpyDeviceToHost);

    printf("conv_2d:        %8.4f ms\n", msNaive2d);
    printf("conv_2d_shared: %8.4f ms  (%.2fx vs conv_2d)\n", msShared2d, msNaive2d / msShared2d);

    cudaEventDestroy(startShared2d);
    cudaEventDestroy(stopShared2d);
    cudaFree(d_N2ds);
    cudaFree(d_P2ds);
    free(h_N2d);
    free(h_M2d);
    free(h_P2d);

    return 0;
}