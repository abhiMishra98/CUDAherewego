#include <cuda_runtime.h>
#include <memory.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_WIDTH 2

__global__ void tile_matMul(float *p, float *m, float *n, int width)
{
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];

    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    float pVal = 0;

    // Collaborative loading of M and N into shared memory

    for (int phase = 0; phase < (width - 1) / TILE_WIDTH + 1; ++phase) // Added because the input matrices are not clearly divisible by TILE_WIDTH
    {
        if (row < width && phase * TILE_WIDTH + threadIdx.x < width)
        {
            ds_M[threadIdx.y][threadIdx.x] = m[row * width + (threadIdx.x + TILE_WIDTH * phase)];
        }
        else
        {
            ds_M[threadIdx.y][threadIdx.x] = 0.0; // Any thread accessing non-existent index needs to load zero to SMEM
        }
        if (phase * TILE_WIDTH + threadIdx.y < width && col < width)
        {
            ds_N[threadIdx.y][threadIdx.x] = n[(threadIdx.y + phase * TILE_WIDTH) * width + col];
        }
        else
        {
            ds_N[threadIdx.y][threadIdx.x] = 0.0; // Any thread accessing non-existent index needs to load zero to SMEM
        }

        __syncthreads();
        if (row < width && col < width)
        {
            for (int i = 0; i < TILE_WIDTH; i++)
            {
                pVal += ds_M[threadIdx.y][i] * ds_N[i][threadIdx.x];
            }
        }
        __syncthreads();
    }
    if (row < width && col < width)
    {
        p[row * width + col] = pVal;
    }
}

int main()
{
    float *m, *n, *p, *d_m, *d_n, *d_p;
    int width = 4;

    m = (float *)calloc(width * width, sizeof(float));
    n = (float *)calloc(width * width, sizeof(float));
    p = (float *)calloc(width * width, sizeof(float));

    srand((unsigned int)time(NULL));
    for (int i = 0; i < width * width; ++i)
    {
        m[i] = (float)(rand() % 10);
        n[i] = (float)(rand() % 10);
    }

    // Allocate on device
    cudaMalloc(&d_m, width * width * sizeof(float));
    cudaMalloc(&d_n, width * width * sizeof(float));
    cudaMalloc(&d_p, width * width * sizeof(float));

    // Move data from host to device
    cudaMemcpy(d_m, m, width * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, n, width * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, p, width * width * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((width + TILE_WIDTH - 1) / TILE_WIDTH, (width + TILE_WIDTH - 1) / TILE_WIDTH); // Notice grid dim changed because no more the tile is divisible by the input matrix

    tile_matMul<<<gridDim, blockDim>>>(d_p, d_m, d_n, width);
    cudaError_t err = cudaGetLastError();
    cudaDeviceSynchronize();
    cudaMemcpy(p, d_p, width * width * sizeof(float), cudaMemcpyDeviceToHost);

    printf("M:\n");
    for (int r = 0; r < width; ++r)
    {
        for (int c = 0; c < width; ++c)
            printf("%6.1f ", m[r * width + c]);
        printf("\n");
    }

    printf("N:\n");
    for (int r = 0; r < width; ++r)
    {
        for (int c = 0; c < width; ++c)
            printf("%6.1f ", n[r * width + c]);
        printf("\n");
    }

    printf("P = M * N:\n");
    for (int r = 0; r < width; ++r)
    {
        for (int c = 0; c < width; ++c)
            printf("%6.1f ", p[r * width + c]);
        printf("\n");
    }

    free(m);
    free(n);
    free(p);

    cudaFree(d_m);
    cudaFree(d_n);
    cudaFree(d_p);

    return 0;
}