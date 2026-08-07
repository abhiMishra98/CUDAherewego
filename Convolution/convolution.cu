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
    int width = 1024;
    int maskW = 5;
    float *d_N, *d_M, *d_P;

    dim3 blockDim1d(256);
    dim3 gridDim1d((width + blockDim1d.x - 1) / blockDim1d.x);
    conv_1d<<<gridDim1d, blockDim1d>>>(d_N, d_M, d_P, maskW, width);

    int width2d = 1024;
    int height2d = 1024;
    int maskW2d = 5;
    float *d_N2, *d_M2, *d_P2;

    dim3 blockDim2d(16, 16);
    dim3 gridDim2d((width2d + blockDim2d.x - 1) / blockDim2d.x, (height2d + blockDim2d.y - 1) / blockDim2d.y);
    conv_2d<<<gridDim2d, blockDim2d>>>(d_N2, d_M2, d_P2, maskW2d, width2d, height2d);

    int width1d = 1024;
    int o_width1d = 1020;
    int b_width1d = 1024;
    float *d_N1d, *d_M1d, *d_P1d;

    dim3 blockDim1dShared(b_width1d);
    dim3 gridDim1dShared(((width - 1) / o_width1d + 1), 1, 1);
    conv_1d_shared<<<gridDim1dShared, blockDim1dShared>>>(d_N1d, d_M1d, d_P1d, maskW, width1d, o_width1d);

    int width2ds = 1024;
    int height2ds = 1024;
    int maskW2ds = 5;
    int o_tile_width2ds = 28; // blockDim = 28 + maskW - 1 = 32, the max block dim the 32x32 SMEM tile allows

    float *d_N2ds, *d_P2ds;
    size_t pitchN2ds, pitchP2ds;
    cudaMallocPitch(&d_N2ds, &pitchN2ds, width2ds * sizeof(float), height2ds);
    cudaMallocPitch(&d_P2ds, &pitchP2ds, width2ds * sizeof(float), height2ds);

    float h_M2ds[MAX_MASK_WIDTH * MAX_MASK_WIDTH] = {0};
    cudaMemcpyToSymbol(d_M, h_M2ds, maskW2ds * maskW2ds * sizeof(float));

    dim3 blockDim2ds(o_tile_width2ds + maskW2ds - 1, o_tile_width2ds + maskW2ds - 1);
    dim3 gridDim2ds((width2ds + o_tile_width2ds - 1) / o_tile_width2ds, (height2ds + o_tile_width2ds - 1) / o_tile_width2ds);
    conv_2d_shared<<<gridDim2ds, blockDim2ds>>>(d_N2ds, d_P2ds, maskW2ds, width2ds, height2ds, o_tile_width2ds, pitchN2ds, pitchP2ds);

    cudaFree(d_N2ds);
    cudaFree(d_P2ds);

    return 0;
}