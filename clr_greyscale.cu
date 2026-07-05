#include <cuda_runtime.h>
#define CHANNEL 3

__global__ void change2grayScale(int width, int height, unsigned char *rgbImage, unsigned char   *grayscale){

    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;

    if(x < width && y <height){
        //Get 1-D coordinate of the pixel
        int grayOffset = y*width + x;
        int rgboffset = grayOffset*CHANNEL;

        unsigned char r = rgbImage[rgboffset];
        unsigned char g = rgbImage[rgboffset + 1];
        unsigned char b = rgbImage[rgboffset + 2];

        grayscale[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

int main(){
    //Occupacny - enough threads/block to saturate the number of active warps possible
    // Not always, high occupancy is good. High occupancy means, lower number of shared resources for each thread which might lead to memory spilling/register spilling.
    dim3 gridDim ((((n-1)/16 +1,(m-1)/16+1,1)));
    dim3 blockDim (16,16,1);
    change2grayScale<<<gridDim,blockDim>>>(w,h,d_Pin,d_Pout);
}
