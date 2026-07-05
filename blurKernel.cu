#include<cuda_runtime.h>
#define BLUR_SIZE 1
__global__ void blurKernel (int width, int height, unsigned char *in, unsigned char *out){
    int y = blockIdx.x  * blockDim.x + threadIdx.x;
    int x = blockIdx.y  * blockDim.y + threadIdx.y;

    if(y < width && x < height){
        int pixVal = 0;
        int pixels = 0;
        
        //Idea is to average nearby elements
        for(int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE;++blurRow){
            for(int blurCol = -BLUR_SIZE; blurCol <=BLUR_SIZE; ++blurCol){
                //Important to understand the below, blurRow changes row and blurCol changes col
                // curRow as made consistent before as well, represents change in y
                // curCol as made consistent before as well, represents change in x
                int curRow = x + blurRow;
                int curCol = y + blurCol;

                if(curRow > -1 && curRow < height && curCol > -1 && curCol < width){
                    pixVal += in[curRow * width + curCol];
                    pixels++;
                }
            }
        }

        out[col*width + row] = (unsigned char)(pixVal/pixels);
    }
}

int main(){

    dim3 gridDim ((n-1)/16 +1, (m-1)/16 + 1,1);
    dim3 blockDim (16,16,1);

    blurKernel<<<gridDim, blockDim>>>(width, height, in, out);
}