#include <stdio.h>

__global__ void applyFilter(const unsigned char *input, unsigned char *output, const unsigned int width, const unsigned int height, const float *kernel, const unsigned int kernelWidth) {

    const unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;

    // if(row < height && col < width) {
    //     size_t ptr = (col + row * width) * 3 + blockIdx.z;
    //     output[ptr] = input[ptr];
    // }
    // return;

    if(row < height && col < width) {
        const int half = kernelWidth / 2;
        float blur = 0.0;
        for(int i = -half; i <= half; i++) {
            for(int j = -half; j <= half; j++) {

                const unsigned int y = max(0, min(height - 1, row + i));
                const unsigned int x = max(0, min(width - 1, col + j));

                const float w = kernel[(j + half) + (i + half) * kernelWidth];
                blur += w * input[(x + y * width) * 3 + blockIdx.z];
            }
        }        
        output[(col + row * width) * 3 + blockIdx.z] = static_cast<unsigned char>(blur);
    }
}

void call_applyFilter(const unsigned char *input, unsigned char *output, const unsigned int width, const unsigned int height, const float *kernel, const unsigned int kernelWidth) {
    //Set reasonable block size (number of threads per block)
    const dim3 blockSize(32, 32, 1);
    const dim3 gridSize(width/blockSize.x+1, height/blockSize.y+1, 3);

    applyFilter<<<gridSize,blockSize>>>(input, output, width, height, kernel, kernelWidth);
}
