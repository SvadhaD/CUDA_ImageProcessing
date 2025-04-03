#include "morphology.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#define TILE_SIZE 16

__global__ void dilationKernel(const unsigned char* inputImage, unsigned char* outputImage, 
                               const int* structuringElement, int imageWidth, int imageHeight, 
                               int elementWidth) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    int halfElement = elementWidth / 2;

    if (row < imageHeight && col < imageWidth) {
        unsigned char maxVal = 0;
        for (int i = -halfElement; i <= halfElement; i++) {
            for (int j = -halfElement; j <= halfElement; j++) {
                int x = min(max(col + j, 0), imageWidth - 1);
                int y = min(max(row + i, 0), imageHeight - 1);
                if (structuringElement[(i + halfElement) * elementWidth + (j + halfElement)] == 1) {
                    maxVal = max(maxVal, inputImage[y * imageWidth + x]);
                }
            }
        }
        outputImage[row * imageWidth + col] = maxVal;
    }
}

__global__ void erosionKernel(const unsigned char* inputImage, unsigned char* outputImage, 
                              const int* structuringElement, int imageWidth, int imageHeight, 
                              int elementWidth) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    int halfElement = elementWidth / 2;

    if (row < imageHeight && col < imageWidth) {
        unsigned char minVal = 255;
        for (int i = -halfElement; i <= halfElement; i++) {
            for (int j = -halfElement; j <= halfElement; j++) {
                int x = min(max(col + j, 0), imageWidth - 1);
                int y = min(max(row + i, 0), imageHeight - 1);
                if (structuringElement[(i + halfElement) * elementWidth + (j + halfElement)] == 1) {
                    minVal = min(minVal, inputImage[y * imageWidth + x]);
                }
            }
        }
        outputImage[row * imageWidth + col] = minVal;
    }
}

void applyDilation(const unsigned char* h_inputImage, unsigned char* h_outputImage, 
                   const int* h_structuringElement, int imageWidth, int imageHeight, int elementWidth) {
    unsigned char *d_inputImage, *d_outputImage;
    int *d_structuringElement;
    int imageSize = imageWidth * imageHeight * sizeof(unsigned char);
    int elementSize = elementWidth * elementWidth * sizeof(int);

    cudaMalloc((void**)&d_inputImage, imageSize);
    cudaMalloc((void**)&d_outputImage, imageSize);
    cudaMalloc((void**)&d_structuringElement, elementSize);

    cudaMemcpy(d_inputImage, h_inputImage, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_structuringElement, h_structuringElement, elementSize, cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((imageWidth + TILE_SIZE - 1) / TILE_SIZE, 
                  (imageHeight + TILE_SIZE - 1) / TILE_SIZE);

    dilationKernel<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, d_structuringElement, imageWidth, imageHeight, elementWidth);
    cudaDeviceSynchronize();

    cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaFree(d_structuringElement);
}

void applyErosion(const unsigned char* h_inputImage, unsigned char* h_outputImage, 
                  const int* h_structuringElement, int imageWidth, int imageHeight, int elementWidth) {
    unsigned char *d_inputImage, *d_outputImage;
    int *d_structuringElement;
    int imageSize = imageWidth * imageHeight * sizeof(unsigned char);
    int elementSize = elementWidth * elementWidth * sizeof(int);

    cudaMalloc((void**)&d_inputImage, imageSize);
    cudaMalloc((void**)&d_outputImage, imageSize);
    cudaMalloc((void**)&d_structuringElement, elementSize);

    cudaMemcpy(d_inputImage, h_inputImage, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_structuringElement, h_structuringElement, elementSize, cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((imageWidth + TILE_SIZE - 1) / TILE_SIZE, 
                  (imageHeight + TILE_SIZE - 1) / TILE_SIZE);

    erosionKernel<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, d_structuringElement, imageWidth, imageHeight, elementWidth);
    cudaDeviceSynchronize();

    cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaFree(d_structuringElement);
}
