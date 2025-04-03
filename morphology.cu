#include "morphology.cuh"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>

// CPU implementation of erosion
void cpu_erosion(const cv::Mat& inputImage, cv::Mat& outputImage, const std::vector<int>& kernel) {
    int kernelSize = 3;
    int offset = kernelSize / 2;
    
    for (int i = offset; i < inputImage.rows - offset; ++i) {
        for (int j = offset; j < inputImage.cols - offset; ++j) {
            uchar minValue = 255;
            for (int ki = 0; ki < kernelSize; ++ki) {
                for (int kj = 0; kj < kernelSize; ++kj) {
                    int ni = i + ki - offset;
                    int nj = j + kj - offset;
                    if (kernel[ki * kernelSize + kj] == 1) {
                        minValue = std::min(minValue, inputImage.at<uchar>(ni, nj));
                    }
                }
            }
            outputImage.at<uchar>(i, j) = minValue;
        }
    }
}

// CPU implementation of dilation
void cpu_dilation(const cv::Mat& inputImage, cv::Mat& outputImage, const std::vector<int>& kernel) {
    int kernelSize = 3; 
    int offset = kernelSize / 2;
    
    for (int i = offset; i < inputImage.rows - offset; ++i) {
        for (int j = offset; j < inputImage.cols - offset; ++j) {
            uchar maxValue = 0;
            for (int ki = 0; ki < kernelSize; ++ki) {
                for (int kj = 0; kj < kernelSize; ++kj) {
                    int ni = i + ki - offset;
                    int nj = j + kj - offset;
                    if (kernel[ki * kernelSize + kj] == 1) {
                        maxValue = std::max(maxValue, inputImage.at<uchar>(ni, nj));
                    }
                }
            }
            outputImage.at<uchar>(i, j) = maxValue;
        }
    }
}

// GPU kernel for erosion
__global__ void gpu_erosion_kernel(const uchar* input, uchar* output, int width, int height, const uchar* kernel, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int offset = kernelSize / 2;

    if (x >= offset && x < width - offset && y >= offset && y < height - offset) {
        uchar minValue = 255;

        for (int ki = -offset; ki <= offset; ++ki) {
            for (int kj = -offset; kj <= offset; ++kj) {
                int ni = y + ki; 
                int nj = x + kj;

                if (kernel[(ki + offset) * kernelSize + (kj + offset)] == 1) {
                    minValue = min(minValue, input[ni * width + nj]);
                }
            }
        }

        output[y * width + x] = minValue;
    }
}

// GPU kernel for dilation
__global__ void gpu_dilation_kernel(const uchar* input, uchar* output, int width, int height, const uchar* kernel, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int offset = kernelSize / 2;

    if (x >= offset && x < width - offset && y >= offset && y < height - offset) {
        uchar maxValue = 0;

        for (int ki = -offset; ki <= offset; ++ki) {
            for (int kj = -offset; kj <= offset; ++kj) {
                int ni = y + ki; 
                int nj = x + kj; 

                if (kernel[(ki + offset) * kernelSize + (kj + offset)] == 1) {
                    maxValue = max(maxValue, input[ni * width + nj]);
                }
            }
        }

        output[y * width + x] = maxValue;
    }
}

// GPU erosion wrapper
void gpu_erosion(const uchar* input, uchar* output, int width, int height, const int* kernel, int kernelSize) {
    uchar *d_input, *d_output, *d_kernel;
    size_t imageSize = width * height * sizeof(uchar);
    size_t kernelSizeBytes = kernelSize * kernelSize * sizeof(int);

    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);
    cudaMalloc(&d_kernel, kernelSizeBytes);
    
    cudaMemcpy(d_input, input, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSizeBytes, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    gpu_erosion_kernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, d_kernel, kernelSize);
    
    cudaMemcpy(output, d_output, imageSize, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}

// GPU dilation wrapper
void gpu_dilation(const uchar* input, uchar* output, int width, int height, const int* kernel, int kernelSize) {
    uchar *d_input, *d_output, *d_kernel;
    size_t imageSize = width * height * sizeof(uchar);
    size_t kernelSizeBytes = kernelSize * kernelSize * sizeof(int);

    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);
    cudaMalloc(&d_kernel, kernelSizeBytes);
    
    cudaMemcpy(d_input, input, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSizeBytes, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    gpu_dilation_kernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, d_kernel, kernelSize);
    
    cudaMemcpy(output, d_output, imageSize, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}
