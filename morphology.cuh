#ifndef MORPHOLOGY_CUH
#define MORPHOLOGY_CUH

#include <cuda_runtime.h>
#include "utils.cuh"

// Kernel for dilation operation
__global__ void dilationKernel(const unsigned char* inputImage, unsigned char* outputImage, 
                               const int* structuringElement, int imageWidth, int imageHeight, 
                               int elementWidth);

// Kernel for erosion operation
__global__ void erosionKernel(const unsigned char* inputImage, unsigned char* outputImage, 
                              const int* structuringElement, int imageWidth, int imageHeight, 
                              int elementWidth);

// Host function to apply dilation
void applyDilation(const unsigned char* h_inputImage, unsigned char* h_outputImage, 
                   const int* h_structuringElement, int imageWidth, int imageHeight, int elementWidth);

// Host function to apply erosion
void applyErosion(const unsigned char* h_inputImage, unsigned char* h_outputImage, 
                  const int* h_structuringElement, int imageWidth, int imageHeight, int elementWidth);

#endif
