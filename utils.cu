#include "utils.cuh"

void* cudaMallocSafe(size_t size) {
    void* devPtr;
    checkCudaError(cudaMalloc(&devPtr, size), "cudaMalloc failed");
    return devPtr;
}

void cudaMemcpySafe(void* dst, const void* src, size_t size, cudaMemcpyKind kind) {
    checkCudaError(cudaMemcpy(dst, src, size, kind), "cudaMemcpy failed");
}

void cudaFreeSafe(void* devPtr) {
    checkCudaError(cudaFree(devPtr), "cudaFree failed");
}
