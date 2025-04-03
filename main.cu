#include "convolution.cuh"
#include "morphology.cuh"
#include "utils.cuh"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cuda_runtime.h>

using namespace cv;
using namespace std;

void comparePerformance(const Mat& inputImage, int kernelSize) {
    Mat cpuOutput(inputImage.size(), CV_8UC1);
    Mat gpuOutput(inputImage.size(), CV_8UC1);
    vector<float> kernel(kernelSize * kernelSize, 1.0f / (kernelSize * kernelSize));
    
    // CPU Convolution
    auto start = chrono::high_resolution_clock::now();
    filter2D(inputImage, cpuOutput, -1, Mat(kernelSize, kernelSize, CV_32F, kernel.data()));
    auto end = chrono::high_resolution_clock::now();
    cout << "CPU Convolution Time: " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
    
    // GPU Convolution
    start = chrono::high_resolution_clock::now();
    applyConvolution(inputImage.data, gpuOutput.data, kernel.data(), inputImage.cols, inputImage.rows, kernelSize);
    end = chrono::high_resolution_clock::now();
    cout << "GPU Convolution Time: " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
    
    imwrite("cpu_convolution.png", cpuOutput);
    imwrite("gpu_convolution.png", gpuOutput);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <image_path>" << endl;
        return -1;
    }
    
    Mat inputImage = imread(argv[1], IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        cerr << "Error: Could not open image!" << endl;
        return -1;
    }
    
    Mat dilatedImage(inputImage.size(), CV_8UC1);
    Mat erodedImage(inputImage.size(), CV_8UC1);
    
    int structuringElement[9] = { 0, 1, 0, 1, 1, 1, 0, 1, 0 };
    
    applyDilation(inputImage.data, dilatedImage.data, structuringElement, inputImage.cols, inputImage.rows, 3);
    applyErosion(inputImage.data, erodedImage.data, structuringElement, inputImage.cols, inputImage.rows, 3);
    
    imwrite("dilated.png", dilatedImage);
    imwrite("eroded.png", erodedImage);
    
    comparePerformance(inputImage, 3);
    
    cout << "Processing complete. Output images saved." << endl;
    return 0;
}
