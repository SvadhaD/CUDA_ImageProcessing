#ifndef MORPHOLOGY_CUH
#define MORPHOLOGY_CUH

#include <opencv2/opencv.hpp>
#include <vector>

void cpu_erosion(const cv::Mat& inputImage, cv::Mat& outputImage, const std::vector<int>& kernel);
void cpu_dilation(const cv::Mat& inputImage, cv::Mat& outputImage, const std::vector<int>& kernel);

void gpu_erosion(const uchar* input, uchar* output, int width, int height, const int* kernel, int kernelSize);
void gpu_dilation(const uchar* input, uchar* output, int width, int height, const int* kernel, int kernelSize);

#endif

