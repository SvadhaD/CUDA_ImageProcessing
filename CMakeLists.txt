cmake_minimum_required(VERSION 3.18)
project(CUDA_Image_Processing LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CUDA_ARCHITECTURES 75 80 86)

add_executable(image_processing
    main.cu
    convolution.cu
    morphology.cu
    utils.cu
)

target_include_directories(image_processing PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(image_processing PRIVATE ${OpenCV_LIBS} cuda)

set_target_properties(image_processing PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}
)
