cmake_minimum_required(VERSION 3.22)
project(ir_rgb_fusion_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "75;86")
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc features2d calib3d)
# Distro nvidia-cuda-toolkit puts headers in /usr/include, not /usr/local/cuda
if(NOT DEFINED CUDAToolkit_INCLUDE_DIR AND EXISTS "/usr/include/cuda_runtime.h")
  set(CUDAToolkit_INCLUDE_DIR "/usr/include")
endif()
find_package(CUDAToolkit REQUIRED)

# CUDA kernel library
add_library(fusion_kernel STATIC src/fusion_kernel.cu)
target_include_directories(fusion_kernel PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(fusion_kernel PUBLIC CUDA::cudart)
set_target_properties(fusion_kernel PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

# Main executable
add_executable(fusion_node
  src/fusion_node.cpp
  src/fusion_pipeline.cpp
)
target_include_directories(fusion_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(fusion_node
  fusion_kernel
  ${OpenCV_LIBS}
  CUDA::cudart
)
ament_target_dependencies(fusion_node
  rclcpp
  sensor_msgs
  message_filters
  rcl_interfaces
)

install(TARGETS fusion_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()
