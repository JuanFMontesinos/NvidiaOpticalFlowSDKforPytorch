cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# Force CUDA 12.9 before project declaration
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)

project(csem_optical_flow LANGUAGES CXX CUDA)

# ==============================================================================
# DEPENDENCIES & ENVIRONMENT
# ==============================================================================
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Find Torch Path via Python
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_FIND_RESULT
)

if(NOT "${TORCH_FIND_RESULT}" STREQUAL "0")
    message(FATAL_ERROR "Failed to import torch from '${Python_EXECUTABLE}'.")
endif()

list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")

# Ensure CUDAToolkit finds CUDA 
set(CUDAToolkit_ROOT "/usr/local/cuda" CACHE PATH "CUDA toolkit root" FORCE)

find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(pybind11 CONFIG REQUIRED)


message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Toolkit Root: ${CUDAToolkit_ROOT}")
message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")

# ==============================================================================
# 2. NVIDIA SDK SETUP
# ==============================================================================
set(NVOF_SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/optical_flow_sdk") 
if(NOT EXISTS "${NVOF_SDK_ROOT}")
    message(FATAL_ERROR "SDK not found at ${NVOF_SDK_ROOT}")
endif()

set(NVOF_SOURCES
    ${NVOF_SDK_ROOT}/Common/NvOFBase/NvOF.cpp
    ${NVOF_SDK_ROOT}/Common/NvOFBase/NvOFCuda.cpp
    ${NVOF_SDK_ROOT}/Common/Utils/NvOFDataLoader.cpp
    ${NVOF_SDK_ROOT}/Common/Utils/NvOFUtils.cpp
    ${NVOF_SDK_ROOT}/Common/Utils/NvOFUtilsCuda.cpp
)

set(NVOF_CUDA_SOURCES
    ${NVOF_SDK_ROOT}/Common/Utils/kernel.cu
)

# ==============================================================================
# DEFINE TARGET
# ==============================================================================

# Create a CUDA library for the kernel
add_library(nvof_cuda_kernels STATIC ${NVOF_CUDA_SOURCES})
set_target_properties(nvof_cuda_kernels PROPERTIES
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(nvof_cuda_kernels PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${NVOF_SDK_ROOT}/Common/NvOFBase
    ${NVOF_SDK_ROOT}/Common/Utils
    ${NVOF_SDK_ROOT}/NvOFInterface
)
target_link_libraries(nvof_cuda_kernels PRIVATE CUDA::cudart)

pybind11_add_module(nvof_torch MODULE src/nvof_torch.cpp ${NVOF_SOURCES})


# This forces the .so file to appear inside the source tree during build for uv local installs
set_target_properties(nvof_torch PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/of/methods"
)

# Enforce C++17
set_property(TARGET nvof_torch PROPERTY CXX_STANDARD 17)

# ==============================================================================
# LINKING & INCLUDES
# ==============================================================================
target_include_directories(nvof_torch PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${NVOF_SDK_ROOT}/Common/NvOFBase
    ${NVOF_SDK_ROOT}/Common/Utils
    ${NVOF_SDK_ROOT}/NvOFInterface
    ${NVOF_SDK_ROOT}/Common/External
)

target_link_libraries(nvof_torch PRIVATE 
    ${TORCH_LIBRARIES} 
    CUDA::cuda_driver 
    CUDA::cudart
    nvof_cuda_kernels)

# Github Fix (Link against libtorch_python.so)
# https://github.com/pytorch/pytorch/issues/108041#issuecomment-1753895665
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH)

if(UNIX AND NOT APPLE)
    target_link_libraries(nvof_torch PRIVATE "-Wl,--no-as-needed" "${TORCH_PYTHON_LIBRARY}" "-Wl,--as-needed")
else()
    target_link_libraries(nvof_torch PRIVATE "${TORCH_PYTHON_LIBRARY}")
endif()


# Look for libtorch.so 2 folders up, inside torch/lib
if(UNIX AND NOT APPLE)
    set_target_properties(nvof_torch PROPERTIES
        INSTALL_RPATH "$ORIGIN/../../torch/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# Install to "of/methods" (relative to site-packages root)
install(TARGETS nvof_torch DESTINATION of/methods)