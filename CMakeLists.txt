cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

# ==============================================================================
# 1. COMPILER & ARCHITECTURE SETUP
# ==============================================================================
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)

# Set default CUDA Architectures for PyTorch if not already provided by env.
# 7.5 = Turing (RTX 20xx, T4)
# 8.0 = Ampere (A100)
# 8.6 = Ampere (RTX 30xx, A10, A40)
# 8.9 = Ada Lovelace (RTX 40xx, L4)
# 9.0 = Hopper (H100) + PTX for forward compatibility
if(NOT DEFINED ENV{TORCH_CUDA_ARCH_LIST})
    set(ENV{TORCH_CUDA_ARCH_LIST} "7.5;8.0;8.6;8.9;9.0+PTX")
    message(STATUS "⚙️  Setting Default PyTorch Architectures: $ENV{TORCH_CUDA_ARCH_LIST}")
endif()

project(torch_nvidia_of_sdk LANGUAGES CXX CUDA)

# ==============================================================================
# 2. LOCATE PYTHON & BUNDLED LIBRARIES
# ==============================================================================
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# A. Find PyTorch Prefix
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c 
    "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")

# B. Find Nvidia Runtime
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c 
    "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE PYTHON_SITE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(BUNDLED_CUDA_ROOT "${PYTHON_SITE}/nvidia/cuda_runtime")

if(NOT EXISTS "${BUNDLED_CUDA_ROOT}")
    message(FATAL_ERROR "Could not find nvidia-cuda-runtime in ${PYTHON_SITE}.")
endif()

message(STATUS "✅ Using Bundled CUDA Root: ${BUNDLED_CUDA_ROOT}")

# ==============================================================================
# 3. DEPENDENCIES
# ==============================================================================
find_package(Torch REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED) 

# ==============================================================================
# 4. SOURCE FILES
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
# 5. BUILD TARGETS
# ==============================================================================

# --- Internal Static Library ---
add_library(nvof_cuda_kernels STATIC ${NVOF_CUDA_SOURCES})

target_include_directories(nvof_cuda_kernels PRIVATE
    "${BUNDLED_CUDA_ROOT}/include"
    ${CUDA_INCLUDE_DIRS}
    ${NVOF_SDK_ROOT}/Common/NvOFBase
    ${NVOF_SDK_ROOT}/Common/Utils
    ${NVOF_SDK_ROOT}/NvOFInterface
)

set_target_properties(nvof_cuda_kernels PROPERTIES
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF 
    POSITION_INDEPENDENT_CODE ON
)

# --- PORTABILITY FIX: Find libcudart.so.12 explicitly ---
# Use file(GLOB) to find the file physically, ignoring CMake's version checks
file(GLOB CUDART_CANDIDATES "${BUNDLED_CUDA_ROOT}/lib/libcudart.so*")

if(CUDART_CANDIDATES)
    list(GET CUDART_CANDIDATES 0 BUNDLED_CUDART_LIB)
    message(STATUS "✅ Linking against Bundled Runtime: ${BUNDLED_CUDART_LIB}")
else()
    message(FATAL_ERROR "❌ Could not find 'libcudart.so*' in ${BUNDLED_CUDA_ROOT}/lib. Check installation.")
endif()

target_link_libraries(nvof_cuda_kernels PRIVATE ${BUNDLED_CUDART_LIB})


# --- Python Extension ---
pybind11_add_module(nvof_torch MODULE src/nvof_torch.cpp ${NVOF_SOURCES})

set_target_properties(nvof_torch PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/of/methods"
    CXX_STANDARD 17
)

target_include_directories(nvof_torch PRIVATE
    "${BUNDLED_CUDA_ROOT}/include"
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
    ${BUNDLED_CUDART_LIB}
    nvof_cuda_kernels)

# Link against libtorch_python.so
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH)
if(UNIX AND NOT APPLE)
    target_link_libraries(nvof_torch PRIVATE "-Wl,--no-as-needed" "${TORCH_PYTHON_LIBRARY}" "-Wl,--as-needed")
else()
    target_link_libraries(nvof_torch PRIVATE "${TORCH_PYTHON_LIBRARY}")
endif()

# ==============================================================================
# 6. RPATH (Runtime Loading)
# ==============================================================================
if(UNIX AND NOT APPLE)
    set_target_properties(nvof_torch PROPERTIES
        INSTALL_RPATH "$ORIGIN/../../torch/lib:$ORIGIN/../../nvidia/cuda_runtime/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

install(TARGETS nvof_torch DESTINATION of/methods)