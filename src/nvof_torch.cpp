#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAFunctions.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <iostream>
#include <unordered_map>

#include "NvOFUtils.h"
#include "NvOFCuda.h"

// --- Helper Maps ---
std::unordered_map<std::string, NV_OF_PERF_LEVEL> presetMap = {
    {"slow", NV_OF_PERF_LEVEL_SLOW},
    {"medium", NV_OF_PERF_LEVEL_MEDIUM},
    {"fast", NV_OF_PERF_LEVEL_FAST}};

class TorchNVOpticalFlow
{
private:
    std::unique_ptr<NvOF> m_nvOpticalFlow;
    std::unique_ptr<NvOFUtilsCuda> m_nvofUtils; // Upsampling utility
    CUcontext m_cuContext = nullptr;

    int m_width;
    int m_height;
    int m_gpu_id;
    std::string m_preset;
    int m_grid_size;
    uint32_t m_hwGridSize;
    bool m_bidirectional;
    int m_out_h;
    int m_out_w;

    // Pre-allocated SDK buffers (reused across calls)
    std::vector<NvOFBufferObj> m_inputBuffers;
    std::vector<NvOFBufferObj> m_outputBuffers;
    std::vector<NvOFBufferObj> m_bwdOutputBuffers;          // For bidirectional mode
    std::vector<NvOFBufferObj> m_upsampledOutputBuffers;    // For upsampled output (allocated on demand)
    std::vector<NvOFBufferObj> m_upsampledBwdOutputBuffers; // For upsampled backward output (allocated on demand)

public:
    TorchNVOpticalFlow(
        int width,
        int height,
        int gpu_id = 0,
        std::string preset = "medium",
        int grid_size = 1, // 1, 2 or 4
        bool bidirectional = false)
        : m_width(width), m_height(height), m_gpu_id(gpu_id), m_preset(preset), m_grid_size(grid_size), m_bidirectional(bidirectional)
    {
        // 1. Context Management (CRITICAL FIX)
        // Get the Primary Context that PyTorch is already using for this GPU.

        // Ensure the device is active
        cudaSetDevice(gpu_id);

        // Retrieve the current context handle
        CUDA_DRVAPI_CALL(cuCtxGetCurrent(&m_cuContext));

        if (m_cuContext == nullptr)
        {
            // Fallback: If no context is active, force initialization of the primary context
            CUdevice device;
            CUDA_DRVAPI_CALL(cuDeviceGet(&device, gpu_id));
            CUDA_DRVAPI_CALL(cuDevicePrimaryCtxRetain(&m_cuContext, device));
            CUDA_DRVAPI_CALL(cuCtxPushCurrent(m_cuContext));
        }

        if (presetMap.find(preset) == presetMap.end())
        {
            TORCH_CHECK(false, "Invalid preset: ", preset);
        }
        NV_OF_PERF_LEVEL nv_preset = presetMap.at(preset);

        m_nvOpticalFlow = NvOFCuda::Create(
            m_cuContext,
            width,
            height,
            NV_OF_BUFFER_FORMAT_ABGR8, // Standard 4-channel input
            NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,
            NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,
            NV_OF_MODE_OPTICALFLOW,
            nv_preset,
            nullptr, // inputStream (we will manage streams later if needed)
            nullptr  // outputStream
        );

        // Grid side ultimately controlled by the SDK upon GPU version and preset
        if (!m_nvOpticalFlow->CheckGridSize(grid_size))
        {
            if (!m_nvOpticalFlow->GetNextMinGridSize(grid_size, m_hwGridSize))
            {
                throw std::runtime_error("Invalid grid size for this GPU/Preset");
            }
        }
        else
        {
            m_hwGridSize = grid_size;
        }
        m_out_h = (m_height + m_hwGridSize - 1) / m_hwGridSize;
        m_out_w = (m_width + m_hwGridSize - 1) / m_hwGridSize;

        // Initialize with negotiated grid siz
        m_nvOpticalFlow->Init(m_hwGridSize, m_hwGridSize, false, false, m_bidirectional);

        m_nvofUtils = std::make_unique<NvOFUtilsCuda>(NV_OF_MODE_OPTICALFLOW);

        // Pre-allocate SDK-managed buffers
        // Unfortunately, the SDK does not support using pre-allocated buffers for CUDA
        // Which would be ideal for zero-copy with PyTorch tensors.
        m_inputBuffers = m_nvOpticalFlow->CreateBuffers(NV_OF_BUFFER_USAGE_INPUT, 2);
        m_outputBuffers = m_nvOpticalFlow->CreateBuffers(NV_OF_BUFFER_USAGE_OUTPUT, 1);

        if (m_bidirectional)
        {
            m_bwdOutputBuffers = m_nvOpticalFlow->CreateBuffers(NV_OF_BUFFER_USAGE_OUTPUT, 1);
        }
    }

    ~TorchNVOpticalFlow()
    {

        m_inputBuffers.clear();
        m_outputBuffers.clear();
        m_bwdOutputBuffers.clear();
        m_upsampledOutputBuffers.clear();
        m_upsampledBwdOutputBuffers.clear();
        m_nvOpticalFlow.reset();
        m_nvofUtils.reset();
        // m_cuContext is managed by PyTorch
    }
    std::vector<int64_t> output_shape() const
    {
        return {m_out_h * m_hwGridSize, m_out_w * m_hwGridSize, 2};
    }
    int gpu_id() const
    {
        return m_gpu_id;
    }

    // --- Compute Flow Method ---
    torch::Tensor compute_flow(torch::Tensor input, torch::Tensor reference, bool upsample = false)
    {
        // 1. Input Validation
        TORCH_CHECK(input.is_cuda(), "Input tensor must be CUDA");
        TORCH_CHECK(reference.is_cuda(), "Reference tensor must be CUDA");
        TORCH_CHECK(input.device().index() == m_gpu_id, "Input tensor on wrong GPU");

        // Validate Shape: [H, W, 4] for ABGR8
        TORCH_CHECK(input.size(-1) == 4, "Input must have 4 channels (RGBA/ABGR). Got: ", input.size(-1));
        TORCH_CHECK(input.scalar_type() == torch::kUInt8, "Input must be uint8");
        TORCH_CHECK(input.size(0) == m_height && input.size(1) == m_width,
                    "Input size mismatch. Expected ", m_height, "x", m_width);

        auto in_contig = input.contiguous();
        auto ref_contig = reference.contiguous();

        // 2. Get PyTorch stream and synchronize
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream(m_gpu_id).stream();

        // 3. GPU-to-GPU copy from PyTorch tensors to SDK buffers
        // Cast to NvOFBufferCudaDevicePtr to access getCudaDevicePtr() and getStrideInfo()
        auto *sdkBuf0 = dynamic_cast<NvOFBufferCudaDevicePtr *>(m_inputBuffers[0].get());
        auto *sdkBuf1 = dynamic_cast<NvOFBufferCudaDevicePtr *>(m_inputBuffers[1].get());
        auto *sdkOutBuf = dynamic_cast<NvOFBufferCudaDevicePtr *>(m_outputBuffers[0].get());

        TORCH_CHECK(sdkBuf0 && sdkBuf1 && sdkOutBuf, "SDK buffers must be CUdeviceptr type");

        // Copy input tensor to SDK buffer 0
        {
            CUDA_MEMCPY2D copyParams = {};
            copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            copyParams.srcDevice = reinterpret_cast<CUdeviceptr>(in_contig.data_ptr());
            copyParams.srcPitch = m_width * 4; // PyTorch tensor is tightly packed
            copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            copyParams.dstDevice = sdkBuf0->getCudaDevicePtr();
            copyParams.dstPitch = sdkBuf0->getStrideInfo().strideInfo[0].strideXInBytes;
            copyParams.WidthInBytes = m_width * 4;
            copyParams.Height = m_height;
            CUDA_DRVAPI_CALL(cuMemcpy2DAsync(&copyParams, (CUstream)stream));
        }

        // Copy reference tensor to SDK buffer 1
        {
            CUDA_MEMCPY2D copyParams = {};
            copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            copyParams.srcDevice = reinterpret_cast<CUdeviceptr>(ref_contig.data_ptr());
            copyParams.srcPitch = m_width * 4;
            copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            copyParams.dstDevice = sdkBuf1->getCudaDevicePtr();
            copyParams.dstPitch = sdkBuf1->getStrideInfo().strideInfo[0].strideXInBytes;
            copyParams.WidthInBytes = m_width * 4;
            copyParams.Height = m_height;
            CUDA_DRVAPI_CALL(cuMemcpy2DAsync(&copyParams, (CUstream)stream));
        }

        // Sync before execute (SDK may use different stream)
        cudaStreamSynchronize(stream);

        // 4. Execute optical flow
        m_nvOpticalFlow->Execute(
            m_inputBuffers[0].get(),
            m_inputBuffers[1].get(),
            m_outputBuffers[0].get(),
            nullptr // hints
        );

        // Determine output buffer to copy from
        NvOFBufferCudaDevicePtr *finalOutBuf = sdkOutBuf;
        int final_out_h = m_out_h;
        int final_out_w = m_out_w;

        // 5. Upsample if requested and grid_size > 1
        if (upsample && m_hwGridSize > 1)
        {
            // Allocate upsampled buffer on first use (lazy initialization)
            // Use overload with explicit dimensions for full-resolution output
            if (m_upsampledOutputBuffers.empty())
            {
                m_upsampledOutputBuffers = m_nvOpticalFlow->CreateBuffers(m_width, m_height, NV_OF_BUFFER_USAGE_OUTPUT, 1);
            }

            auto *upsampledBuf = dynamic_cast<NvOFBufferCudaDevicePtr *>(m_upsampledOutputBuffers[0].get());
            TORCH_CHECK(upsampledBuf, "Upsampled buffer must be CUdeviceptr type");

            // SDK's hardware-accelerated upsampling (includes magnitude scaling)
            m_nvofUtils->Upsample(m_outputBuffers[0].get(), m_upsampledOutputBuffers[0].get(), m_hwGridSize);

            finalOutBuf = upsampledBuf;
            final_out_h = m_height;
            final_out_w = m_width;
        }

        auto options = torch::TensorOptions()
                           .dtype(torch::kInt16)
                           .layout(torch::kStrided)
                           .device(torch::kCUDA, m_gpu_id);

        torch::Tensor output = torch::empty({final_out_h, final_out_w, 2}, options);

        // 6. GPU-to-GPU copy from SDK output buffer to PyTorch tensor
        {
            CUDA_MEMCPY2D copyParams = {};
            copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            copyParams.srcDevice = finalOutBuf->getCudaDevicePtr();
            copyParams.srcPitch = finalOutBuf->getStrideInfo().strideInfo[0].strideXInBytes;
            copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            copyParams.dstDevice = reinterpret_cast<CUdeviceptr>(output.data_ptr());
            copyParams.dstPitch = final_out_w * sizeof(NV_OF_FLOW_VECTOR); // Tightly packed
            copyParams.WidthInBytes = final_out_w * sizeof(NV_OF_FLOW_VECTOR);
            copyParams.Height = final_out_h;
            CUDA_DRVAPI_CALL(cuMemcpy2DAsync(&copyParams, (CUstream)stream));
        }

        // Sync to ensure output is ready
        cudaStreamSynchronize(stream);

        return output;
    }

    // --- Compute Flow Bidirectional Method ---
    std::tuple<torch::Tensor, torch::Tensor> compute_flow_bidirectional(torch::Tensor input, torch::Tensor reference, bool upsample = false)
    {
        TORCH_CHECK(m_bidirectional, "Engine must be initialized with bidirectional=True");

        // 1. Input Validation
        TORCH_CHECK(input.is_cuda(), "Input tensor must be CUDA");
        TORCH_CHECK(reference.is_cuda(), "Reference tensor must be CUDA");
        TORCH_CHECK(input.device().index() == m_gpu_id, "Input tensor on wrong GPU");

        // Validate Shape: [H, W, 4] for ABGR8
        TORCH_CHECK(input.size(-1) == 4, "Input must have 4 channels (RGBA/ABGR). Got: ", input.size(-1));
        TORCH_CHECK(input.scalar_type() == torch::kUInt8, "Input must be uint8");
        TORCH_CHECK(input.size(0) == m_height && input.size(1) == m_width,
                    "Input size mismatch. Expected ", m_height, "x", m_width);

        // Ensure contiguous memory (row-major)
        auto in_contig = input.contiguous();
        auto ref_contig = reference.contiguous();

        // 2. Get PyTorch stream and synchronize
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream(m_gpu_id).stream();

        // 3. GPU-to-GPU copy from PyTorch tensors to SDK buffers
        auto *sdkBuf0 = dynamic_cast<NvOFBufferCudaDevicePtr *>(m_inputBuffers[0].get());
        auto *sdkBuf1 = dynamic_cast<NvOFBufferCudaDevicePtr *>(m_inputBuffers[1].get());
        auto *sdkOutBuf = dynamic_cast<NvOFBufferCudaDevicePtr *>(m_outputBuffers[0].get());
        auto *sdkBwdOutBuf = dynamic_cast<NvOFBufferCudaDevicePtr *>(m_bwdOutputBuffers[0].get());

        TORCH_CHECK(sdkBuf0 && sdkBuf1 && sdkOutBuf && sdkBwdOutBuf, "SDK buffers must be CUdeviceptr type");

        // Copy input tensor to SDK buffer 0
        {
            CUDA_MEMCPY2D copyParams = {};
            copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            copyParams.srcDevice = reinterpret_cast<CUdeviceptr>(in_contig.data_ptr());
            copyParams.srcPitch = m_width * 4;
            copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            copyParams.dstDevice = sdkBuf0->getCudaDevicePtr();
            copyParams.dstPitch = sdkBuf0->getStrideInfo().strideInfo[0].strideXInBytes;
            copyParams.WidthInBytes = m_width * 4;
            copyParams.Height = m_height;
            CUDA_DRVAPI_CALL(cuMemcpy2DAsync(&copyParams, (CUstream)stream));
        }

        // Copy reference tensor to SDK buffer 1
        {
            CUDA_MEMCPY2D copyParams = {};
            copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            copyParams.srcDevice = reinterpret_cast<CUdeviceptr>(ref_contig.data_ptr());
            copyParams.srcPitch = m_width * 4;
            copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            copyParams.dstDevice = sdkBuf1->getCudaDevicePtr();
            copyParams.dstPitch = sdkBuf1->getStrideInfo().strideInfo[0].strideXInBytes;
            copyParams.WidthInBytes = m_width * 4;
            copyParams.Height = m_height;
            CUDA_DRVAPI_CALL(cuMemcpy2DAsync(&copyParams, (CUstream)stream));
        }

        // Sync before execute
        cudaStreamSynchronize(stream);

        // 4. Execute bidirectional optical flow
        m_nvOpticalFlow->ExecuteBidirectional(
            m_inputBuffers[0].get(),
            m_inputBuffers[1].get(),
            m_outputBuffers[0].get(),
            m_bwdOutputBuffers[0].get(),
            nullptr // hints
        );

        // Determine output buffers to copy from
        NvOFBufferCudaDevicePtr *finalFwdBuf = sdkOutBuf;
        NvOFBufferCudaDevicePtr *finalBwdBuf = sdkBwdOutBuf;
        int final_out_h = m_out_h;
        int final_out_w = m_out_w;

        // 5. Upsample if requested and grid_size > 1
        if (upsample && m_hwGridSize > 1)
        {
            // Allocate upsampled buffers on first use (lazy initialization)
            // Use overload with explicit dimensions for full-resolution output
            if (m_upsampledOutputBuffers.empty())
            {
                m_upsampledOutputBuffers = m_nvOpticalFlow->CreateBuffers(m_width, m_height, NV_OF_BUFFER_USAGE_OUTPUT, 1);
            }
            if (m_upsampledBwdOutputBuffers.empty())
            {
                m_upsampledBwdOutputBuffers = m_nvOpticalFlow->CreateBuffers(m_width, m_height, NV_OF_BUFFER_USAGE_OUTPUT, 1);
            }

            auto *upsampledFwdBuf = dynamic_cast<NvOFBufferCudaDevicePtr *>(m_upsampledOutputBuffers[0].get());
            auto *upsampledBwdBuf = dynamic_cast<NvOFBufferCudaDevicePtr *>(m_upsampledBwdOutputBuffers[0].get());
            TORCH_CHECK(upsampledFwdBuf && upsampledBwdBuf, "Upsampled buffers must be CUdeviceptr type");

            // SDK's hardware-accelerated upsampling (includes magnitude scaling)
            m_nvofUtils->Upsample(m_outputBuffers[0].get(), m_upsampledOutputBuffers[0].get(), m_hwGridSize);
            m_nvofUtils->Upsample(m_bwdOutputBuffers[0].get(), m_upsampledBwdOutputBuffers[0].get(), m_hwGridSize);

            finalFwdBuf = upsampledFwdBuf;
            finalBwdBuf = upsampledBwdBuf;
            final_out_h = m_height;
            final_out_w = m_width;
        }

        auto options = torch::TensorOptions()
                           .dtype(torch::kInt16)
                           .layout(torch::kStrided)
                           .device(torch::kCUDA, m_gpu_id);

        torch::Tensor fwd_output = torch::empty({final_out_h, final_out_w, 2}, options);
        torch::Tensor bwd_output = torch::empty({final_out_h, final_out_w, 2}, options);

        // 6. GPU-to-GPU copy for forward flow
        {
            CUDA_MEMCPY2D copyParams = {};
            copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            copyParams.srcDevice = finalFwdBuf->getCudaDevicePtr();
            copyParams.srcPitch = finalFwdBuf->getStrideInfo().strideInfo[0].strideXInBytes;
            copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            copyParams.dstDevice = reinterpret_cast<CUdeviceptr>(fwd_output.data_ptr());
            copyParams.dstPitch = final_out_w * sizeof(NV_OF_FLOW_VECTOR);
            copyParams.WidthInBytes = final_out_w * sizeof(NV_OF_FLOW_VECTOR);
            copyParams.Height = final_out_h;
            CUDA_DRVAPI_CALL(cuMemcpy2DAsync(&copyParams, (CUstream)stream));
        }

        // 7. GPU-to-GPU copy for backward flow
        {
            CUDA_MEMCPY2D copyParams = {};
            copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            copyParams.srcDevice = finalBwdBuf->getCudaDevicePtr();
            copyParams.srcPitch = finalBwdBuf->getStrideInfo().strideInfo[0].strideXInBytes;
            copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            copyParams.dstDevice = reinterpret_cast<CUdeviceptr>(bwd_output.data_ptr());
            copyParams.dstPitch = final_out_w * sizeof(NV_OF_FLOW_VECTOR);
            copyParams.WidthInBytes = final_out_w * sizeof(NV_OF_FLOW_VECTOR);
            copyParams.Height = final_out_h;
            CUDA_DRVAPI_CALL(cuMemcpy2DAsync(&copyParams, (CUstream)stream));
        }

        // Sync to ensure output is ready
        cudaStreamSynchronize(stream);

        return std::make_tuple(fwd_output, bwd_output);
    }

    // --- Factory Methods (Cleaned) ---

    // Instance method for resizing logic
    std::unique_ptr<TorchNVOpticalFlow> initialize_if_needed(torch::Tensor input)
    {
        int h = input.size(-2); // Height
        int w = input.size(-1); // This is likely 4 (Channels) if shape is HWC, check your tensor layout!
        // Correction: If input is [H, W, 4], width is dim(1).
        if (input.dim() == 3 && input.size(2) == 4)
        {
            w = input.size(1);
            h = input.size(0);
        }
        else if (input.dim() == 4)
        {
            // NCHW or NHWC? Usually flow inputs are HWC for ABGR.
            // Assuming [B, H, W, 4] or single image [H, W, 4]
            TORCH_CHECK(false, "Please pass single image [H, W, 4]");
        }

        int gpu = input.get_device();

        if (this->m_width == w &&
            this->m_height == h &&
            this->m_gpu_id == gpu)
        {
            return nullptr; // No change needed
        }

        // Return new instance
        return std::make_unique<TorchNVOpticalFlow>(w, h, gpu, this->m_preset, this->m_grid_size, this->m_bidirectional);
    }

    // Static factory
    static std::unique_ptr<TorchNVOpticalFlow> from_tensor(
        torch::Tensor input, std::string preset = "medium", int grid_size = 1, bool bidirectional = false)
    {
        // Extract dims assuming [H, W, C]
        int h = input.size(0);
        int w = input.size(1);
        int gpu = input.get_device();
        return std::make_unique<TorchNVOpticalFlow>(w, h, gpu, preset, grid_size, bidirectional);
    }
};

PYBIND11_MODULE(nvof_torch, m)
{
    py::class_<TorchNVOpticalFlow>(m, "TorchNVOpticalFlow")
        // Constructor Binding
        .def(py::init<int, int, int, std::string, int, bool>(),
             py::arg("width"),
             py::arg("height"),
             py::arg("gpu_id") = 0,
             py::arg("preset") = "medium",
             py::arg("grid_size") = 1,
             py::arg("bidirectional") = false)

        .def("compute_flow", &TorchNVOpticalFlow::compute_flow,
             py::arg("input"),
             py::arg("reference"),
             py::arg("upsample") = true)

        .def("compute_flow_bidirectional", &TorchNVOpticalFlow::compute_flow_bidirectional,
             py::arg("input"),
             py::arg("reference"),
             py::arg("upsample") = true)

        .def("initialize_if_needed", &TorchNVOpticalFlow::initialize_if_needed,
             py::arg("input"))

        .def("gpu_id", &TorchNVOpticalFlow::gpu_id)

        .def("output_shape", &TorchNVOpticalFlow::output_shape);
}