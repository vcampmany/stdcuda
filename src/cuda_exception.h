#ifndef STDCUDA_CUDA_EXCEPTION_H
#define STDCUDA_CUDA_EXCEPTION_H

#include <string>

namespace stdcuda {

class CudaException : public std::exception {
 public:
  explicit CudaException(cudaError_t error_code) : error_code_(error_code)
  {
    msg_ = ("Cuda Error " + std::to_string(error_code_) + ": " +
            std::string(cudaGetErrorString(error_code_)))
               .c_str();
  }
  const char* what() const noexcept override
  {
    return msg_;
  }

 private:
  cudaError_t error_code_;
  const char* msg_;
};

void cudaSuccess(const cudaError_t error)
{
  if (error != cudaError::cudaSuccess) {
    throw CudaException(error);
  }
}

}  // namespace stdcuda

#endif  // STDCUDA_CUDA_EXCEPTION_H
