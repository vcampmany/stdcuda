#ifndef STDCUDA_VECTOR_H
#define STDCUDA_VECTOR_H

#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

namespace stdcuda {

void checkError(const cudaError_t error)
{
  if (error != cudaError::cudaSuccess) {
    throw std::runtime_error("cuda error");
  }
}

template <typename T>
class Vector {
 public:
  explicit Vector(const std::vector<T>& vec) : size_(vec.size())
  {
    checkError(cudaMalloc(&ptr_, size_ * sizeof(T)));
    checkError(cudaMemcpy(ptr_, vec.data(), size_ * sizeof(T),
                          cudaMemcpyHostToDevice));
  }

  Vector(const Vector<T>&) = delete;
  Vector<T>& operator=(const Vector<T>&) = delete;
  Vector(Vector<T>&&) noexcept = default;
  Vector<T>& operator=(Vector<T>&&) noexcept = default;
  ~Vector()
  {
    checkError(cudaFree(ptr_));
  }

  T* getDevicePtr() const
  {
    return ptr_;
  }

 private:
  T* ptr_{nullptr};
  size_t size_{0};
};

}  // namespace stdcuda

#endif  // STDCUDA_VECTOR_H
