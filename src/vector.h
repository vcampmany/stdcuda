#ifndef STDCUDA_VECTOR_H
#define STDCUDA_VECTOR_H

#include <cuda_runtime.h>

#include <vector>

#include "cudart_error.h"

namespace stdcuda {

template <typename T>
class Vector {
 public:
  explicit Vector(const std::vector<T>& vec)
      : size_(vec.size()), bytes_(vec.size() * sizeof(T))
  {
    checkError(cudaMalloc(&ptr_, bytes_));
    checkError(cudaMemcpy(ptr_, vec.data(), bytes_, cudaMemcpyHostToDevice));
  }

  Vector(const Vector<T>& vec) : size_(vec.size_), bytes_(vec.bytes_)
  {
    checkError(cudaMalloc(&ptr_, bytes_));
    checkError(cudaMemcpy(ptr_, vec.data(), bytes_, cudaMemcpyDeviceToDevice));
  }

  Vector<T>& operator=(const Vector<T>& vec)
  {
    if (this != &vec) {
      cudaFree(ptr_);
      size_ = vec.size_;
      bytes_ = vec.bytes_;
      checkError(cudaMalloc(&ptr_, bytes_));
      checkError(
          cudaMemcpy(ptr_, vec.data(), bytes_, cudaMemcpyDeviceToDevice));
    }
    return *this;
  }

  Vector(Vector<T>&& vec) noexcept
      : ptr_(vec.ptr_), size_(vec.size_), bytes_(vec.bytes_)
  {
    vec.ptr_ = nullptr;
    vec.size_ = 0;
    vec.bytes_ = 0;
  }

  Vector<T>& operator=(Vector<T>&& vec) noexcept
  {
    ptr_ = vec.ptr_;
    size_ = vec.size_;
    bytes_ = vec.bytes_;
    vec.ptr_ = nullptr;
    vec.size_ = 0;
    vec.bytes_ = 0;
    return *this;
  }

  ~Vector()
  {
    checkError(cudaFree(ptr_));
  }

  std::vector<T> toStdVector()
  {
    std::vector<T> h_vec;
    h_vec.resize(size_);
    checkError(cudaMemcpy(h_vec.data(), ptr_, bytes_, cudaMemcpyDeviceToHost));
    return h_vec;
  }

  T* data() const noexcept
  {
    return ptr_;
  }

  size_t size() const noexcept
  {
    return size_;
  }

 private:
  T* ptr_;
  size_t size_;
  size_t bytes_;
};

}  // namespace stdcuda

#endif  // STDCUDA_VECTOR_H
