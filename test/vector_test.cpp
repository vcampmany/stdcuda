#include "vector.h"

#include <catch2/catch.hpp>

TEST_CASE("Vector constructor from std::vector", "[vector, std]")
{
  std::vector<float> h_vals = {1.0, 2.0, 3.0};
  stdcuda::Vector<float> vals(h_vals);
  cudaPointerAttributes attributes;
  cudaError_t err = cudaPointerGetAttributes(&attributes, vals.getDevicePtr());
  REQUIRE(err == cudaSuccess);
  REQUIRE(attributes.type == cudaMemoryTypeDevice);
}