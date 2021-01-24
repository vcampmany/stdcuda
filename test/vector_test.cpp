#include "vector.h"

#include <catch2/catch.hpp>

TEST_CASE("Vector constructors and assigment operators",
          "[vector, copy, move, constructor]")
{
  std::vector<float> h_vec = {1.0, 2.0, 3.0};
  SECTION("Construct from a host vector")
  {
    stdcuda::Vector<float> d_vec(h_vec);
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, d_vec.data());
    REQUIRE(err == cudaSuccess);
    REQUIRE(attributes.type == cudaMemoryTypeDevice);
    REQUIRE(d_vec.size() == h_vec.size());
    REQUIRE(h_vec == d_vec.toStdVector());

    SECTION("Copy constructor")
    {
      stdcuda::Vector<float> d_vec_copy = d_vec;
      REQUIRE(d_vec.data() != d_vec_copy.data());
      REQUIRE(d_vec.size() == d_vec_copy.size());
      REQUIRE(d_vec.toStdVector() == d_vec_copy.toStdVector());
    }

    SECTION("Assignment operator")
    {
      std::vector<float> other_vec = {32.0F, 1.0F};
      stdcuda::Vector<float> d_other_vec(other_vec);
      d_other_vec = d_vec;
      REQUIRE(d_other_vec.data() != d_vec.data());
      REQUIRE(d_other_vec.size() == d_vec.size());
      REQUIRE(d_other_vec.toStdVector() == d_vec.toStdVector());
    }

    SECTION("Move")
    {
      stdcuda::Vector<float> d_moved_vec(h_vec);
      const float* d_ptr = d_moved_vec.data();
      const size_t d_size = d_moved_vec.size();
      SECTION("Move constructor")
      {
        stdcuda::Vector<float> d_receiver_vec(std::move(d_moved_vec));
        REQUIRE(d_moved_vec.data() == nullptr);  // NOLINT
        REQUIRE(d_moved_vec.size() == 0);
        REQUIRE(d_receiver_vec.data() == d_ptr);
        REQUIRE(d_receiver_vec.size() == d_size);
      }
      SECTION("Move assigment operator")
      {
        d_vec = std::move(d_moved_vec);
        REQUIRE(d_moved_vec.data() == nullptr);  // NOLINT
        REQUIRE(d_moved_vec.size() == 0);
        REQUIRE(d_vec.data() == d_ptr);
        REQUIRE(d_vec.size() == d_size);
      }
    }
  }
}
