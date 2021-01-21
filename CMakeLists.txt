cmake_minimum_required(VERSION 3.17)
project(stdcuda LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_subdirectory(src)
add_subdirectory(test)