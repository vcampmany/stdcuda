find_package(CUDAToolkit)
if (NOT ${CUDAToolkit_FOUND})
    message(FATAL_ERROR "CUDA toolkit not found")
endif ()

add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(
        ${PROJECT_NAME}
        INTERFACE
        ${CUDAToolkit_INCLUDE_DIRS}
        "${PROJECT_SOURCE_DIR}/src"
        )

target_link_libraries(
        ${PROJECT_NAME}
        INTERFACE
        CUDA::cudart
        )
