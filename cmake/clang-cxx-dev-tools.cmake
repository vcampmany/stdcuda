# Code based on https://www.labri.fr/perso/fleury/posts/programming/using-clang-tidy-and-clang-format.html
set(SOURCE_DIR src)
set(TEST_DIR test)
file(GLOB_RECURSE
        ALL_CXX_SOURCE_FILES
        ${PROJECT_SOURCE_DIR}/*.cu
        ${PROJECT_SOURCE_DIR}/*.cuh
        ${PROJECT_SOURCE_DIR}/*.[chi]pp
        ${PROJECT_SOURCE_DIR}/*.[chi]xx
        ${PROJECT_SOURCE_DIR}/*.cc
        ${PROJECT_SOURCE_DIR}/*.hh
        ${PROJECT_SOURCE_DIR}/*.ii
        ${PROJECT_SOURCE_DIR}/*.[CHI]
        )

find_program(CLANG_FORMAT "clang-format")
if (CLANG_FORMAT)
    add_custom_target(
            clang-format
            COMMAND /usr/bin/clang-format
            -i
            -style=file
            ${ALL_CXX_SOURCE_FILES}
    )
endif ()

find_program(CLANG_TIDY "clang-tidy")
if (CLANG_TIDY)
    add_custom_target(
            clang-tidy
            COMMAND ${CMAKE_SOURCE_DIR}/tools/run-clang-tidy.py
            -p ${CMAKE_BINARY_DIR}
            \${JOBS}
    )
endif ()