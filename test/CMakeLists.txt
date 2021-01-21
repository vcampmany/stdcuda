set(SRCS
        test_main.cpp
        vector_test.cpp
        )

set(UNIT_TESTS ${PROJECT_NAME}_test)
add_executable(${UNIT_TESTS})
target_sources(
        ${UNIT_TESTS}
        PRIVATE
        ${SRCS}
        )


target_link_libraries(${UNIT_TESTS}
        PRIVATE
        ${CONAN_LIBS}
        ${PROJECT_NAME}
        )