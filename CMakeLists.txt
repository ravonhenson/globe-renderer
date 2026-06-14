cmake_minimum_required(VERSION 3.20)

project(vulkan_triangle LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
)

FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
)

FetchContent_MakeAvailable(glfw glm)

find_package(Vulkan REQUIRED)
find_program(GLSLC_EXECUTABLE glslc REQUIRED HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/Bin32")

set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)

file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

add_custom_command(
    OUTPUT ${SHADER_OUTPUT_DIR}/sphere.vert.spv
    COMMAND ${GLSLC_EXECUTABLE} ${SHADER_SOURCE_DIR}/sphere.vert -o ${SHADER_OUTPUT_DIR}/sphere.vert.spv
    DEPENDS ${SHADER_SOURCE_DIR}/sphere.vert
    COMMENT "Compiling sphere.vert"
    VERBATIM
)

add_custom_command(
    OUTPUT ${SHADER_OUTPUT_DIR}/sphere.frag.spv
    COMMAND ${GLSLC_EXECUTABLE} ${SHADER_SOURCE_DIR}/sphere.frag -o ${SHADER_OUTPUT_DIR}/sphere.frag.spv
    DEPENDS ${SHADER_SOURCE_DIR}/sphere.frag
    COMMENT "Compiling sphere.frag"
    VERBATIM
)

add_custom_command(
    OUTPUT ${SHADER_OUTPUT_DIR}/wireframe.frag.spv
    COMMAND ${GLSLC_EXECUTABLE} ${SHADER_SOURCE_DIR}/wireframe.frag -o ${SHADER_OUTPUT_DIR}/wireframe.frag.spv
    DEPENDS ${SHADER_SOURCE_DIR}/wireframe.frag
    COMMENT "Compiling wireframe.frag"
    VERBATIM
)

add_custom_target(shaders ALL DEPENDS ${SHADER_OUTPUT_DIR}/sphere.vert.spv ${SHADER_OUTPUT_DIR}/sphere.frag.spv ${SHADER_OUTPUT_DIR}/wireframe.frag.spv)

add_executable(vulkan_triangle
    src/main.cpp
    src/GlobeApp.cpp
    src/UvSphereMesh.cpp
    src/tiff_loader.cpp
)

add_dependencies(vulkan_triangle shaders)

target_link_libraries(vulkan_triangle PRIVATE glfw glm::glm Vulkan::Vulkan)

target_compile_definitions(vulkan_triangle PRIVATE
    NOMINMAX
    WIN32_LEAN_AND_MEAN
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE
)

if (MSVC)
    target_compile_options(vulkan_triangle PRIVATE /W4 /permissive-)
else()
    target_compile_options(vulkan_triangle PRIVATE -Wall -Wextra -Wpedantic)
endif()
