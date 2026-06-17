cmake_minimum_required(VERSION 3.27)
project(ulvt_gpu VERSION 0.1 LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Third-party dependencies
add_subdirectory(third-party/Catch2)
add_subdirectory(third-party/nvbench)

# Settings for our own targets (defined after the third-party ones above, so
# these do not leak into Catch2 / nvbench).
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_STANDARD 20)
# Collect all of our executables in the build root (e.g. ./build/ntt_tests),
# regardless of which subdirectory defines them.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_compile_definitions(
    $<$<CONFIG:Release>:STRIP_ASSERTIONS>
    $<$<CONFIG:Release>:STRIP_CUDA_CHECK>
)

# Usage requirements shared by every module: the source root as an include path
# and the C++ standard. (The CUDA standard is set via CMAKE_CUDA_STANDARD above.)
add_library(ulvt_common INTERFACE)
target_include_directories(ulvt_common INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(ulvt_common INTERFACE cxx_std_20)

# Common nvcc options, applied to our executables.
add_library(ulvt_cuda_opts INTERFACE)
target_compile_options(ulvt_cuda_opts INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info --use_fast_math>)

add_subdirectory(src/ulvt)

# Umbrella target: link this to pull in the NTT and finite-field code (the
# finite-field math is header-only, exposed via ulvt_common's include path).
add_library(ulvt_gpu INTERFACE)
target_link_libraries(ulvt_gpu INTERFACE ulvt_ntt ulvt_utils unrolled)
