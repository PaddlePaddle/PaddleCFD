# Copyright 2022-2025 MetaOPT Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

cmake_minimum_required(VERSION 3.11)  # for FetchContent
project(fused_segment_csr LANGUAGES CXX)

include(FetchContent)

set(THIRD_PARTY_DIR "${CMAKE_CURRENT_BINARY_DIR}/third_party")
if(NOT DEFINED PYBIND11_VERSION AND NOT "$ENV{PYBIND11_VERSION}" STREQUAL "")
    set(PYBIND11_VERSION "$ENV{PYBIND11_VERSION}")
endif()
if(NOT PYBIND11_VERSION)
    set(PYBIND11_VERSION stable)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
add_compile_definitions(PADDLE_WITH_CUDA=1)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # -fPIC
set(CMAKE_CXX_VISIBILITY_PRESET hidden)  # -fvisibility=hidden

if(MSVC)
    string(APPEND CMAKE_CXX_FLAGS " /Wall")
    string(APPEND CMAKE_CXX_FLAGS_DEBUG " /Zi")
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " /O2 /Ob2")
else()
    string(APPEND CMAKE_CXX_FLAGS " -Wall")
    string(APPEND CMAKE_CXX_FLAGS_DEBUG " -g -Og")
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O3")
endif()

if(NOT DEFINED USE_FP16 AND NOT "$ENV{USE_FP16}" STREQUAL "")
    set(USE_FP16 "$ENV{USE_FP16}")
endif()

if(NOT DEFINED USE_FP16)
    set(USE_FP16 OFF)
    message(WARNING "FP16 support disabled, compiling without paddle.HalfTensor. Suppress this warning with -DUSE_FP16=ON or -DUSE_FP16=OFF.")
elseif(USE_FP16)
    message(STATUS "FP16 support enabled, compiling with paddle.HalfTensor.")
else()
    message(STATUS "FP16 support disabled, compiling without paddle.HalfTensor.")
endif()

if(USE_FP16)
    add_definitions(-DUSE_FP16)
endif()


function(system)
    set(options STRIP)
    set(oneValueArgs OUTPUT_VARIABLE ERROR_VARIABLE WORKING_DIRECTORY)
    set(multiValueArgs COMMAND)
    cmake_parse_arguments(
            SYSTEM
            "${options}"
            "${oneValueArgs}"
            "${multiValueArgs}"
            "${ARGN}"
    )

    if(NOT DEFINED SYSTEM_WORKING_DIRECTORY)
        set(SYSTEM_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    endif()

    execute_process(
            COMMAND ${SYSTEM_COMMAND}
            OUTPUT_VARIABLE STDOUT
            ERROR_VARIABLE STDERR
            WORKING_DIRECTORY "${SYSTEM_WORKING_DIRECTORY}"
    )

    if("${SYSTEM_STRIP}")
        string(STRIP "${STDOUT}" STDOUT)
        string(STRIP "${STDERR}" STDERR)
    endif()

    set("${SYSTEM_OUTPUT_VARIABLE}" "${STDOUT}" PARENT_SCOPE)

    if(DEFINED SYSTEM_ERROR_VARIABLE)
        set("${SYSTEM_ERROR_VARIABLE}" "${STDERR}" PARENT_SCOPE)
    endif()
endfunction()

if(NOT DEFINED PYTHON_EXECUTABLE)
    if(WIN32)
        set(PYTHON_EXECUTABLE "python.exe")
    else()
        set(PYTHON_EXECUTABLE "python")
    endif()
endif()

if(UNIX)
    system(
            STRIP OUTPUT_VARIABLE PYTHON_EXECUTABLE
            COMMAND bash -c "type -P '${PYTHON_EXECUTABLE}'"
    )
endif()

system(
        STRIP OUTPUT_VARIABLE PYTHON_VERSION
        COMMAND "${PYTHON_EXECUTABLE}" -c "print('.'.join(map(str, __import__('sys').version_info[:3])))"
)

message(STATUS "Use Python version: ${PYTHON_VERSION}")
message(STATUS "Use Python executable: \"${PYTHON_EXECUTABLE}\"")

if(NOT DEFINED PYTHON_INCLUDE_DIR)
    message(STATUS "Auto detecting Python include directory...")
    system(
            STRIP OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
            COMMAND "${PYTHON_EXECUTABLE}" -c "print(__import__('sysconfig').get_path('platinclude'))"
    )
endif()

if("${PYTHON_INCLUDE_DIR}" STREQUAL "")
    message(FATAL_ERROR "Python include directory not found")
else()
    message(STATUS "Detected Python include directory: \"${PYTHON_INCLUDE_DIR}\"")
    include_directories("${PYTHON_INCLUDE_DIR}")
endif()

system(
        STRIP OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        COMMAND "${PYTHON_EXECUTABLE}" -c "print(__import__('sysconfig').get_path('purelib'))"
)
message(STATUS "Detected Python site packages: \"${PYTHON_SITE_PACKAGES}\"")

find_package(CUDAToolkit REQUIRED)

if(CUDAToolkit_FOUND AND NOT WIN32)
    message(STATUS "Found CUDA Toolkit, potentially enabling CUDA support.")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD "${CMAKE_CXX_STANDARD}")
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_definitions(-D__USE_CUDA__)

    string(APPEND CMAKE_CUDA_FLAGS " $ENV{PADDLE_NVCC_FLAGS}")

    # Execute Python code to get and process Paddle's supported CUDA architectures
    system(
            STRIP OUTPUT_VARIABLE CMAKE_CUDA_ARCHITECTURES
            COMMAND "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_nvcc_flags.py"
    )

    if(CMAKE_CUDA_ARCHITECTURES)
        message(STATUS "Found Paddle CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        message(WARNING "CUDA is not available in the detected Paddle, or Paddle is not installed. Building for all available CUDA architectures.")
        # CMake will default to building for all if CMAKE_CUDA_ARCHITECTURES is not set
    endif()

    set(CUDA_ARCH_FLAGS "") # No need for cuda_select_nvcc_arch_flags
    message(STATUS "CMAKE_CUDA_ARCHITECTURES: \"${CMAKE_CUDA_ARCHITECTURES}\"")

    list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr" "--expt-extended-lambda")
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "7.5")
        if (USE_FP16)
            message(STATUS "Found CUDA Toolkit with FP16 support, compiling with paddle.cuda.HalfTensor.")
            string(APPEND CMAKE_CUDA_FLAGS " -DCUDA_HAS_FP16=1"
                    " -D__CUDA_NO_HALF_OPERATORS__"
                    " -D__CUDA_NO_HALF_CONVERSIONS__"
                    " -D__CUDA_NO_HALF2_OPERATORS__"
                    " -D__CUDA_NO_BFLOAT16_CONVERSIONS__")
        else()
            message(STATUS "Found CUDA Toolkit with FP16 support, but it is suppressed by the compile options, compiling without paddle.cuda.HalfTensor.")
        endif()
    else()
        message(STATUS "Could not find CUDA Toolkit with FP16 support (version < 7.5), compiling without paddle.cuda.HalfTensor.")
    endif()

    foreach(FLAG ${CUDA_NVCC_FLAGS})
        string(FIND "${FLAG}" " " flag_space_position)
        if(NOT flag_space_position EQUAL -1)
            message(FATAL_ERROR "Found spaces in CUDA_NVCC_FLAGS entry '${FLAG}'")
        endif()
        string(APPEND CMAKE_CUDA_FLAGS " ${FLAG}")
    endforeach()
    string(STRIP "${CMAKE_CUDA_FLAGS}" CMAKE_CUDA_FLAGS)
    message(STATUS "CMAKE_CUDA_FLAGS: \"${CMAKE_CUDA_FLAGS}\"")

    if(MSVC)
        set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} /O2 /Ob2")
    else()
        set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
    endif()

    # Ensure the CUDA targets use the defined architectures
    set(CUDA_SELECT_NVCC_ARCH_FLAGS_CALLED TRUE) # To avoid potential issues if other parts expect this
else()
    if(NOT CUDAToolkit_FOUND)
        message(STATUS "CUDA Toolkit not found, build for CPU-only.")
    else()
        message(STATUS "CUDA Toolkit found, but build for CPU-only on Windows.")
    endif()
endif()

set(PYBIND11_PYTHON_VERSION "${PYTHON_VERSION}")

if(NOT DEFINED PYBIND11_CMAKE_DIR)
    message(STATUS "Auto detecting pybind11 CMake directory...")
    system(
            STRIP OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
            COMMAND "${PYTHON_EXECUTABLE}" -m pybind11 --cmakedir
    )
endif()

if("${PYBIND11_CMAKE_DIR}" STREQUAL "")
    FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG "${PYBIND11_VERSION}"
            GIT_SHALLOW TRUE
            SOURCE_DIR "${THIRD_PARTY_DIR}/pybind11"
            BINARY_DIR "${THIRD_PARTY_DIR}/.cmake/pybind11/build"
            STAMP_DIR "${THIRD_PARTY_DIR}/.cmake/pybind11/stamp"
    )
    FetchContent_GetProperties(pybind11)

    if(NOT pybind11_POPULATED)
        message(STATUS "Populating Git repository pybind11@${PYBIND11_VERSION} to third_party/pybind11...")
        FetchContent_MakeAvailable(pybind11)
    endif()
else()
    message(STATUS "Detected Pybind11 CMake directory: \"${PYBIND11_CMAKE_DIR}\"")
    find_package(pybind11 CONFIG PATHS "${PYBIND11_CMAKE_DIR}")
endif()

if(NOT DEFINED PADDLE_INCLUDE_PATH)
    message(STATUS "Auto detecting Paddle include directory...")
    system(
            STRIP OUTPUT_VARIABLE PADDLE_INCLUDE_PATH
            COMMAND "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_paddle_include_paths.py"
    )

    if("${PADDLE_INCLUDE_PATH}" STREQUAL "")
        set(PADDLE_INCLUDE_PATH "${PYTHON_SITE_PACKAGES}/paddle/include")
    endif()
endif()

if("${PADDLE_INCLUDE_PATH}" STREQUAL "")
    message(FATAL_ERROR "Paddle include directory not found. Got: \"${PADDLE_INCLUDE_PATH}\"")
else()
    message(STATUS "Detected Paddle include directory: \"${PADDLE_INCLUDE_PATH}\"")
    include_directories(${PADDLE_INCLUDE_PATH})
endif()

if(NOT DEFINED PADDLE_LIBRARY_PATH)
    message(STATUS "Auto detecting Paddle library directory...")
    system(
            STRIP OUTPUT_VARIABLE PADDLE_LIBRARY_PATH
            COMMAND "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_paddle_library_paths.py"
    )

    if("${PADDLE_LIBRARY_PATH}" STREQUAL "")
        set(PADDLE_LIBRARY_PATH "${PYTHON_SITE_PACKAGES}/paddle/lib")
    endif()
endif()

if("${PADDLE_LIBRARY_PATH}" STREQUAL "")
    message(FATAL_ERROR "Paddle library directory not found. Got: \"${PADDLE_LIBRARY_PATH}\"")
else()
    message(STATUS "Detected Paddle library directory: \"${PADDLE_LIBRARY_PATH}\"")
endif()

set(PADDLE_LIBRARY "")

foreach(VAR_PATH ${PADDLE_LIBRARY_PATH})
    file(GLOB ALL_FILES_AND_DIRS "${VAR_PATH}/*")

    list(FILTER ALL_FILES_AND_DIRS EXCLUDE REGEX "\\.py$")
    list(FILTER ALL_FILES_AND_DIRS EXCLUDE REGEX "\\.h$")
    list(FILTER ALL_FILES_AND_DIRS EXCLUDE REGEX "\\.cuh$")
    list(FILTER ALL_FILES_AND_DIRS EXCLUDE REGEX "\\.pyi$")
    list(FILTER ALL_FILES_AND_DIRS EXCLUDE REGEX "libflashattn\\.so$")

    set(FILTERED_FILES "")
    foreach(ITEM ${ALL_FILES_AND_DIRS})
        if(NOT IS_DIRECTORY "${ITEM}")
            list(APPEND FILTERED_FILES "${ITEM}")
        endif()
    endforeach()

    list(APPEND PADDLE_LIBRARY ${FILTERED_FILES})
endforeach()

message(STATUS "All Paddle libraries: \"${PADDLE_LIBRARY}\"")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/csrc/include")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/csrc/src")
