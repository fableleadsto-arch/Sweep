cmake_minimum_required(VERSION 3.15)
project(sweep_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(sweep_native sweep_native.cpp)

if(MSVC)
    target_compile_options(sweep_native PRIVATE /O2 /W4)
else()
    target_compile_options(sweep_native PRIVATE -O2 -Wall -Wextra -Wpedantic)
endif()

install(TARGETS sweep_native LIBRARY DESTINATION .)
