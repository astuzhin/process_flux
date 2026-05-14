cmake_minimum_required(VERSION 3.16)
project(process_flux LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ROOT setup
message(STATUS "ROOTSYS: $ENV{ROOTSYS}")
list(APPEND CMAKE_PREFIX_PATH $ENV{ROOTSYS})
find_package(ROOT REQUIRED COMPONENTS Core RIO Tree Hist)

# Source files
file(GLOB_RECURSE SRC_FILES src/*.cpp)
add_library(process_flux SHARED ${SRC_FILES})

# Include files
target_include_directories(process_flux PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(process_flux PUBLIC ROOT::Core ROOT::RIO ROOT::Tree ROOT::Hist)
