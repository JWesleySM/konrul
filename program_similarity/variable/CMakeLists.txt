# mkdir build
# cd build
# cmake -DLLVM_INSTALL_DIR=$LLVM_INSTALL_DIR ..
# make -j$(nproc)

#===============================================================================
# See: https://github.com/banach-space/llvm-tutor/blob/master/CMakeLists.txt
#===============================================================================
cmake_minimum_required(VERSION 3.13.4)
project(VariableSim)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#===============================================================================
# 1. VERIFY LLVM INSTALLATION DIR
#===============================================================================
# Set this to a valid LLVM installation dir:
set(LLVM_INSTALL_DIR "" CACHE PATH "LLVM installation directory")

# A bit of a sanity checking:
set(LLVM_INSTALL_INCLUDE "${LLVM_INSTALL_DIR}/include/llvm")
if(NOT EXISTS "${LLVM_INSTALL_INCLUDE}")
message(FATAL_ERROR
    " LLVM_INSTALL_INCLUDE (${LLVM_INSTALL_INCLUDE}) is invalid.")
endif()

set(LLVM_INSTALL_CMAKE "${LLVM_INSTALL_DIR}/lib/cmake/llvm/LLVMConfig.cmake")
if(NOT EXISTS "${LLVM_INSTALL_CMAKE}")
message(FATAL_ERROR
    " LLVM_INSTALL_CMAKE (${LLVM_INSTALL_CMAKE}) is invalid.")
endif()


#===============================================================================
# 2. LOAD LLVM CONFIGURATION
#    For more: http://llvm.org/docs/CMake.html#embedding-llvm-in-your-project
#===============================================================================
# Add the location of LLVMConfig.cmake to CMake search paths (so that
# find_package can locate it):
list(APPEND CMAKE_PREFIX_PATH "${LLVM_INSTALL_DIR}/lib/cmake/llvm/")

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_INSTALL_DIR}")

message("LLVM STATUS:
  Definitions ${LLVM_DEFINITIONS}
  Includes    ${LLVM_INCLUDE_DIRS}
  Libraries   ${LLVM_LIBRARY_DIRS}
  Targets     ${LLVM_TARGETS_TO_BUILD}"
)

# Set the LLVM header and library paths
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

#===============================================================================
# 3. BUILD CONFIGURATION
#===============================================================================
set(CMAKE_CXX_STANDARD 17 CACHE STRING "")

# Build type
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE
      STRING "Build type (default Debug):" FORCE)
endif()

# Compiler flags
#iset(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall\
#    -fdiagnostics-color=always")

# LLVM is normally built without RTTI. Be consistent with that.
if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()

add_library(VariableSim SHARED VariableSim.cpp)

target_link_libraries(VariableSim
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")
