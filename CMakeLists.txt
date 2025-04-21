cmake_minimum_required(VERSION 4.0)

project(RISCV-Simulator)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()


include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(
        Simulator
        src/MainCPU.cpp
        src/MemoryManager.cpp
        src/Simulator.cpp
        src/BranchPredictor.cpp
        src/Cache.cpp
)

add_executable(
        CacheSim
        src/MainCache.cpp
        src/MemoryManager.cpp
        src/Cache.cpp
)

add_executable(
        CacheOptimized
        src/MainCacheOptimization.cpp
        src/MemoryManager.cpp
        src/Cache.cpp
)

add_executable(ToDirenoTrace src/ToDirenoTrace.cpp)