cmake_minimum_required(VERSION 4.0)
project(ppr_semestralka)

set(CMAKE_CXX_STANDARD 23)



find_library(OpenCL_LIB OpenCL)

add_executable(ppr_semestralka src/main.cpp
        libs/stringUtils.cpp
        libs/stringUtils.h
        include/dataReader.h
        include/ReadDexcomData.h
        src/DataReader.cpp
        src/CPUSequential.cpp
        src/CPUParallel.cpp
        src/GPUParallel.cpp)
target_link_libraries(ppr_semestralka PRIVATE ${OpenCL_LIB})

