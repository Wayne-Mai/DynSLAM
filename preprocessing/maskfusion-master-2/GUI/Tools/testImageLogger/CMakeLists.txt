cmake_minimum_required(VERSION 3.0.2)

set(CMAKE_CXX_STANDARD 14)


# policy to export a different link interface
cmake_policy(SET CMP0022 OLD)

project(DebugImageLog)

set(PARENT_PROJECT_PATH /home/wayne/SLAM/DynSLAM/preprocessing/maskfusion-master-2)

# find_package(OpenCV 3.1 REQUIRED)
find_package(OpenCV 3.1 REQUIRED PATHS "${PARENT_PROJECT_PATH}/deps/opencv/build")
find_package(OpenMP)


if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# file(GLOB utils_srcs1 ${PARENT_PROJECT_PATH}/Core/Utils/Resolution.h)
#file(GLOB utils_srcs2 ${PARENT_PROJECT_PATH}/Core/Utils/Stopwatch.h)


message("Using: ")
message("=======")
message("Using python:  ${PYTHONLIBS_VERSION_STRING} ")
message("Using PYTHON_LIBRARIES:  ${PYTHON_LIBRARIES} ")
message("Using PYTHON_INCLUDE_DIRS:  ${PYTHON_INCLUDE_DIRS} ")
# message("Using Utils file:  ${utils_srcs1} ${utils_srcs2} ")


include_directories(${OpenCV_INCLUDE_DIRS})


# Eigen3
set(Eigen3_INCLUDE_DIRS ${PARENT_PROJECT_PATH}/deps/)
include_directories(${Eigen3_INCLUDE_DIRS})

# pangolin
find_package(Pangolin 0.4 REQUIRED)
message("Pangolin_INCLUDE_DIRS: ${Pangolin_INCLUDE_DIRS}")
include_directories(${Pangolin_INCLUDE_DIRS})
message("Using PANGOLINS:  ${Pangolin_INCLUDE_DIRS} ")

# Boost

# find_package(Boost 1.58.0 REQUIRED PATHS "${PARENT_PROJECT_PATH}/deps/boost")
# set(BOOST_INCLUDE_DIRS ${PARENT_PROJECT_PATH}/deps/boost/include)
 set(BOOST_ROOT ${PARENT_PROJECT_PATH}/deps/boost)
# set(BOOST_DIR ${PARENT_PROJECT_PATH}/deps/boost)
# include_directories(${BOOST_INCLUDE_DIRS})
# link_directories(${Boost_LIBRARY_DIRS})
message("Using BOOSTS:  ${BOOST_ROOT} ")



# include_directories(${utils_srcs})

set(CMAKE_BUILD_TYPE "Debug")


add_executable(Main main.cpp)







#add_library(${PROJECT_NAME} SHARED
#            ${utils_srcs1}
#            ${utils_srcs2}
#)






