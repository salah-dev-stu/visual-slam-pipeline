cmake_minimum_required(VERSION 3.10)
project(slam_system)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Eigen3 REQUIRED)

# Find ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS /usr/local/include /usr/include
)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
    PATHS /usr/local/lib /usr/lib
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(WARNING "ONNX Runtime not found")
endif()

# Find g2o
find_package(g2o REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/external
    ${OpenCV_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

add_executable(slam_system
    src/main.cpp
    src/Frame.cpp
    src/FeatureExtractor.cpp
    src/DepthEstimator.cpp
    src/MapPoint.cpp
    src/Map.cpp
    src/Slam.cpp
    src/Optimizer.cpp
    src/LoopCloser.cpp
    src/Viewer.cpp
)

target_link_libraries(slam_system
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
    ${ONNXRUNTIME_LIBRARY}
    g2o::core
    g2o::types_slam3d
    g2o::solver_eigen
)
