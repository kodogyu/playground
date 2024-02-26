project(playground_cpp)

cmake_minimum_required(VERSION 3.16)

find_package(Eigen3 REQUIRED)
find_package(GTSAM REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)

include_directories(
    ${EIGEN3_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

##=========================EXECUTABLES=========================##
# STL files
add_executable(basic_copy_array basic_copy_array.cpp)
add_executable(basic_file basic_file.cpp)
add_executable(basic_nothing basic_nothing.cpp)
add_executable(basic_unordered_map basic_unordered_map.cpp)
add_executable(chrono_measure_time chrono_measure_time.cpp)
# eigen
add_executable(eigen_transform eigen_transform.cpp)
add_executable(eigen_basic eigen_basic.cpp)

# GTSAM files
add_executable(gtsam_pose gtsam_pose.cpp)
target_link_libraries(gtsam_pose gtsam)

add_executable(gtsam_between gtsam_between.cpp)
target_link_libraries(gtsam_between gtsam)

add_executable(gtsam_NonLinearFactorGraph gtsam_NonLinearFactorGraph.cpp)
target_link_libraries(gtsam_NonLinearFactorGraph gtsam)

# OpenCV files
add_executable(opencv_cv_mat opencv_cv_mat.cpp)
target_link_libraries(opencv_cv_mat ${OpenCV_LIBS})

add_executable(opencv_eigen2cv opencv_eigen2cv.cpp)
target_link_libraries(opencv_eigen2cv ${OpenCV_LIBS})

add_executable(opencv_matchTemplate opencv_matchTemplate.cpp)
target_link_libraries(opencv_matchTemplate ${OpenCV_LIBS})

add_executable(opencv_stereo_rectify opencv_stereo_rectify.cpp)
target_link_libraries(opencv_stereo_rectify ${OpenCV_LIBS})

# Sophus files
add_executable(sophus_basic sophus_basic.cpp)
target_link_libraries(sophus_basic Sophus::Sophus)

# Applications
add_executable(get_right_image get_right_image.cpp)
target_link_libraries(get_right_image ${OpenCV_LIBS})