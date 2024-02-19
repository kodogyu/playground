project(playground_cpp)

find_package(Eigen3)
find_package(GTSAM)
find_package(OpenCV)

include_directories(
    ${EIGEN3_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# GTSAM files
add_executable(gtsam_pose gtsam_pose.cpp)
target_link_libraries(gtsam_pose gtsam)

add_executable(gtsam_between gtsam_between.cpp)
target_link_libraries(gtsam_between gtsam)


add_executable(gtsam_NonLinearFactorGraph gtsam_NonLinearFactorGraph.cpp)
target_link_libraries(gtsam_NonLinearFactorGraph gtsam)

# OpenCV files
add_executable(opencv_stereo_rectify opencv_stereo_rectify.cpp)
target_link_libraries(opencv_stereo_rectify
${OpenCV_LIBS}
)