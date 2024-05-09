project(playground_cpp)

cmake_minimum_required(VERSION 3.16)
# set(CMAKE_CXX_COMPILER "/usr/bin/g++-11")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_BUILD_TYPE Debug)
# list(APPEND CMAKE_MODULE_PATH "/usr/local/include/eigen-3.4.0/cmake")

##=========================DEPENDENCIES=========================##
find_package(Eigen3 3.4 REQUIRED)
find_package(GTSAM REQUIRED)
find_package(OpenCV 4 REQUIRED)   # -DOpenCV_DIR=/home/kodogyu/opencv/build
find_package(Sophus REQUIRED)
# find_package(cv_bridge REQUIRED)
find_package(Pangolin 0.8 REQUIRED)

message(STATUS "opencv version: ${OpenCV_VERSION}")
message(STATUS "gtsam path: ${GTSAM_DIR}")
message(STATUS "eigen path: ${Eigen3_DIR}")

include_directories(
    ${EIGEN3_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    # ${cv_bridge_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
)

##=========================EXECUTABLES=========================##
# STL files
add_executable(basic_copy_array basic_copy_array.cpp)
add_executable(basic_static basic_static.cpp)
add_executable(basic_file basic_file.cpp)
add_executable(basic_nothing basic_nothing.cpp)
add_executable(basic_unordered_map basic_unordered_map.cpp)
add_executable(chrono_measure_time chrono_measure_time.cpp)
add_executable(basic_filesystem basic_filesystem.cpp)
add_executable(basic_vector basic_vector.cpp)
add_executable(basic_map basic_map.cpp)
add_executable(basic_random basic_random.cpp)
add_executable(basic_enum_class_flag basic_enum_class_flag.cpp)
# boost
add_executable(boost_format boost_format.cpp)
# eigen
add_executable(eigen_transform eigen_transform.cpp)
add_executable(eigen_basic eigen_basic.cpp)
add_executable(eigen_rpe eigen_rpe.cpp)
# multi thread
add_executable(basic_multi_thread basic_multi_thread.cpp)
target_link_libraries(basic_multi_thread pthread)  # PRIVATE
add_executable(basic_multi_thread_mutex basic_multi_thread_mutex.cpp)
target_link_libraries(basic_multi_thread_mutex pthread)  # PRIVATE
add_executable(basic_multi_thread_2threads basic_multi_thread_2threads.cpp)
target_link_libraries(basic_multi_thread_2threads pthread)  # PRIVATE


# GTSAM files
add_executable(gtsam_basic gtsam_basic.cpp)
target_link_libraries(gtsam_basic gtsam)

add_executable(gtsam_pose gtsam_pose.cpp)
target_link_libraries(gtsam_pose gtsam)

add_executable(gtsam_between gtsam_between.cpp)
target_link_libraries(gtsam_between gtsam)

add_executable(gtsam_NonLinearFactorGraph gtsam_NonLinearFactorGraph.cpp)
target_link_libraries(gtsam_NonLinearFactorGraph gtsam)

add_executable(gtsam_Pose2SLAMExample_graphviz gtsam_Pose2SLAMExample_graphviz.cpp)
target_link_libraries(gtsam_Pose2SLAMExample_graphviz gtsam)

add_executable(gtsam_OdometryExample gtsam_OdometryExample.cpp)
target_link_libraries(gtsam_OdometryExample gtsam)

add_executable(gtsam_StereoVOExample gtsam_StereoVOExample.cpp)
target_link_libraries(gtsam_StereoVOExample gtsam)

# OpenCV files
add_executable(opencv_cv_mat opencv_cv_mat.cpp)
target_link_libraries(opencv_cv_mat ${OpenCV_LIBS})

add_executable(opencv_eigen2cv opencv_eigen2cv.cpp)
target_link_libraries(opencv_eigen2cv ${OpenCV_LIBS})

add_executable(opencv_matchTemplate opencv_matchTemplate.cpp)
target_link_libraries(opencv_matchTemplate ${OpenCV_LIBS})

add_executable(opencv_stereo_rectify opencv_stereo_rectify.cpp)
target_link_libraries(opencv_stereo_rectify ${OpenCV_LIBS})

add_executable(opencv_image_concat opencv_image_concat.cpp)
target_link_libraries(opencv_image_concat ${OpenCV_LIBS})

add_executable(opencv_cv_fileStorage opencv_cv_fileStorage.cpp)
target_link_libraries(opencv_cv_fileStorage ${OpenCV_LIBS})

add_executable(opencv_fisheye_camera opencv_fisheye_camera.cpp)
target_link_libraries(opencv_fisheye_camera ${OpenCV_LIBS})

add_executable(new new.cpp)
target_link_libraries(new ${OpenCV_LIBS})

add_executable(opencv_vis3d opencv_vis3d.cpp)
target_link_libraries(opencv_vis3d ${OpenCV_LIBS})

# Sophus files
add_executable(sophus_basic sophus_basic.cpp)
target_link_libraries(sophus_basic Sophus::Sophus)

# Pangolin files
add_executable(pangolin_HelloPangolin pangolin_HelloPangolin.cpp)
target_link_libraries(pangolin_HelloPangolin pango_display pango_python)

add_executable(pangolin_ui pangolin_ui.cpp)
target_link_libraries(pangolin_ui pango_display pango_python)

# Applications
add_executable(get_right_image get_right_image.cpp)
target_link_libraries(get_right_image ${OpenCV_LIBS})

add_executable(visual_odometry_example visual_odometry_example.cpp)
target_link_libraries(visual_odometry_example ${OpenCV_LIBS} ${Pangolin_LIBRARIES} gtsam opengv)

add_executable(vo_patch vo_patch.cpp)
target_link_libraries(vo_patch ${OpenCV_LIBS} ${Pangolin_LIBRARIES})

add_executable(vo_lower_bound vo_lower_bound.cpp)
target_link_libraries(vo_lower_bound ${OpenCV_LIBS} ${Pangolin_LIBRARIES} gtsam)

add_executable(pose_converter pose_converter.cpp)

add_executable(find_fundamental_matrix find_fundamental_matrix.cpp)
target_link_libraries(find_fundamental_matrix ${OpenCV_LIBS} ${Pangolin_LIBRARIES})

