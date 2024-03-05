// g++ eigen.cpp -I/usr/include/eigen3 -o eigen
#include <iostream>
#include <Eigen/Dense>

int main() {
    // Vector
    Eigen::Vector2d vec2;
    vec2 << 5, 10;

    std::cout << vec2[0] << std::endl;

    // Matrix
    Eigen::Isometry3d pose;
    Eigen::Matrix3d rotationMatrix_eigen;
    Eigen::Vector3d translation_eigen;
    rotationMatrix_eigen << 1, 2, 3,
                            4, 5, 6,
                            7, 8, 9;
    translation_eigen << 11, 12, 13;
    pose.matrix().block<3, 3>(0, 0) = rotationMatrix_eigen;
    pose.matrix().block<3, 1>(0, 3) = translation_eigen;

    std::cout << "pose: \n" << pose.matrix() << std::endl;
    return 0;
}