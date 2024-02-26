#include <iostream>
#include <Eigen/Core>
#include <sophus/se3.hpp>

int main() {
    // Rotation
    Sophus::Matrix3d R = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();
    Sophus::SO3d SO3_R(R);
    // Translation
    Eigen::Vector3d translation(1, 2, 3);
    
    // Transform
    Sophus::SE3d transform(SO3_R, translation);

    // Print
    std::cout << "rotation: \n" << transform.rotationMatrix() << std::endl;
    std::cout << "translation: \n" << transform.translation() << std::endl;
    return 0;
}