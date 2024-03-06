#include <iostream>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/geometry/Pose3.h>

#include <Eigen/Dense>

int main() {
    // auto noise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.3, 0.3, 0.1));
    // auto noise2 = gtsam::noiseModel::Isotropic::Sigma(3, 2);
    // std::cout << noise2->covariance() << std::endl;

    // gtsam::Point3 v1, v2;
    // v1 << 1, 2, 3;
    // v2 = v1 * 2;
    // std::cout << v2 << std::endl;

    // noise model dimension
    gtsam::noiseModel::Isotropic::shared_ptr noiseModel = gtsam::noiseModel::Isotropic::Sigma(3, 1);
    std::cout << noiseModel->covariance() << std::endl;
    std::cout << "noise model dim: " << noiseModel->dim() << std::endl;

    gtsam::Pose3 init_pose;
    std::cout << init_pose.matrix() << std::endl;

    // convert eigen isometry to gtsam pose
    Eigen::Isometry3d eigen_isometry;
    Eigen::Matrix3d matrix3d;
    matrix3d << 1, 0, 0,
                0, 0, 1,
                0, 1, 0;
    eigen_isometry.matrix().block<3,3>(0,0) = matrix3d;
    eigen_isometry.matrix().block<3,1>(0,3) = Eigen::Vector3d(1, 2, 3);
    std::cout << "eigen_isometry rotation matrix:\n" << eigen_isometry.rotation() << std::endl;

    gtsam::Pose3 eigen2gtsam(gtsam::Rot3(eigen_isometry.rotation()), gtsam::Point3(eigen_isometry.translation()));

    eigen2gtsam.print();

    return 0;
}