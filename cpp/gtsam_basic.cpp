#include <iostream>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/geometry/Pose3.h>

int main() {
    // auto noise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.3, 0.3, 0.1));
    // auto noise2 = gtsam::noiseModel::Isotropic::Sigma(3, 2);

    // std::cout << noise2->covariance() << std::endl;

    gtsam::Point3 v1, v2;
    v1 << 1, 2, 3;
    v2 = v1 * 2;

    std::cout << v2 << std::endl;

    return 0;
}