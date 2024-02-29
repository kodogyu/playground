#include <iostream>
#include <gtsam/linear/NoiseModel.h>

int main() {
    auto noise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.3, 0.3, 0.1));
    auto noise2 = gtsam::noiseModel::Isotropic::Sigma(3, 2);

    std::cout << noise2->covariance() << std::endl;

    return 0;
}