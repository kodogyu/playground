// g++ eigen.cpp -I/usr/include/eigen3 -o eigen
#include <iostream>
#include <Eigen/Dense>

int main() {
    // body -> left camera
    Eigen::Matrix3d rotation_left;
    rotation_left << 0.0148655429818, -0.999880929698, 0.00414029679422,
                     0.999557249008, 0.0149672133247, 0.025715529948,
                    -0.0257744366974, 0.00375618835797, 0.999660727178;
    Eigen::Vector3d translation_left(-0.0216401454975,
                                     -0.064676986768,
                                      0.00981073058949);

    Eigen::Transform<double, 3, Eigen::Affine> body_T_left_cam;
    body_T_left_cam.linear() = rotation_left;
    body_T_left_cam.translation() = translation_left;

    // body -> right camera
    Eigen::Matrix3d rotation_right;
    rotation_left << 0.0125552670891, -0.999755099723, 0.0182237714554,
                     0.999598781151, 0.0130119051815, 0.0251588363115,
                    -0.0253898008918, 0.0179005838253, 0.999517347078;
    Eigen::Vector3d translation_right(-0.0198435579556,
                                       0.0453689425024,
                                       0.00786212447038);

    Eigen::Transform<double, 3, Eigen::Affine> body_T_right_cam;
    body_T_right_cam.linear() = rotation_right;
    body_T_right_cam.translation() = translation_right;

    std::cout << body_T_left_cam.matrix().inverse() * body_T_right_cam.matrix() << std::endl;

    return 0;
}