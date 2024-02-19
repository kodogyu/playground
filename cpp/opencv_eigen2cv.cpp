#include <iostream>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

int main() {
    // Eigen 행렬 생성
    Eigen::Matrix<float, 3, 3> eigenMatrix;
    eigenMatrix << 1, 2, 3,
                   4, 5, 6,
                   7, 8, 9;

    std::cout << "Eigen Matrix:\n" << eigenMatrix << std::endl;

    // Eigen 행렬을 OpenCV 행렬로 변환
    cv::Mat cvMatrix;
    cv::eigen2cv(eigenMatrix, cvMatrix);

    std::cout << "OpenCV Matrix:\n" << cvMatrix << std::endl;

    return 0;
}
