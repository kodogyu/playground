// g++ cv_mat.cpp -I/usr/include/opencv4/ -lopencv_core -o cv_mat

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

int main() {
    // block read/write
    cv::Mat image = cv::Mat::zeros(3, 3, CV_16UC3);
    image.at<double>(0, 0) = 10;
    // image.at<cv::Vec3s>(0, 0) = cv::Vec3s(10, 10, 10);
    cv::Vec3s elem = image.at<cv::Vec3s>(0,0);

    for (int i = 0; i < 3; i++) {
        cout << elem[i] << endl;
    }

    // cv::Mat basics
    cv::Mat color_image = cv::imread("../ipynb/color_frames/color_image484.png", cv::IMREAD_COLOR);
    cout << color_image.cols << endl;
    cout << color_image.dims << endl;

    // cv::Mat empty?
    cv::Mat empty_mat;
    cout << "is empty?: " << empty_mat.empty() << endl;

    // cout
    cv::Mat mat1 = (cv::Mat_<int>(2, 2) << 1, 2, 3, 4);
    cout <<"mat1:" << endl;
    cout << mat1 << endl;

    // cv::Mat initialize with a vector
    cv::Mat small_mat = (cv::Mat_<int>(1, 3) << 1, 2, 3);
    std::vector<cv::Mat> vec{small_mat, small_mat};
    cout << "small mat type: " << small_mat.type() << endl;
    cv::Mat mat2;
    for (auto elem : vec) {
        mat2.push_back(elem);
    }
    cout <<"mat2:" << endl;
    cout << mat2 << endl;

    return 0;
}
