// g++ cv_mat.cpp -I/usr/include/opencv4/ -lopencv_core -o cv_mat

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

int main() {
    cv::Mat image = cv::Mat::zeros(3, 3, CV_16UC3);
    image.at<double>(0, 0) = 10;
    // image.at<cv::Vec3s>(0, 0) = cv::Vec3s(10, 10, 10);
    cv::Vec3s elem = image.at<cv::Vec3s>(0,0);

    for (int i = 0; i < 3; i++) {
        cout << elem[i] << endl;
    }
    // cout << image << endl;

    return 0;
}
