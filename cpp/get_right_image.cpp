// g++ -o get_right_image get_right_image.cpp `pkg-config opencv --cflags --libs`
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;

double focal_length_ = 604;
double baseline_ = 50;

void getRightImage(cv::Mat const &left_image, cv::Mat const &depth_image, cv::Mat *right_image) {
    // ROS_INFO_STREAM("calculating right image ...");
    // ROS_INFO_STREAM("left image type : " << left_image.type());

    short depth = 0;
    int disparity = 0;

    for (int row = 0; row < left_image.rows; row++) {
        for (int col = 0; col < left_image.cols; col++) {
            depth = depth_image.at<short>(row, col);
            if (depth > 0) {
                disparity = (int)(focal_length_ * baseline_ / depth);

                // ROS_DEBUG_STREAM("(row, col): (" << row << ", " << col << ")");
                // ROS_DEBUG_STREAM("disparity: " << disparity);
                if (col - disparity > 0) {
                    right_image->at<cv::Vec3b>(row, col - disparity) = left_image.at<cv::Vec3b>(row, col);
                }
            }
        }
    }
}

int main() {
    cv::Mat color_image = cv::imread("../ipynb/color_frames/color_image700.png", cv::IMREAD_COLOR);
    cv::Mat depth_image = cv::imread("../ipynb/depth_frames/depth_image429.png", cv::IMREAD_ANYDEPTH);
    cv::Mat right_image = cv::Mat::zeros(color_image.rows, color_image.cols, color_image.type());;
    cout << "color image type: " << color_image.type() << endl;
    cout << "depth image type: " << depth_image.type() << endl;
    cout << "right image type: " << right_image.type() << endl;

    double min, max;
    cv::minMaxLoc(depth_image, &min, &max);
    cout << "max depth: " << max << endl;

    cout << "right value: " << right_image.at<cv::Vec3d>(400,600) << endl;
    getRightImage(color_image, depth_image, &right_image);
    // right_image.at<cv::Vec3d>(400,600) = cv::Vec3d(1, 2, 3);
    cout << "----func----" << endl;
    cout << "depth value: " << depth_image.at<double>(400,600) << endl;
    cout << "right value: " << right_image.at<cv::Vec3w>(400,600) << endl;

    cv::imwrite("images/right_image50w.png", right_image);
    // cv::imwrite("images/color_image.png", color_image);
    // cv::imwrite("images/depth_image.png", depth_image);
    return 0;
}