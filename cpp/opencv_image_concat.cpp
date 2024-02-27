#include <opencv2/opencv.hpp>

int main() {
    cv::Mat left_image = cv::imread("../ipynb/color_frames/color_image484.png", cv::IMREAD_COLOR);
    cv::Mat right_image = cv::imread("../ipynb/color_frames/color_image700.png", cv::IMREAD_COLOR);

    cv::Mat h_image;

    // concat image horizontal
    cv::hconcat(left_image, right_image, h_image);

    cv::imshow("concat image", h_image);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // image copy
    cv::Mat l_img;
    left_image.copyTo(l_img);
    cv::rectangle(l_img, cv::Point(300, 220), cv::Point(340, 260), cv::Scalar(255, 0, 0));
    cv::imshow("original", left_image);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}