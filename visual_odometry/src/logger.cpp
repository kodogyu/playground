#include <opencv2/opencv.hpp>

#include "myslam/logger.h"

namespace myslam {

Logger::Logger() {
    log_path_ = "/home/kodogyu/playground/visual_odometry/output_logs/";
    log_file_.open(log_path_ + "paths.csv");
}

Logger::~Logger() {
    log_file_.close();
}

void Logger::logPose(const SE3 &pose) {
    // log pose
    Eigen::Quaterniond quat = pose.unit_quaternion();
    Eigen::Vector3d trans = pose.translation();

    log_file_ << trans.x() << "," << trans.y() << "," << trans.z() << ",";  // translation
    log_file_ << quat.w() << "," << quat.x() << "," << quat.y() << "," << quat.z() << std::endl;  // rotation
}

void Logger::logImage(const std::string filename, const cv::Mat image) {
    // log image
    cv::imwrite(log_path_ + filename, image);
}

void Logger::logFeatureMatchImages(const Frame::Ptr frame) {
    cv::Mat left_image, right_image;
    cv::cvtColor(frame->left_img_, left_image, cv::COLOR_GRAY2BGR);
    cv::cvtColor(frame->right_img_, right_image, cv::COLOR_GRAY2BGR);

    cv::Mat h_image;
    cv::hconcat(left_image, right_image, h_image);

    auto left_features = frame->features_left_;
    auto right_features = frame->features_right_;

    for (int i = 0; i < left_features.size(); i++) {
        cv::Point left_kp;
        cv::Point right_kp;

        // feature exists in both images
        if (left_features[i] != nullptr &&
                right_features[i] != nullptr) {
            left_kp = left_features[i]->position_.pt;
            right_kp = right_features[i]->position_.pt;
            right_kp.x += left_image.cols;

            cv::rectangle(h_image,
                left_kp - cv::Point(5, 5),
                left_kp + cv::Point(5, 5),
                cv::Scalar(0, 255, 0));  // green
            cv::rectangle(h_image,
                right_kp - cv::Point(5, 5),
                right_kp + cv::Point(5, 5),
                cv::Scalar(0, 255, 0));  // green
            cv::line(h_image, left_kp, right_kp, cv::Scalar(0, 255, 0));
        }
        // feature only exists in left image
        else if (left_features[i] != nullptr &&
                right_features[i] == nullptr) {
            left_kp = left_features[i]->position_.pt;

            cv::rectangle(h_image,
                left_kp - cv::Point(5, 5),
                left_kp + cv::Point(5, 5),
                cv::Scalar(0, 0, 255));  // red
        }
        // feature only exists in right image
        else if (left_features[i] == nullptr &&
                right_features[i] != nullptr) {
            right_kp = right_features[i]->position_.pt;
            right_kp.x += left_image.cols;

            cv::rectangle(h_image,
            right_kp - cv::Point(5, 5),
            right_kp + cv::Point(5, 5),
            cv::Scalar(255, 0, 0));  // blue
        }
    }

    cv::imwrite(log_path_ + "concat_images/feature_match" + std::to_string(frame->id_) + ".png", h_image);
}

}