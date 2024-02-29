#include <boost/format.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "myslam/dataset.h"
#include "myslam/frame.h"

using namespace std;

namespace myslam {

Dataset::Dataset(const std::string& dataset_path)
    : dataset_path_(dataset_path) {
        
        ifstream frame_ids_file;
        string frame_id;

        frame_ids_file.open("/home/kodogyu/playground/ipynb/files/interesting_frames.csv");
        while (getline(frame_ids_file, frame_id)) {
            frame_ids_.push_back(frame_id);
        }
        frame_ids_file.close();

        // Open bag file
        input_bag_.open(input_filename_, rosbag::bagmode::Read);
        topics_.push_back(left_topic_);
        topics_.push_back(right_topic_);
    }

Dataset::~Dataset() {
    // Close bag files
    input_bag_.close();
      
}

bool Dataset::Init() {
    // read camera intrinsics and extrinsics
    ifstream fin(dataset_path_ + "/calib.txt");
    if (!fin) {
        LOG(ERROR) << "cannot find " << dataset_path_ << "/calib.txt!";
        return false;
    }

    for (int i = 0; i < 4; ++i) {
        char camera_name[3];
        for (int k = 0; k < 3; ++k) {
            fin >> camera_name[k];
        }
        double projection_data[12];
        for (int k = 0; k < 12; ++k) {
            fin >> projection_data[k];
        }
        Mat33 K;
        K << projection_data[0], projection_data[1], projection_data[2],
            projection_data[4], projection_data[5], projection_data[6],
            projection_data[8], projection_data[9], projection_data[10];
        Vec3 t;
        t << projection_data[3], projection_data[7], projection_data[11];
        // t = K.inverse() * t;
        // K = K * 0.5;
        Camera::Ptr new_camera(new Camera(K(0, 0), K(1, 1), K(0, 2), K(1, 2),
                                          t.norm(), SE3(SO3(), t)));
        cameras_.push_back(new_camera);
        LOG(INFO) << "Camera " << i << " extrinsics: " << t.transpose();
    }
    fin.close();
    current_image_index_ = 0;
    return true;
}

Frame::Ptr Dataset::NextFrame() {
    cv::Mat image_left, image_right;
    std::ostringstream image_suffix;
    // read images
    image_left =
        cv::imread((dataset_path_ + "/left_frames/left_image" + frame_ids_[current_image_index_] + ".png"),
                   cv::IMREAD_GRAYSCALE);
    image_right =
        cv::imread((dataset_path_ + "/right_frames/right_image" + frame_ids_[current_image_index_] + ".png"),
                   cv::IMREAD_GRAYSCALE);
    if (image_left.data == nullptr || image_right.data == nullptr) {
        LOG(WARNING) << "cannot find images at index " << current_image_index_ 
                    << " frame_id: " << frame_ids_[current_image_index_]
                    << " frame_name: " << dataset_path_ + "/****_frames/****_image" + frame_ids_[current_image_index_] + ".png";
        return nullptr;
    }

    auto new_frame = Frame::CreateFrame();
    new_frame->left_img_ = image_left;
    new_frame->right_img_ = image_right;
    // new_frame->left_img_ = image_left_resized;
    // new_frame->right_img_ = image_right_resized;
    current_image_index_++;
    return new_frame;
}

Frame::Ptr Dataset::RosBagNextFrame() {
    rosbag::View input_view(input_bag_, rosbag::TopicQuery(topics_));
    cv::Mat image_left, image_right;
    sensor_msgs::ImageConstPtr left_message_ptr, right_message_ptr;
    cv_bridge::CvImagePtr left_image_ptr, right_image_ptr;
    bool got_left_image = false;
    bool got_right_image = false;

    for (rosbag::MessageInstance const m : input_view) {
        // left frame
        if (m.getTopic() == left_topic_ && !got_left_image && m.getTime().toNSec() > last_timestamp_) {
          left_message_ptr = m.instantiate<sensor_msgs::Image>();
          left_image_ptr = cv_bridge::toCvCopy(left_message_ptr, sensor_msgs::image_encodings::MONO8);
          left_image_ptr->image.copyTo(image_left);

          got_left_image = true;
        }
        // right frame
        if (m.getTopic() == right_topic_ && !got_right_image && m.getTime().toNSec() > last_timestamp_) {
          right_message_ptr = m.instantiate<sensor_msgs::Image>();
          right_image_ptr = cv_bridge::toCvCopy(right_message_ptr, sensor_msgs::image_encodings::MONO8);
          right_image_ptr->image.copyTo(image_right);

          got_right_image = true;
        }

        // both frames are received
        if (got_left_image && got_right_image) {
            auto new_frame = Frame::CreateFrame();
            new_frame->left_img_ = image_left;
            new_frame->right_img_ = image_right;
            
            current_image_index_++;
            last_timestamp_ = m.getTime().toNSec();

            return new_frame;
        }
      }

}

}  // namespace myslam