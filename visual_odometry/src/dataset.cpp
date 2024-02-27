#include "myslam/dataset.h"
#include "myslam/frame.h"

#include <boost/format.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>
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
        t = K.inverse() * t;
        K = K * 0.5;
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
    boost::format fmt("%s/%s_frames/%s%s.png");
    cv::Mat image_left, image_right;
    // read images
    image_left =
        cv::imread((fmt % dataset_path_ % "left" % "left_image" % frame_ids_[current_image_index_]).str(),
                   cv::IMREAD_GRAYSCALE);
    image_right =
        cv::imread((fmt % dataset_path_ % "right" % "right_image" % frame_ids_[current_image_index_]).str(),
                   cv::IMREAD_GRAYSCALE);

    if (image_left.data == nullptr || image_right.data == nullptr) {
        LOG(WARNING) << "cannot find images at index " << current_image_index_ 
                    << " frame_id: " << frame_ids_[current_image_index_]
                    << " frame_name: " << fmt % dataset_path_ % "left" % "left_image" % frame_ids_[current_image_index_];
        return nullptr;
    }

    cv::Mat image_left_resized, image_right_resized;
    cv::resize(image_left, image_left_resized, cv::Size(), 0.5, 0.5,
               cv::INTER_NEAREST);
    cv::resize(image_right, image_right_resized, cv::Size(), 0.5, 0.5,
               cv::INTER_NEAREST);

    auto new_frame = Frame::CreateFrame();
    new_frame->left_img_ = image_left_resized;
    new_frame->right_img_ = image_right_resized;
    current_image_index_++;
    return new_frame;
}

}  // namespace myslam