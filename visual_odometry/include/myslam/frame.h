#pragma once

#ifndef MYSLAM_FRAME_H
#define MYSLAM_FRAME_H

#include "myslam/camera.h"
#include "myslam/common_include.h"

namespace myslam {

// forward declare
struct MapPoint;
struct Feature;

/**
 * 帧
 * 각 프레임에는 독립적인 ID가 할당되고, 키 프레임에는 키 프레임 ID가 할당됩니다.
 */
struct Frame {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frame> Ptr;

    unsigned long id_ = 0;           // id of this frame
    unsigned long keyframe_id_ = 0;  // id of key frame
    bool is_keyframe_ = false;       // 키프레임 여부
    double time_stamp_;              // 타임스탬프, 아직 사용되지 않음
    SE3 pose_;                       // Tcw 형식 포즈
    std::mutex pose_mutex_;          // Pose data mutex
    cv::Mat left_img_, right_img_;   // stereo images

    // extracted features in left image
    std::vector<std::shared_ptr<Feature>> features_left_;
    // corresponding features in right image, set to nullptr if no corresponding
    std::vector<std::shared_ptr<Feature>> features_right_;

   public:  // data members
    Frame() {}

    Frame(long id, double time_stamp, const SE3 &pose, const Mat &left,
          const Mat &right);

    // set and get pose, thread safe
    SE3 Pose() {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        return pose_;
    }

    void SetPose(const SE3 &pose) {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        pose_ = pose;
    }

    /// 키프레임 및 keyframe_id 설정
    void SetKeyFrame();

    /// 새 프레임 생성 및 ID 할당
    static std::shared_ptr<Frame> CreateFrame();
};

}  // namespace myslam

#endif  // MYSLAM_FRAME_H
