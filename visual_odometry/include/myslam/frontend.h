#pragma once
#ifndef MYSLAM_FRONTEND_H
#define MYSLAM_FRONTEND_H

#include <opencv2/features2d.hpp>
#include <math.h>

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/map.h"
#include "myslam/logger.h"

namespace myslam {

class Backend;
class Viewer;

enum class FrontendStatus { INITING, TRACKING_GOOD, TRACKING_BAD, LOST };

struct SubpixelParams {
    cv::Size window_size_ = cv::Size(10, 10);
    cv::Size zero_zone_ = cv::Size(-1, -1);
    cv::TermCriteria term_criteria_ = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                       10,
                       0.01);
};

struct FeatureDetectorParams {
    public:
        int max_features_per_frame_ = 400;
        int min_distance_btw_tracked_and_detected_features_ = 20;
        int nr_horizontal_bins_ = 7;
        int nr_vertical_bins_ = 5;
        Eigen::MatrixXd binning_mask_;
        bool enable_subpixel_corner_refinement_ = true;
        SubpixelParams subpixel_corner_finder_params_;
};

/**
 * 前端
 * 估计当前帧Pose，在满足关键帧条件时向地图加入关键帧并触发优化
 */
class Frontend {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frontend> Ptr;

    Frontend();

    /// 外部接口，添加一个帧并计算其定位结果
    bool AddFrame(Frame::Ptr frame);

    /// Set函数
    void SetMap(Map::Ptr map) { map_ = map; }

    void SetBackend(std::shared_ptr<Backend> backend) { backend_ = backend; }

    void SetViewer(std::shared_ptr<Viewer> viewer) { viewer_ = viewer; }

    void SetLogger(std::shared_ptr<Logger> logger) { logger_ = logger; }

    FrontendStatus GetStatus() const { return status_; }

    void SetCameras(Camera::Ptr left, Camera::Ptr right) {
        camera_left_ = left;
        camera_right_ = right;
    }

   private:
    /**
     * Track in normal mode
     * @return true if success
     */
    bool Track();

    /**
     * Reset when lost
     * @return true if success
     */
    bool Reset();

    /**
     * Track with last frame
     * @return num of tracked points
     */
    int TrackLastFrame();

    /**
     * estimate current frame's pose
     * @return num of inliers
     */
    int EstimateCurrentPose();

    /**
     * set current frame as a keyframe and insert it into backend
     * @return true if success
     */
    bool InsertKeyframe();

    /**
     * Try init the frontend with stereo images saved in current_frame_
     * @return true if success
     */
    bool StereoInit();

    /**
     * Detect features in left image in current_frame_
     * keypoints will be saved in current_frame_
     * @return
     */
    int DetectFeatures();

    /**
     * Find the corresponding features in right image of current_frame_
     * @return num of features found
     */
    int FindFeaturesInRight();

    /**
     * Build the initial map with single image
     * @return true if succeed
     */
    bool BuildInitMap();

    /**
     * Triangulate the 2D points in current frame
     * @return num of triangulated points
     */
    int TriangulateNewPoints();

    /**
     * Set the features in keyframe as new observation of the map points
     */
    void SetObservationsForKeyFrame();

    // Kimera-VIO functions
    std::vector<cv::KeyPoint> featureDetection(const Frame& cur_frame,
                                              const int& need_n_corners);

    std::vector<cv::KeyPoint> suppressNonMax(
        const std::vector<cv::KeyPoint>& keyPoints,
        const int& numRetPoints,
        const float& tolerance,
        const int& cols,
        const int& rows,
        const int& nr_horizontal_bins,
        const int& nr_vertical_bins,
        const Eigen::MatrixXd& binning_mask);

    std::vector<cv::KeyPoint> binning(
        const std::vector<cv::KeyPoint>& keyPoints,
        const int& numKptsToRetain,
        const int& imgCols,
        const int& imgRows,
        const int& nr_horizontal_bins,
        const int& nr_vertical_bins,
        const Eigen::MatrixXd& binning_mask);

    // data
    FrontendStatus status_ = FrontendStatus::INITING;

    Frame::Ptr current_frame_ = nullptr;  // 현재 프레임
    Frame::Ptr last_frame_ = nullptr;     // 이전 프레임
    Camera::Ptr camera_left_ = nullptr;   // 왼쪽 카메라
    Camera::Ptr camera_right_ = nullptr;  // 오른쪽 카메라

    Map::Ptr map_ = nullptr;
    std::shared_ptr<Backend> backend_ = nullptr;
    std::shared_ptr<Viewer> viewer_ = nullptr;
    std::shared_ptr<Logger> logger_ = nullptr;

    SE3 relative_motion_;  // 현재 프레임과 이전 프레임 사이의 상대 모션으로, 현재 프레임의 초기 포즈 값을 추정하는 데 사용됩니다.

    int tracking_inliers_ = 0;  // inliers, used for testing new keyframes

    // params
    int num_features_ = 200;
    int num_features_init_ = 100;
    int num_features_tracking_ = 50;
    int num_features_tracking_bad_ = 20;
    int num_features_needed_for_keyframe_ = 80;

    // utilities
    cv::Ptr<cv::GFTTDetector> gftt_;  // feature detector in opencv

    // Kimera-VIO variables
    FeatureDetectorParams feature_detector_params_;
};

}  // namespace myslam

#endif  // MYSLAM_FRONTEND_H
