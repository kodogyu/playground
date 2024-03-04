//
// Created by gaoxiang on 19-5-2.
//

#include <opencv2/opencv.hpp>

#include "myslam/algorithm.h"
#include "myslam/backend.h"
#include "myslam/config.h"
#include "myslam/feature.h"
#include "myslam/frontend.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"
#include "myslam/viewer.h"

namespace myslam {

Frontend::Frontend() {
    gftt_ =
        cv::GFTTDetector::create(Config::Get<int>("num_features"), 0.01, 20);
    num_features_init_ = Config::Get<int>("num_features_init");
    num_features_ = Config::Get<int>("num_features");
}

bool Frontend::AddFrame(myslam::Frame::Ptr frame) {
    current_frame_ = frame;

    switch (status_) {
        case FrontendStatus::INITING:
            StereoInit();
            break;
        case FrontendStatus::TRACKING_GOOD:
        case FrontendStatus::TRACKING_BAD:
            Track();
            break;
        case FrontendStatus::LOST:
            Reset();
            break;
    }

    last_frame_ = current_frame_;
    return true;
}

bool Frontend::Track() {
    if (last_frame_) {
        current_frame_->SetPose(relative_motion_ * last_frame_->Pose());
    }

    int num_track_last = TrackLastFrame();
    tracking_inliers_ = EstimateCurrentPose();

    if (tracking_inliers_ > num_features_tracking_) {
        // tracking good
        status_ = FrontendStatus::TRACKING_GOOD;
    } else if (tracking_inliers_ > num_features_tracking_bad_) {
        // tracking bad
        status_ = FrontendStatus::TRACKING_BAD;
    } else {
        // lost
        status_ = FrontendStatus::LOST;
    }

    InsertKeyframe();
    relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();

    if (viewer_) viewer_->AddCurrentFrame(current_frame_);
    return true;
}

bool Frontend::InsertKeyframe() {
    if (tracking_inliers_ >= num_features_needed_for_keyframe_) {
        // still have enough features, don't insert keyframe
        return false;
    }
    // current frame is a new keyframe
    current_frame_->SetKeyFrame();
    map_->InsertKeyFrame(current_frame_);

    LOG(INFO) << "Set frame " << current_frame_->id_ << " as keyframe "
              << current_frame_->keyframe_id_;

    SetObservationsForKeyFrame();
    // DetectFeatures();  // detect new features


    std::vector<cv::KeyPoint> new_features;
    int need_n_corners = std::max(
        feature_detector_params_.max_features_per_frame_ - current_frame_->features_left_.size(), ulong(0));
    new_features = featureDetection(*current_frame_, need_n_corners);

    for (int i = 0; i < new_features.size(); i++) {
        std::shared_ptr<Feature> new_feature = std::make_shared<Feature>();
        new_feature->frame_ = current_frame_;
        new_feature->position_ = new_features[i];
        current_frame_->features_left_.push_back(new_feature);
    }
    LOG(INFO) << "Detect " << new_features.size() << " new features";
    logger_->logImage("left_image" + std::to_string(current_frame_->id_), current_frame_->left_img_);

    // track in right image
    FindFeaturesInRight();
    // triangulate map points
    TriangulateNewPoints();
    // update backend because we have a new keyframe
    backend_->UpdateMap();

    if (viewer_) viewer_->UpdateMap();

    return true;
}

void Frontend::SetObservationsForKeyFrame() {
    for (auto &feat : current_frame_->features_left_) {
        auto mp = feat->map_point_.lock();
        if (mp) mp->AddObservation(feat);
    }
}

int Frontend::TriangulateNewPoints() {
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    SE3 current_pose_Twc = current_frame_->Pose().inverse();
    int cnt_triangulated_pts = 0;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        if (current_frame_->features_left_[i]->map_point_.expired() &&
            current_frame_->features_right_[i] != nullptr) {
            // 왼쪽 이미지의 특징점은 지도 포인트와 연결되어 있지 않으며 오른쪽 이미지에는 일치하는 포인트가 있습니다.삼각측량을 시도해보세요.
            std::vector<Vec3> points{
                camera_left_->pixel2camera(
                    Vec2(current_frame_->features_left_[i]->position_.pt.x,
                         current_frame_->features_left_[i]->position_.pt.y)),
                camera_right_->pixel2camera(
                    Vec2(current_frame_->features_right_[i]->position_.pt.x,
                         current_frame_->features_right_[i]->position_.pt.y))};
            Vec3 pworld = Vec3::Zero();

            if (triangulation(poses, points, pworld) && pworld[2] > 0) {
                auto new_map_point = MapPoint::CreateNewMappoint();
                pworld = current_pose_Twc * pworld;
                new_map_point->SetPos(pworld);
                new_map_point->AddObservation(
                    current_frame_->features_left_[i]);
                new_map_point->AddObservation(
                    current_frame_->features_right_[i]);

                current_frame_->features_left_[i]->map_point_ = new_map_point;
                current_frame_->features_right_[i]->map_point_ = new_map_point;
                map_->InsertMapPoint(new_map_point);
                cnt_triangulated_pts++;
            }
        }
    }
    LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;
    return cnt_triangulated_pts;
}

int Frontend::EstimateCurrentPose() {
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
        LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // vertex
    VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(current_frame_->Pose());
    optimizer.addVertex(vertex_pose);

    // K
    Mat33 K = camera_left_->K();

    // edges
    int index = 1;
    std::vector<EdgeProjectionPoseOnly *> edges;
    std::vector<Feature::Ptr> features;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        auto mp = current_frame_->features_left_[i]->map_point_.lock();
        if (mp) {
            features.push_back(current_frame_->features_left_[i]);
            EdgeProjectionPoseOnly *edge =
                new EdgeProjectionPoseOnly(mp->pos_, K);
            edge->setId(index);
            edge->setVertex(0, vertex_pose);
            edge->setMeasurement(
                toVec2(current_frame_->features_left_[i]->position_.pt));
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->setRobustKernel(new g2o::RobustKernelHuber);
            edges.push_back(edge);
            optimizer.addEdge(edge);
            index++;
        }
    }

    // estimate the Pose the determine the outliers
    const double chi2_th = 5.991;
    int cnt_outlier = 0;
    for (int iteration = 0; iteration < 4; ++iteration) {
        vertex_pose->setEstimate(current_frame_->Pose());
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        cnt_outlier = 0;

        // count the outliers
        for (size_t i = 0; i < edges.size(); ++i) {
            auto e = edges[i];
            if (features[i]->is_outlier_) {
                e->computeError();
            }
            if (e->chi2() > chi2_th) {
                features[i]->is_outlier_ = true;
                e->setLevel(1);
                cnt_outlier++;
            } else {
                features[i]->is_outlier_ = false;
                e->setLevel(0);
            };

            if (iteration == 2) {
                e->setRobustKernel(nullptr);
            }
        }
    }

    LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
              << features.size() - cnt_outlier;
    // Set pose and outlier
    current_frame_->SetPose(vertex_pose->estimate());

    LOG(INFO) << "Current Pose = \n" << current_frame_->Pose().matrix();

    LOG(INFO) << "Logging pose...";
    logger_->logPose(current_frame_->Pose());

    for (auto &feat : features) {
        if (feat->is_outlier_) {
            feat->map_point_.reset();
            feat->is_outlier_ = false;  // maybe we can still use it in future
        }
    }
    return features.size() - cnt_outlier;
}

int Frontend::TrackLastFrame() {
    // use LK flow to estimate points in the right image
    std::vector<cv::Point2f> kps_last, kps_current;
    for (auto &kp : last_frame_->features_left_) {
        if (kp->map_point_.lock()) {
            // use project point
            auto mp = kp->map_point_.lock();
            auto px =
                camera_left_->world2pixel(mp->pos_, current_frame_->Pose());
            kps_last.push_back(kp->position_.pt);
            kps_current.push_back(cv::Point2f(px[0], px[1]));
        } else {
            kps_last.push_back(kp->position_.pt);
            kps_current.push_back(kp->position_.pt);
        }
    }

    std::vector<uchar> status;
    Mat error;
    cv::calcOpticalFlowPyrLK(
        last_frame_->left_img_, current_frame_->left_img_, kps_last,
        kps_current, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    int num_good_pts = 0;

    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            cv::KeyPoint kp(kps_current[i], 7);
            Feature::Ptr feature(new Feature(current_frame_, kp));
            feature->map_point_ = last_frame_->features_left_[i]->map_point_;
            current_frame_->features_left_.push_back(feature);
            num_good_pts++;
        }
    }

    LOG(INFO) << "Find " << num_good_pts << " in the last image.";
    return num_good_pts;
}

bool Frontend::StereoInit() {
    int num_features_left = DetectFeatures();
    int num_coor_features = FindFeaturesInRight();
    if (num_coor_features < num_features_init_) {
        return false;
    }

    bool build_map_success = BuildInitMap();
    if (build_map_success) {
        status_ = FrontendStatus::TRACKING_GOOD;
        if (viewer_) {
            viewer_->AddCurrentFrame(current_frame_);
            viewer_->UpdateMap();
        }
        return true;
    }
    return false;
}

int Frontend::DetectFeatures() {
    cv::Mat mask(current_frame_->left_img_.size(), CV_8UC1, 255);
    for (auto &feat : current_frame_->features_left_) {
        cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                      feat->position_.pt + cv::Point2f(10, 10), 0, cv::FILLED);
    }

    std::vector<cv::KeyPoint> keypoints;
    // cv::Mat detecting_image;
    // cv::cvtColor(current_frame_->left_img_, detecting_image, cv::COLOR_GRAY2BGR);
    gftt_->detect(current_frame_->left_img_, keypoints, mask);
    int cnt_detected = 0;
    for (auto &kp : keypoints) {
        current_frame_->features_left_.push_back(
            Feature::Ptr(new Feature(current_frame_, kp)));
        cnt_detected++;
    }

    LOG(INFO) << "Detect " << cnt_detected << " new features";
    return cnt_detected;
}

int Frontend::FindFeaturesInRight() {
    // use LK flow to estimate points in the right image
    std::vector<cv::Point2f> kps_left, kps_right;
    for (auto &kp : current_frame_->features_left_) {
        kps_left.push_back(kp->position_.pt);
        auto mp = kp->map_point_.lock();
        if (mp) {
            // use projected points as initial guess
            auto px =
                camera_right_->world2pixel(mp->pos_, current_frame_->Pose());
            kps_right.push_back(cv::Point2f(px[0], px[1]));
        } else {
            // use same pixel in left iamge
            kps_right.push_back(kp->position_.pt);
        }
    }

    std::vector<uchar> status;
    Mat error;
    cv::calcOpticalFlowPyrLK(
        current_frame_->left_img_, current_frame_->right_img_, kps_left,
        kps_right, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    int num_good_pts = 0;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            cv::KeyPoint kp(kps_right[i], 7);
            Feature::Ptr feat(new Feature(current_frame_, kp));
            feat->is_on_left_image_ = false;
            current_frame_->features_right_.push_back(feat);
            num_good_pts++;
        } else {
            current_frame_->features_right_.push_back(nullptr);
        }
    }
    LOG(INFO) << "Find " << num_good_pts << " in the right image.";

    // log matched keypoint image
    LOG(INFO) << "Logging Matched keypoint image...";
    logger_->logFeatureMatchImages(current_frame_);

    return num_good_pts;
}

bool Frontend::BuildInitMap() {
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    size_t cnt_init_landmarks = 0;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        if (current_frame_->features_right_[i] == nullptr) continue;
        // create map point from triangulation
        std::vector<Vec3> points{
            camera_left_->pixel2camera(
                Vec2(current_frame_->features_left_[i]->position_.pt.x,
                     current_frame_->features_left_[i]->position_.pt.y)),
            camera_right_->pixel2camera(
                Vec2(current_frame_->features_right_[i]->position_.pt.x,
                     current_frame_->features_right_[i]->position_.pt.y))};
        Vec3 pworld = Vec3::Zero();

        if (triangulation(poses, points, pworld) && pworld[2] > 0) {
            auto new_map_point = MapPoint::CreateNewMappoint();
            new_map_point->SetPos(pworld);
            new_map_point->AddObservation(current_frame_->features_left_[i]);
            new_map_point->AddObservation(current_frame_->features_right_[i]);
            current_frame_->features_left_[i]->map_point_ = new_map_point;
            current_frame_->features_right_[i]->map_point_ = new_map_point;
            cnt_init_landmarks++;
            map_->InsertMapPoint(new_map_point);
        }
    }
    current_frame_->SetKeyFrame();
    map_->InsertKeyFrame(current_frame_);
    backend_->UpdateMap();

    LOG(INFO) << "Initial map created with " << cnt_init_landmarks
              << " map points";

    return true;
}

bool Frontend::Reset() {
    LOG(INFO) << "Reset is not implemented. ";
    return true;
}


// Source Code from Kimera-VIO
// https://github.com/MIT-SPARK/Kimera-VIO


std::vector<cv::KeyPoint> Frontend::featureDetection(const Frame& cur_frame,
                                              const int& need_n_corners) {

    cv::Mat mask(cur_frame.left_img_.size(), CV_8UC1, 255);
    for (auto &feat : cur_frame.features_left_) {
        cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                      feat->position_.pt + cv::Point2f(10, 10), 0, cv::FILLED);
        cv::circle(mask,
                 feat->position_.pt,
                 feature_detector_params_
                     .min_distance_btw_tracked_and_detected_features_,
                 cv::Scalar(0),
                 cv::FILLED);
    }

  // Actual raw feature detection
  std::vector<cv::KeyPoint> keypoints;
    gftt_->detect(cur_frame.left_img_, keypoints, mask);
  LOG(INFO) << "Number of points detected : " << keypoints.size();

  LOG(INFO) << "Need n corners: " << need_n_corners;
  // Tolerance of the number of returned points in percentage.
  std::vector<cv::KeyPoint>& max_keypoints = keypoints;
    static constexpr float tolerance = 0.1;
    max_keypoints = suppressNonMax(
        keypoints,
        need_n_corners,
        tolerance,
        cur_frame.left_img_.cols,
        cur_frame.left_img_.rows,
        feature_detector_params_.nr_horizontal_bins_,
        feature_detector_params_.nr_vertical_bins_,
        feature_detector_params_.binning_mask_);
  // NOTE: if we don't use max_suppression we may end with more corners than
  // requested...

  // TODO(Toni): we should be using cv::KeyPoint... not cv::Point2f...

  return max_keypoints;
}

std::vector<cv::KeyPoint> Frontend::suppressNonMax(
    const std::vector<cv::KeyPoint>& keyPoints,
    const int& numRetPoints,
    const float& tolerance,
    const int& cols,
    const int& rows,
    const int& nr_horizontal_bins,
    const int& nr_vertical_bins,
    const Eigen::MatrixXd& binning_mask) {
  if (keyPoints.size() == 0) {
    LOG(WARNING) << "No keypoints for non-max suppression...";
    return std::vector<cv::KeyPoint>();
  }

  // Sorting keypoints by deacreasing order of strength
  VLOG(5) << "Sorting keypoints in decreasing order of strength.";
  std::vector<int> responseVector;
  for (unsigned int i = 0; i < keyPoints.size(); i++) {
    responseVector.push_back(keyPoints[i].response);
  }
  std::vector<int> Indx(responseVector.size());
  std::iota(std::begin(Indx), std::end(Indx), 0);
  cv::sortIdx(responseVector, Indx, cv::SortFlags::SORT_DESCENDING);
  std::vector<cv::KeyPoint> keyPointsSorted;
  for (unsigned int i = 0; i < keyPoints.size(); i++) {
    keyPointsSorted.push_back(keyPoints[Indx[i]]);
  }

  std::vector<cv::KeyPoint> keypoints;
  VLOG(5) << "Starting Adaptive Non-Maximum Suppression.";
    keypoints = binning(keyPointsSorted,
                        numRetPoints,
                        cols,
                        rows,
                        nr_horizontal_bins,
                        nr_vertical_bins,
                          binning_mask);
  return keypoints;
}

std::vector<cv::KeyPoint> Frontend::binning(
    const std::vector<cv::KeyPoint>& keyPoints,
    const int& numKptsToRetain,
    const int& imgCols,
    const int& imgRows,
    const int& nr_horizontal_bins,
    const int& nr_vertical_bins,
    const Eigen::MatrixXd& binning_mask) {
  if (static_cast<size_t>(numKptsToRetain) > keyPoints.size()) {
    return keyPoints;
  }

  float binRowSize = float(imgRows) / float(nr_vertical_bins);
  float binColSize = float(imgCols) / float(nr_horizontal_bins);

  // Note: features should be already sorted by score at this point from detect

  // 0. count the number of valid bins (as specified by the user in the yaml
  float nrActiveBins = binning_mask.sum();  // sum of 1's in binary mask

  // 1. compute how many features we want to retain in each bin
  // numRetPointsPerBin
  const int numRetPointsPerBin =
      std::round(float(numKptsToRetain) / float(nrActiveBins));

  // 2. assign keypoints to bins and retain top numRetPointsPerBin for each bin
  std::vector<cv::KeyPoint> binnedKpts;  // binned keypoints we want to output
  Eigen::MatrixXd nrKptsInBin = Eigen::MatrixXd::Zero(
      nr_vertical_bins,
      nr_horizontal_bins);  // store number of kpts for each bin
  for (size_t i = 0; i < keyPoints.size(); i++) {
    const size_t binRowInd =
        static_cast<size_t>(keyPoints[i].pt.y / binRowSize);
    const size_t binColInd =
        static_cast<size_t>(keyPoints[i].pt.x / binColSize);
    // if bin is active and needs more keypoints
    if (binning_mask(binRowInd, binColInd) == 1 &&
        nrKptsInBin(binRowInd, binColInd) <
            numRetPointsPerBin) {  // if we need more kpts in that bin
      binnedKpts.push_back(keyPoints[i]);
      nrKptsInBin(binRowInd, binColInd) += 1;
    }
  }
  return binnedKpts;
}

}  // namespace myslam