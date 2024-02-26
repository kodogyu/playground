/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <fstream>

#include "myslam/map.h"
#include "myslam/feature.h"

namespace myslam {

void Map::InsertKeyFrame(Frame::Ptr frame) {
    current_frame_ = frame;
    if (keyframes_.find(frame->keyframe_id_) == keyframes_.end()) {
        keyframes_.insert(make_pair(frame->keyframe_id_, frame));
        active_keyframes_.insert(make_pair(frame->keyframe_id_, frame));
    } else {
        keyframes_[frame->keyframe_id_] = frame;
        active_keyframes_[frame->keyframe_id_] = frame;
    }

    if (active_keyframes_.size() > num_active_keyframes_) {
        RemoveOldKeyframe();
    }
}

void Map::InsertMapPoint(MapPoint::Ptr map_point) {
    if (landmarks_.find(map_point->id_) == landmarks_.end()) {
        landmarks_.insert(make_pair(map_point->id_, map_point));
        active_landmarks_.insert(make_pair(map_point->id_, map_point));
    } else {
        landmarks_[map_point->id_] = map_point;
        active_landmarks_[map_point->id_] = map_point;
    }
}

void Map::RemoveOldKeyframe() {
    if (current_frame_ == nullptr) return;
    // 현재 프레임에서 가장 가까운 키프레임과 가장 먼 두 개의 키프레임을 찾습니다.
    double max_dis = 0, min_dis = 9999;
    double max_kf_id = 0, min_kf_id = 0;
    auto Twc = current_frame_->Pose().inverse();
    for (auto& kf : active_keyframes_) {
        if (kf.second == current_frame_) continue;
        auto dis = (kf.second->Pose() * Twc).log().norm();
        if (dis > max_dis) {
            max_dis = dis;
            max_kf_id = kf.first;
        }
        if (dis < min_dis) {
            min_dis = dis;
            min_kf_id = kf.first;
        }
    }

    const double min_dis_th = 0.2;  // 최단거리 임계값
    Frame::Ptr frame_to_remove = nullptr;
    if (min_dis < min_dis_th) {
        // 매우 가까운 프레임이 있는 경우 가장 가까운 프레임을 먼저 삭제하세요.
        frame_to_remove = keyframes_.at(min_kf_id);
    } else {
        // 가장 먼 것을 삭제
        frame_to_remove = keyframes_.at(max_kf_id);
    }

    // LOG(INFO) << "log removing keyframe pose";
    // // log removing keyframe pose
    // std::ofstream log_file;
    // log_file.open("/home/kodogyu/slambook2/ch13/output_logs/paths.csv");

    // auto pose = frame_to_remove->Pose();
    // Eigen::Quaterniond quat = pose.unit_quaternion();
    // Eigen::Vector3d trans = pose.translation();

    // log_file << "%f,%f,%f,%f,", quat.w(), quat.x(), quat.y(), quat.z();  // rotation
    // log_file << "%f,%f,%f\n", trans.x(), trans.y(), trans.z();  // translation
    // log_file.close();

    LOG(INFO) << "remove keyframe " << frame_to_remove->keyframe_id_;
    // remove keyframe and landmark observation
    active_keyframes_.erase(frame_to_remove->keyframe_id_);
    for (auto feat : frame_to_remove->features_left_) {
        auto mp = feat->map_point_.lock();
        if (mp) {
            mp->RemoveObservation(feat);
        }
    }
    for (auto feat : frame_to_remove->features_right_) {
        if (feat == nullptr) continue;
        auto mp = feat->map_point_.lock();
        if (mp) {
            mp->RemoveObservation(feat);
        }
    }

    CleanMap();
}

void Map::CleanMap() {
    int cnt_landmark_removed = 0;
    for (auto iter = active_landmarks_.begin();
         iter != active_landmarks_.end();) {
        if (iter->second->observed_times_ == 0) {
            iter = active_landmarks_.erase(iter);
            cnt_landmark_removed++;
        } else {
            ++iter;
        }
    }
    LOG(INFO) << "Removed " << cnt_landmark_removed << " active landmarks";
}

}  // namespace myslam
