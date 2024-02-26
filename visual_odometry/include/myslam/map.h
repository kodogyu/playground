#pragma once
#ifndef MAP_H
#define MAP_H

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/mappoint.h"

namespace myslam {

/**
 * @brief 지도
 * 지도와의 상호 작용: 
 *  프런트 엔드는 InsertKeyframe 및 InsertMapPoint를 호출하여 새 프레임과 지도 지점을 삽입하고, 
 *  백 엔드는 지도 구조를 유지하고 이상값/제거 등을 결정합니다.
 */
class Map {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Map> Ptr;
    typedef std::unordered_map<unsigned long, MapPoint::Ptr> LandmarksType;
    typedef std::unordered_map<unsigned long, Frame::Ptr> KeyframesType;

    Map() {}

    /// 키프레임 추가
    void InsertKeyFrame(Frame::Ptr frame);
    /// 지도에 정점 추가
    void InsertMapPoint(MapPoint::Ptr map_point);

    /// 모든 지도 포인트 가져오기
    LandmarksType GetAllMapPoints() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return landmarks_;
    }
    /// 모든 키프레임 가져오기
    KeyframesType GetAllKeyFrames() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return keyframes_;
    }

    /// 활성 맵 포인트 가져오기
    LandmarksType GetActiveMapPoints() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_landmarks_;
    }

    /// 활성 키프레임 가져오기
    KeyframesType GetActiveKeyFrames() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_keyframes_;
    }

    /// 관측 수가 0인 맵의 포인트 정리
    void CleanMap();

   private:
    // 이전 키프레임 비활성화
    void RemoveOldKeyframe();

    std::mutex data_mutex_;
    LandmarksType landmarks_;         // all landmarks
    LandmarksType active_landmarks_;  // active landmarks
    KeyframesType keyframes_;         // all key-frames
    KeyframesType active_keyframes_;  // active key-frames

    Frame::Ptr current_frame_ = nullptr;

    // settings
    int num_active_keyframes_ = 7;  // 활성 키프레임 수
};
}  // namespace myslam

#endif  // MAP_H
