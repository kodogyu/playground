//
// Created by gaoxiang on 19-5-2.
//

#ifndef MYSLAM_BACKEND_H
#define MYSLAM_BACKEND_H

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/map.h"

namespace myslam {
class Map;

/**
 * 백엔드
 * 맵이 업데이트될 때 최적화를 시작하는 별도의 최적화 스레드가 있습니다.
 * 지도 업데이트는 프런트엔드에 의해 트리거됩니다.
 */ 
class Backend {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Backend> Ptr;

    /// 생성자에서 최적화 스레드를 시작하고 정지시킵니다.
    Backend();

    // 왼쪽, 오른쪽 카메라 설정 및 내부, 외부 매개변수 가져오기
    void SetCameras(Camera::Ptr left, Camera::Ptr right) {
        cam_left_ = left;
        cam_right_ = right;
    }

    /// 지도 설정
    void SetMap(std::shared_ptr<Map> map) { map_ = map; }

    /// 지도 업데이터, 최적화 시작
    void UpdateMap();

    /// 백엔드 스레드 정지
    void Stop();

   private:
    /// 백엔드 스레드
    void BackendLoop();

    /// 주어진 키프레임 및 랜드마크 포인트 최적화
    void Optimize(Map::KeyframesType& keyframes, Map::LandmarksType& landmarks);

    std::shared_ptr<Map> map_;
    std::thread backend_thread_;
    std::mutex data_mutex_;

    std::condition_variable map_update_;
    std::atomic<bool> backend_running_;

    Camera::Ptr cam_left_ = nullptr, cam_right_ = nullptr;
};

}  // namespace myslam

#endif  // MYSLAM_BACKEND_H