#ifndef MYSLAM_DATASET_H
#define MYSLAM_DATASET_H
#include "myslam/camera.h"
#include "myslam/common_include.h"
#include "myslam/frame.h"

namespace myslam {

/**
 * 데이터 세트 읽기
 * 구성 파일 경로는 생성 중에 전달되며 구성 파일의 dataset_dir는 데이터세트 경로입니다.
 * 초기화 후 카메라와 다음 프레임 이미지를 얻을 수 있습니다.
 */
class Dataset {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Dataset> Ptr;
    Dataset(const std::string& dataset_path);

    /// 初始化，返回是否成功
    bool Init();

    /// create and return the next frame containing the stereo images
    Frame::Ptr NextFrame();

    /// get camera by id
    Camera::Ptr GetCamera(int camera_id) const {
        return cameras_.at(camera_id);
    }

   private:
    std::string dataset_path_;
    int current_image_index_ = 0;

    std::vector<Camera::Ptr> cameras_;

    std::vector<std::string> frame_ids_;
};
}  // namespace myslam

#endif