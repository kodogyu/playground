#include <iostream>

#include <Eigen/Dense>

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Point3.h>

using namespace std;

int main() {
    // Left camera
    Eigen::Matrix3d body_R_Lcam;
    body_R_Lcam <<  0.99983705,  0.01441438, -0.01086774,
                    -0.01420986,  0.99972476,  0.01866756,
                    0.01113383, -0.01851009,  0.99976668;
    Eigen::Vector3d body_Tran_Lcam;
    body_Tran_Lcam << 0.01691018, 0.00378241, 0.01245686;

    gtsam::Pose3 body_pose_Lcam = gtsam::Pose3::Create(gtsam::Rot3(body_R_Lcam), body_Tran_Lcam);
    
    // Right camera
    Eigen::Matrix3d body_R_Rcam;
    body_R_Rcam <<   0.99988216,  0.01417   , -0.00590628,
                    -0.01405608,  0.9997225 ,  0.01890377,
                     0.00617251, -0.01881853,  0.99980386;
    Eigen::Vector3d body_Tran_Rcam;
    body_Tran_Rcam << 0.00802129, 0.00374016, 0.0125582;

    gtsam::Pose3 body_pose_Rcam = gtsam::Pose3::Create(gtsam::Rot3(body_R_Rcam), body_Tran_Rcam);

    // 생성된 Pose3 객체의 값을 출력합니다.
    // std::cout << "Initial Pose:" << std::endl;
    // std::cout << "Left camera pose:\n" << body_pose_Lcam.matrix() << std::endl;
    // std::cout << "Right camera pose:\n" << body_pose_Rcam.matrix() << std::endl;

    gtsam::Pose3 Lcam_pose_Rcam = body_pose_Lcam.between(body_pose_Rcam);
    // std::cout << "Left Between Right:\n" << Lcam_pose_Rcam.matrix() << std::endl;
    std::cout << "Between Inverse:" << std::endl;
    std::cout << "rotation:\n" << Lcam_pose_Rcam.inverse().rotation().matrix() << std::endl;
    std::cout << "translation:\n" << Lcam_pose_Rcam.inverse().translation().matrix() << std::endl;

    return 0;
}
