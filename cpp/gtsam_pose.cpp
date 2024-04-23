#include <iostream>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <Eigen/Dense>

using namespace std;

int main() {
    // // 초기 회전과 위치를 지정하여 Pose3 객체를 생성합니다.
    // gtsam::Rot3 initialRotation = gtsam::Rot3::Quaternion(1.0, 0.0, 0.0, 0.0); // Quaternion (1, 0, 0, 0) : no rotation
    // gtsam::Point3 initialTranslation(0.0, 0.0, 0.0); // Translation (0, 0, 0)
    // gtsam::Pose3 pose(initialRotation, initialTranslation);

    // // 생성된 Pose3 객체의 값을 출력합니다.
    // std::cout << "Initial Pose:" << std::endl;
    // std::cout << "Rotation:\n" << pose.rotation().matrix() << std::endl;
    // std::cout << "Translation:\n" << pose.translation().matrix() << std::endl;

    // // 새로운 회전과 위치 값을 지정하여 Pose3 객체에 할당합니다.
    // gtsam::Rot3 newRotation = gtsam::Rot3::Ry(0.1); // 회전 각도가 0.1 라디안인 회전 행렬을 생성합니다.
    // gtsam::Point3 newTranslation(1.0, 2.0, 3.0); // (1, 2, 3)으로 이동합니다.
    // pose = gtsam::Pose3(newRotation, newTranslation);

    // // 새로 할당된 Pose3 객체의 값을 출력합니다.
    // std::cout << "\nNew Pose:" << std::endl;
    // std::cout << "Rotation:\n" << pose.rotation().matrix() << std::endl;
    // std::cout << "Translation:\n" << pose.translation().matrix() << std::endl;

    gtsam::Rot3 rotation = gtsam::Rot3::Quaternion(0.999998,0.00061395,0.00201124,0.000332651);
    std::cout << "rotation:\n" << rotation.matrix().row(0) << rotation.matrix().row(1) << rotation.matrix().row(2) << std::endl;
    return 0;
}
