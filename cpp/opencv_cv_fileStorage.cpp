#include <iostream>
#include <opencv2/opencv.hpp>
// #include <cv_bridge/cv_bridge.h>

using namespace std;

int main() {
    // std::string filepath = "/home/kodogyu/playground/visual_odometry/config/l515_stereo.yaml";
    std::string filepath = "/home/kodogyu/playground/cpp/files/visual_odometry_example.yaml";
    cv::FileStorage fs = cv::FileStorage(filepath.c_str(), cv::FileStorage::READ);
    std::string s;

    // fs["dataset_dir"] >> s;

    // cout << "opencv version: " << CV_VERSION << endl;
    // cout << "dataset dir: " << s << endl;

    cv::FileNode fn = fs["images_left"];
    for (int i = 0; i < 5; i++) {
        std::string str = fn[i];
        cout << "i = " << i << ": " << str << endl;
    }
    return 0;
}