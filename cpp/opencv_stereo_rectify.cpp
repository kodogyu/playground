#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

int main() {
    // 카메라 매트릭스 및 왜곡 계수 설정
    cv::Mat cameraMatrix1 = (cv::Mat_<double>(3, 3) << 620.070096090849, 0, 325.29844703787114, 0, 618.2102185572654, 258.48711395621467, 0, 0, 1); // 왼쪽 카메라 매트릭스
    cv::Mat cameraMatrix2 = (cv::Mat_<double>(3, 3) << 620.7205430589681, 0, 322.27715209601246, 0, 618.5483414745424, 258.3905254657877, 0, 0, 1); // 오른쪽 카메라 매트릭스
    cv::Mat distCoeffs1 = (cv::Mat_<double>(1, 4) << 0.14669701, -0.27353153,  0.00730068, -0.003734); // 왼쪽 카메라 왜곡 계수
    cv::Mat distCoeffs2 = (cv::Mat_<double>(1, 4) << 0.14148691, -0.26292268,  0.00708119, -0.00551033); // 오른쪽 카메라 왜곡 계수
    
    cv::Mat camL_Rot_camR = (cv::Mat_<double>(3, 3) << 0.999988,  0.000246216,  -0.00495778,
                                         -0.000247748,            1, -0.000305755,
                                           0.00495771,  0.000306972,     0.999988);
    cv::Mat camL_Tran_camR = (cv::Mat_<double>(1, 3) <<   0.00888662, 0.000170101, -0.000153022);
    cv::Size imageSize(640, 480); // 이미지 크기

    cv::Mat R1, R2, P1, P2, Q;
    cv::Rect *Roi1, *Roi2;

    // 스테레오 보정을 위한 보정 매개변수 계산
    // cv::stereoRectify(cameraMatrix1,
    //                   distCoeffs1, 
    //                   cameraMatrix2,
    //                   distCoeffs2,
    //                   imageSize,
    //                   camL_Rot_camR, 
    //                   camL_Tran_camR, 
    //                   R1, 
    //                   R2, 
    //                   P1, 
    //                   P2, 
    //                   Q, 
    //                   cv::CALIB_ZERO_DISPARITY,
    //                   0,
    //                   cv::Size(),
    //                   Roi1,
    //                   Roi2);

    cout << "Size:\n" << imageSize << endl;
    // cout << "cameraMatrix1:\n" << cameraMatrix1 << endl;
    // cout << "distCoeffs1:\n" << distCoeffs1.rows << ", " << distCoeffs1.cols << endl;
    return 0;
}