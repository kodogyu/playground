#include <iostream>
#include <zip.h>
#include <opencv2/opencv.hpp>

int main() {
    // ZIP 파일 열기
    const char *zip_filename = "/home/kodogyu/Downloads/images.zip";
    int err = 0;
    zip *z = zip_open(zip_filename, 0, &err);
    if (z == nullptr) {
        zip_error_t ziperror;
        zip_error_init_with_code(&ziperror, err);
        std::cerr << "Failed to open zip file: " << zip_error_strerror(&ziperror) << std::endl;
        zip_error_fini(&ziperror);
        return 1;
    }

    // ZIP 파일 내 이미지 파일 열기
    const char *image_filename = "0000.png";
    struct zip_stat st;
    zip_stat_init(&st);
    zip_stat(z, image_filename, 0, &st);

    // 이미지 파일 읽기
    zip_file *f = zip_fopen(z, image_filename, 0);
    if (f == nullptr) {
        std::cerr << "Failed to open file inside zip: " << image_filename << std::endl;
        zip_close(z);
        return 1;
    }

    // 이미지 데이터를 버퍼에 읽기
    char *contents = new char[st.size];
    zip_fread(f, contents, st.size);
    zip_fclose(f);

    // zip 파일 닫기
    zip_close(z);

    // OpenCV로 이미지 디코딩
    std::vector<uchar> data(contents, contents + st.size);
    cv::Mat img = cv::imdecode(data, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Failed to decode image" << std::endl;
        delete[] contents;
        return 1;
    }

    // 이미지 출력
    cv::imshow("Image", img);
    cv::waitKey(0);

    // 메모리 해제
    delete[] contents;
    return 0;
}