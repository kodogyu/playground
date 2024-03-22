#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>

int main() {
    std::filesystem::path p("/home/kodogyu/playground/test_image_sequences/l515_aligned_vertical_stereo_kf/left_frames");
    std::filesystem::directory_iterator itr(p);
    // std::filesystem::directory_iterator itr(std::filesystem::current_path());

    std::vector<std::string> files;

    while (itr != std::filesystem::end(itr)) {
        const std::filesystem::directory_entry entry = *itr;
        files.push_back(entry.path());
        itr++;
    }

    std::sort(files.begin(), files.end());
    for (const auto entry : files) {
        std::cout << entry << std::endl;
    }
    return 0;
}