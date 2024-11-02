#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <string>

int main(int argc, char** argv) {
    std::ifstream file(argv[1]); // CSV 파일 경로
    std::ofstream txt_file(argv[2]);  // txt 파일 경로

    if (!file.is_open() || !txt_file.is_open()) {
        std::cerr << "파일을 열 수 없습니다.\n";
        return 1;
    }

    std::string line;
    std::vector<size_t> row_lengths;

    while (std::getline(file, line)) {
        size_t pos = line.find(',');
        std::string timestamp_str;

        timestamp_str = line.substr(0, pos);

        txt_file << timestamp_str << std::endl;
    }

    file.close();
    txt_file.close();

    return 0;
}
