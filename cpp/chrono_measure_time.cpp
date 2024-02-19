#include <iostream>
#include <chrono>

int main(int argc, char* argv[]) {
    std::cout << "Time Measure Example" << std::endl;
    std::cout << "====================" << std::endl;

    // 시작 시각
    const std::chrono::time_point<std::chrono::steady_clock> start = 
        std::chrono::steady_clock::now();

    // 함수 실행
    for (auto i = 0; i < 1000000; i++);

    // 종료 시각
    const auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> sec = end - start;
    std::cout << "elapsed time: " << sec.count() << std::endl;
}