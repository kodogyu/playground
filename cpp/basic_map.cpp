#include <map>
#include <iostream>

int main() {
    std::map<int, int> m;
    m[0] = 3;
    m[3] = 7;
    m[6] = 2;

    // for (auto elem : m) {
    //     std::cout << elem.first << ", " << elem.second << std::endl;
    // }

    std::cout << m.size() << std::endl;

    return 0;
}