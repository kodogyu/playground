#include <map>
#include <iostream>
#include <vector>

int main() {
    std::map<int, int> m;
    m[0] = 3;
    m[3] = 7;
    m[6] = 2;

    // 1
    for (auto elem : m) {
        std::cout << elem.first << ", " << elem.second << std::endl;
    }
    std::cout << m.size() << std::endl;
    std::cout << "---------------" << std::endl;


    // 2
    for (int i = 0; i < m.size(); i++) {
        std::cout << "i: " << i << ", " << m.find(i)->second << std::endl;
    }
    std::cout << "---------------" << std::endl;


    // 3
    std::cout << "m.end.second: " << m.end()->second << std::endl;
    std::cout << "---------------" << std::endl;


    // 4
    for (int i = 0; i < 10; i++) {
        std::string str = m.find(i) == m.end() ? "not foudnd" : "found";
        std::cout << "i: " << i << ", " << str << std::endl;
    }
    std::cout << "---------------" << std::endl;


    // // 5
    // for (int i = 0; i < 10; i++) {
    //     std::cout << "i: " << i << ", " << m[i] << std::endl;
    // }
    // for (auto elem : m) {
    //     std::cout << elem.first << ", " << elem.second << std::endl;
    // }
    // std::cout << m.size() << std::endl;
    // std::cout << "---------------" << std::endl;


    // 6
    std::vector<int> vec;
    for (auto elem : m) {
        vec.push_back(m.find(elem.first)->second);

        std::cout << "elem second: " << m.find(elem.first)->second << std::endl;
    }
    std::cout << "---------------" << std::endl;

    return 0;
}