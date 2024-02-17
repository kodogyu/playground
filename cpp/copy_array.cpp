#include <iostream>

using namespace std;
int main() {
    std::cout << "Copy Array Example" << std::endl;
    std::cout << "====================" << std::endl;

    float dest[3];
    float target[3] = {0.0f, 0.1f, 0.5f};

    // print target
    cout << "target: ";
    for (int i = 0; i < 3; i++) {
        cout << target[i] << ", ";
    }
    cout << endl;

    // copy
    // for (int i = 0; i < 3; i++) {
    //     dest[i] = target[i];
    // }
    copy(begin(target), end(target), begin(dest));

    // print destination
    // for (int i = 0; i < 3; i++) {
    //     cout << dest[i] << ", ";
    // }
    // cout << endl;
    cout << "dest: ";
    for (auto elem : dest) {
        cout << elem << ", ";
    }
    cout << endl;

    return 0;
}