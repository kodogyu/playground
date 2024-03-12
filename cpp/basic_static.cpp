#include <iostream>

struct A {
    A() {
        pvar = new int(3);
    };
    static int* pvar;
};

int main() {
    A a1;
    A a2;

    std::cout << a1.pvar << std::endl;
    std::cout << a2.pvar << std::endl;

    return 0;
}