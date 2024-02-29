#include <iostream>
#include <thread>

using namespace std;

void func1() {
    for (int i = 0; i < 10; i++) {
        cout << "thread1! " << endl;
    }
}

void func2() {
    for (int i = 0; i < 10; i++) {
        cout << "thread2! " << endl;
    }
}

void func3() {
    for (int i = 0; i < 10; i++) {
        cout << "thread3! " << endl;
    }
}

int main() {
    thread t1(func1);
    thread t2(func2);
    thread t3(func3);

    t1.join();
    t2.join();
    t3.join();

    return 0;
}