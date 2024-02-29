#include <iostream>
#include <thread>
#include <mutex>

void thread_func(int &counter) {
    for (int i = 0; i < 100000; i++) {
        counter++;
    }
}

void thread_safe_func(int &counter, std::mutex &m) {
    for (int i = 0; i < 100000; i++) {
        m.lock();
        counter++;  // critical section
        m.unlock();
    }
}

void thread_safe_func2(int &counter, std::mutex &m) {
    for (int i = 0; i < 100000; i++) {
        std::lock_guard<std::mutex> lock(m);
        counter++;  // critical section
    }
}

int main() {
    // not thread safe
    std::cout << "thread unsafe function:" << std::endl;
    int counter = 0;

    std::thread t1(thread_func, std::ref(counter));
    std::thread t2(thread_func, std::ref(counter));
    std::thread t3(thread_func, std::ref(counter));
    
    t1.join();
    t2.join();
    t3.join();
    std::cout << "counter result: " << counter << std::endl << std::endl;

    // thread safe
    std::cout << "thread safe function:" << std::endl;
    counter = 0;
    std::mutex m;
    
    std::thread t4(thread_safe_func, std::ref(counter), std::ref(m));
    std::thread t5(thread_safe_func, std::ref(counter), std::ref(m));
    std::thread t6(thread_safe_func, std::ref(counter), std::ref(m));

    t4.join();
    t5.join();
    t6.join();
    std::cout << "counter result: " << counter << std::endl << std::endl;

    // thread safe with lock_guard
    std::cout << "thread safe function with lock_guard:" << std::endl;
    counter = 0;
    std::mutex m2;
    
    std::thread t7(thread_safe_func2, std::ref(counter), std::ref(m2));
    std::thread t8(thread_safe_func2, std::ref(counter), std::ref(m2));
    std::thread t9(thread_safe_func2, std::ref(counter), std::ref(m2));

    t7.join();
    t8.join();
    t9.join();
    std::cout << "counter result: " << counter << std::endl;

    return 0;
}