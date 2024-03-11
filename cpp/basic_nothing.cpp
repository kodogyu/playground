#include <iostream>
#include <vector>

using namespace std;

int main() {
  //* vector reserve
  // vector<int> v;
  // v.reserve(3);

  // cout << "after reserve: " << v.size() << endl;

  // v.push_back(3);
  // cout << "after push back: " << v.size() << endl;

  // cout << "what: " << std::vector<int>{0, 1, 2}.size() << endl;

  //* initialization
  // vector<int> v;
  // int elem = 0;
  // for (int i = 0; i < 5; i++) {
  //   elem = i;
  //   v.push_back(elem);
  // }
  // cout << "case 1" << endl;
  // for (const int e: v) {
  //   cout << "elem: " << e << endl;
  // }

  // vector<int> v2;
  // for (int i = 0; i < 5; i++) {
  //   int elem2 = i;
  //   v2.push_back(elem2);
  // }
  // cout << "case 2" << endl;
  // for (const int e2: v2) {
  //   cout << "elem: " << e2 << endl;
  // }

  //* <= operator
  for (int i = 0; i <=5; i++) {
    cout << i << endl;
  }
  return 0;
}