#include <iostream>
#include <vector>

using namespace std;

int main() {
  vector<int> v(3, 1);
  cout << "vector size: " << v.size() << endl;

  for (int elem: v) {
    cout << "elem: " << elem << endl;
  }

  cout << "vector[-1]: " << v[-1] << endl;

  vector<int> v0(0, 3);
  v0.push_back(2);
  cout << "v0: " << *v0.begin() << endl;
  return 0;
}