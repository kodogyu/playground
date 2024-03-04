#include <iostream>
#include <vector>

using namespace std;

int main() {
  vector<int> v;
  v.reserve(3);

  cout << "after reserve: " << v.size() << endl;

  v.push_back(3);
  cout << "after push back: " << v.size() << endl;

  cout << "what: " << std::vector<int>{0, 1, 2}.size() << endl;

  return 0;
}