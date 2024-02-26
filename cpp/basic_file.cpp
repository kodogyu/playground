#include <fstream>
#include <iostream>

using namespace std;

int main() {
    ofstream file;
    int var1 = 10, var2 = 11, var3 = 12;
    cout << "opening file" << endl;
    file.open("files/csv_file.csv");

    file << var1 << "," << var2 << "," << var3 << endl;
    file << "hello,file";
    // file << "basic,csv,file\n";
    // file << "hello,file";

    cout << "file closing" << endl;
    file.close();

    return 0;
}