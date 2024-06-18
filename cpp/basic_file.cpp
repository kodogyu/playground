#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

int main() {
    // // 1. CSV file

    // ofstream file;
    // int var1 = 10, var2 = 11, var3 = 12;
    // cout << "opening file" << endl;
    // file.open("files/csv_file.csv");

    // file << var1 << "," << var2 << "," << var3 << endl;
    // file << "hello,file";
    // // file << "basic,csv,file\n";
    // // file << "hello,file";

    // cout << "file closing" << endl;
    // file.close();

    // 2. txt file
    cout << "2. txt file" << endl;

    ifstream text_file;
    string line;
    text_file.open("files/feature_frame0.txt");

    if (!text_file.is_open()) {
        cout << "cannot open file" << endl;
        return 0;
    }

    getline(text_file, line);
    cout << line << endl;

    getline(text_file, line);
    cout << line << endl;

    stringstream ss(line);

    string word;
    vector<string> words;

    while (getline(ss, word, ' ')) {
        words.push_back(word);
    }

    for (int i = 0; i < words.size(); i++) {
        string word = words[i];

        if (i == 3) {
            word = word.substr(1, word.size()-2);
        }
        else if (i == 4) {
            word = word.substr(0, word.size()-1);
        }

        // cout << std::stof(word) << endl;
    }
    // cout << endl;

    for (auto word : words) {
        cout << word << endl;
    }

    // while (getline(text_file, line)) {
    //     stringstream ss(line);

    //     string word;
    //     vector<string> words;

    //     while (getline(ss, word, ' ')) {
    //         words.push_back(word);
    //     }
    // }

    text_file.close();

    return 0;
}