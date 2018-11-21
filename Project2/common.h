#ifndef COMMON_H
#define COMMON_H

#include <thread>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <stdlib.h>
#include <string.h>
#include <time.h>

//Sleep function isn't the same for windows and linux
#ifdef _WIN32
#include <Windows.h>
#elif defined(__APPLE__)
#include <>unistd.h>
#elif defined(__linux__)
#include <>unistd.h>
#endif

using namespace std;

struct TrainingSample {
    vector<float> input;
    vector<float> output;
};

struct TrainingSet {
    TrainingSet() : maxInputSize(0) {}
    map<float, string> outputMap;
    vector<TrainingSample> samples;
    unsigned int maxInputSize;
};

/*Error codes:
0: Error while using files
1: Bad command line arguments
*/
struct Error {
    Error(int code, string msg) : code(code), msg(msg) {}
    int code;
    string msg;
};

void progressBar(unsigned int *progress, unsigned int range);
float unify(float input, int range);
void clear();
void sleep(unsigned int time);
float randToOne();
string stripExtension(const char *path);
vector<float> stringToInput(string &input);
string inputToString(const vector<float> &input);

#endif