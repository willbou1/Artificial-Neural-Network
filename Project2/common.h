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
#include <Windows.h>
#include <time.h>

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
float randToOne();
string stripExtension(const char *path);
vector<float> stringToInput(string &input);
string inputToString(const vector<float> &input);

#endif