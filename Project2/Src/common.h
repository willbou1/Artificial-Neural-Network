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
#include <unistd.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

using namespace std;

struct TrainingSample {
	TrainingSample() : inputSize(0), outputSize(0){}
	TrainingSample(unsigned int inputSize, unsigned int outputSize) :
		inputSize(inputSize),
		outputSize(outputSize) {}
	vector<float> input;
    vector<float> output;
	unsigned int inputSize;
	unsigned int outputSize;
};

struct TrainingSet {
    TrainingSet(map<float, string> outputMap = map<float, string>(), unsigned int maxInputSize = 0) :
		maxInputSize(maxInputSize),
		nbSamples(0),
		outputMap(outputMap) {}
	void computeNbSamples();
    map<float, string> outputMap;
    vector<TrainingSample> samples;
    unsigned int maxInputSize;
	unsigned int nbSamples;
};

/*Error codes:
0: Error while using files
1: Bad command line arguments
*/
struct Error {
    Error(int code, string msg) :
		code(code),
		msg(msg) {}
    int code;
    string msg;
};

void progressBar(unsigned int *progress, unsigned int range);
float unify(float input, int range);
void clear();
void sleep(unsigned int time);
float randToOne();
string stripExtension(const string &path);
string stripPath(const string &path);
vector<float> stringToInput(const string &str);
string inputToString(const vector<float> &input);
string sanitize(const string &str, bool toLower = true);

#endif