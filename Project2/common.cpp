#include "common.h"

void progressBar(unsigned int *progress, unsigned int range) {
	//Code for the thread that creates and updates a progress bar
	unsigned int pourcentage = 0;
	do {
		pourcentage = 100 * (*progress) / range + 1;
		cout << '\r' << string(105, ' ') << '\r';
		for (int i = 0; i < pourcentage / 2; i++)
			cout << '=';
		cout << "  ";
		cout << setw(2) << setfill('0') << pourcentage << '%';
		sleep();
	} while (pourcentage < 100);
	cout << '\r' << string(105, ' ') << "\rDone!" << endl;
}

float unify(float input, int range) {
	//Converts a float ranging from 0 to range to a float ranging from 0 to 1
    return input / range;
}

void clear() {
    #ifdef _WIN32
    system("cls");
    #elif defined(__APPLE__)
    system("clear");
    #elif defined(__linux__)
    system("clear");
    #endif
}

void sleep(unsigned int time) {
    #ifdef _WIN32
    Sleep(time);
    #elif defined(__APPLE__)
    usleep(time);
    #elif defined(__linux__)
    usleep(time);
    #endif
}

float randToOne() {
	//Generates a random float ranging from 0.0001 to 1
	static unsigned int seed = 0;
    srand(seed);
	seed += 10;
    return (float)(rand() % 100) / 100 + 0.01;
}

string stripExtension(const char *path) {
	//Removes the extension from a file path
    string temp = path;
    int lastindex = temp.find_last_of("."); 
    return temp.substr(0, lastindex); 
}

vector<float> stringToInput(string &input) {
	//Converts a string to a vector of floats ranging from 0 to 1
	vector<float> ret;
	for (int i = 0; i < input.length(); i++)
		ret.push_back(unify(input[i], 255));
	return ret;
}

string inputToString(const vector<float> &input) {
	string buffer;
	for (int i = 0; i < input.size(); i++)
		buffer += (char)(input[i] * 256);
	return buffer;
}