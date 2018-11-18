#include "common.h"

void progressBar(unsigned int *progress, unsigned int range) {
	//Code for the thread that creates and updates a progress bar
	unsigned int pourcentage = 0;
	do {
		pourcentage = 100 * (*progress) / range;
		cout << string(105, ' ') << '\r';
		for (int i = 0; i < pourcentage; i++)
			cout << '=';
		cout << setw(2) << setfill('0') << "  ";
		cout << pourcentage << '%';
		Sleep(500);
	} while (pourcentage < 99);
	cout << string(105, ' ') << "\rDone!";
}

float unify(float input, int range) {
	//Converts a float ranging from 0 to range to a float ranging from 0 to 1
    return input / range;
}

void clear() {
    system("cls");
}

float randToOne() {
	//Generates a random float ranging from 0.0001 to 1
	static unsigned int seed = 0;
    srand(seed);
	seed += 10;
    return (float)(rand() % 10000) / 10000 + 0.0001;
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
		ret.push_back(input[i]);
	return ret;
}