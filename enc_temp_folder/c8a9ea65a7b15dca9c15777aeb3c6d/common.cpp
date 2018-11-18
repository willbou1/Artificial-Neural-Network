#include "common.h"

void progressBar(unsigned int *progress, unsigned int range) {
	int pourcentage = 0;
	do {
		pourcentage = 100 * (*progress) / range;
		cout << setw(2) << setfill('0');
		cout << pourcentage << '%' << '\r';
		Sleep(500);
	} while (pourcentage < 99);
}

float unify(float input, int range) {
    return input / range;
}

void clear() {
    system("cls");
}

float randToOne() {
	static unsigned int seed = 0;
    srand(seed);
	seed += 10;
    return (float)(rand() % 10000) / 10000 + 0.0001;
}

string stripExtension(const char *path) {
    string temp = path;
    int lastindex = temp.find_last_of("."); 
    return temp.substr(0, lastindex); 
}

vector<float> stringToInput(string &input) {
	vector<float> ret;
	for (int i = 0; i < input.length(); i++)
		ret.push_back(input[i]);
	return ret;
}