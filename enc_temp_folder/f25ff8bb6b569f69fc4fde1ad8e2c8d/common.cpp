#include "common.h"

//TrainingSet
void TrainingSet::computeNbSamples() {
	nbSamples = samples.size();
}

void progressBar(unsigned int *progress, unsigned int range) {
	//Code for the thread that creates and updates a progress bar
	unsigned int pourcentage = 0;
	do {
		pourcentage = 100 * (*progress) / range + 1;
		cout << '\r' << string(80, ' ') << '\r'; //Clearing the line
		cout << '[';
		for (int i = 0; i < pourcentage / 2; i++) //Printing = (pourcentage / 2) times
			cout << '=';
		cout << string(50 - (pourcentage / 2), ' ') << "] "; //Filling the rest of the progress bar with spaces
		cout << setw(2) << setfill('0') << pourcentage << "% (";
		cout << *progress << '/' << range << ')';
		sleep(100);
	} while (pourcentage < 100);
	cout << '\r' << string(80, ' ') << "\rDone!" << endl;
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

string stripExtension(const string &path) {
	//Removes the extension from a file path
    unsigned int lastindex = path.find_last_of("."); 
    return path.substr(0, lastindex); 
}

string stripPath(const string &path) {
	//Returns the name of a file from a path
	unsigned int firstindex = path.find_last_of('\\') + 1;
	return path.substr(firstindex, -1);
}

vector<float> stringToInput(const string &str) {
	//Converts a string to a vector of floats ranging from 0 to 1
	vector<float> ret;
	for (int i = 0; i < str.length(); i++)
		ret.push_back(unify(str[i], 255));
	return ret;
}

string inputToString(const vector<float> &input) {
	string buffer;
	for (int i = 0; i < input.size(); i++)
		buffer += (char)(input[i] * 256);
	return buffer;
}

string sanitize(const string &str, bool toLower) {
	string temp = str, ret;
	//Convert upper case letters, if any, to lower case letters
	if (toLower)
		std::transform(temp.begin(), temp.end(), temp.begin(), tolower);
	//Remove spaces in the string
	for (int i = 0; i < temp.length(); i++)
		if (temp[i] != ' ')
			ret.push_back(temp[i]);
	return str;
}