#include <iostream>
#include <vector>
#include <string>
#include <fstream>

using namespace std;

unsigned int nbWords[20] = {
	2, 30, 114, 311, 583,
	692, 778, 771, 613, 456,
	279, 195, 100, 52, 14,
	5, 3, 1, 1, 0
};

int main() {
	vector<vector<string>> words(20);
	const char *path = "English words.txt";
	fstream file;
	file.open(path, fstream::in);
	if (!file.is_open())
		return 1;
	string line;
	bool isFrench = true;
	while (getline(file, line)) {
		unsigned int wordLength = line.length();
		if (wordLength < 20) {
			if (words[wordLength - 1].size() < nbWords[wordLength - 1]) {
				words[wordLength - 1].push_back(line);
			}
		}
		isFrench = !isFrench;
	}
	file.close();
	file.open(path, fstream::out | fstream::trunc);
	for (int i = 0; i < 20; i++) {
		for (int j = 0; j < words[i].size(); j++)
			file << words[i][j] << endl;
	}
	file.close();
	system("pause");
	return 0;
}