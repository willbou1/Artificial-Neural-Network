#include "common.h"
#include "ANN.h"
#include "WordClassifier.h"
#include "File.h"

using namespace std;

void printUsage(char *progPath) {
    cout << "Invalid parameters!" << endl;
    cout << "Usage:" << endl;
    cout << stripPath(progPath) << " train file [nbHiddenLayers] [nbNeuronsPerHiddenLayer]" << endl;
    cout << stripPath(progPath) << " test file [input] [input]..." << endl;
}

void printError(Error e) {
    cout << "Error " << e.code << ": " << e.msg << endl;
}

void train(char *trainingFilePath, unsigned int nbHLayers, unsigned int nbHNeurons) {
	File *trainingFile = new File(string(trainingFilePath));
	TrainingSet *trainingSet = trainingFile->readTrainingSet();
	trainingFile->close();
	WordClassifier *wordClassifier = new WordClassifier(trainingSet, nbHLayers, nbHNeurons, 1);
	File *annFile = new File(stripExtension(string(trainingFilePath)) + ".adf");
	annFile->saveWordClassifier(wordClassifier);
	annFile->close();
	delete trainingFile;
	delete annFile;
	delete wordClassifier;
	delete trainingSet;
}

void doTrain(int argc, char *argv[]) {
	if(argc < 3)
		throw Error(1, "The action train takes one argument minimum");
	if (argc > 5)
		throw Error(1, "The action train only takes two arguments maximum");
	//Default values of these arguments
	unsigned int nbHLayers = 4, nbHNeurons = 0;
	if (argc > 3) {
		nbHLayers = atoi(argv[3]);
		if (nbHLayers < 0)
			throw Error(1, "You can't have a negative number of hidden layers");
		if (argc == 5) {
			nbHNeurons = atoi(argv[4]);
			if (nbHNeurons < 0)
				throw Error(1, "You can't have a negative number of neurons per hidden layer");
		}
	}
	train(argv[2], nbHLayers, nbHNeurons);
}

void doTest(int argc, char *argv[]) {
	if (argc < 3)
		throw Error(1, "The action test takes at least one argument");
	File *annFile = new File(string(argv[2]));
	WordClassifier *wordClassifier = annFile->readWordClassifier();
	//If no input is supplied
	if (argc == 3) {
		cout << "Type q or quit to quit " << stripPath(argv[0]) << endl;
		string buffer;
		while(1) {
			cout << "input> ";
			getline(cin, buffer);
			buffer = sanitize(buffer);
			if (buffer == "q" | buffer == "quit")
				break;
			if (buffer.empty())
				continue;
			if (buffer.length() > wordClassifier->getMaxInputLength()) {
				printError(Error(2, "The input exceeds the number of input neurons"));
				continue;
			}
			cout << wordClassifier->probe(buffer) << endl;
		}
	}
	//Testing all the inputs supplied
	for (int i = 0; i < argc - 3; i++) {
		if (strlen(argv[3 + i]) > wordClassifier->getMaxInputLength())
			printError(Error(2, "The input exceeds the number of input neurons"));
		cout << wordClassifier->probe(sanitize(string(argv[3 + i]))) << endl;
	}
	annFile->close();
	delete annFile;
	delete wordClassifier;
}

int main(int argc, char *argv[]) {
    clear();
	try {
		if (argc == 1)
			throw Error(1, "You must specify what action you want to do");
		if (string(argv[1]) == "train") {
			doTrain(argc, argv);
		} else if (string(argv[1]) == "test") {
			doTest(argc, argv);
		} else {
			throw Error(1, string(argv[1]) + " isn't an action you can do");
		}
	} catch (Error e) {
		printError(e);
		if (e.code == 1) {
			printUsage(argv[0]);
		}
	}
	system("pause");
    return 0;
}