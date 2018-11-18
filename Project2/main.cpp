#include "common.h"
#include "ANN.h"
#include "File.h"

using namespace std;

void printUsage(char *progName) {
    cout << "Invalid parameters!" << endl;
    cout << "Usage:" << endl;
    cout << progName << " train file [nbHiddenLayers] [nbNeuronsPerHiddenLayer]" << endl;
    cout << progName << " test file input [input]..." << endl;
	system("pause");
    exit(0);
}

void printError(Error e) {
    cout << "Error " << e.code << ": " << e.msg << endl;
}

void train(char *trainingFilePath, unsigned int nbHLayers, unsigned int nbHNeurons) {
	File *trainingFile = new File(string(trainingFilePath));
	TrainingSet *trainingSet = trainingFile->readTrainingSet();
	trainingFile->close();
	ANN *neuralNet = new ANN(trainingSet, nbHLayers, nbHNeurons, 1);
	File *annFile = new File(stripExtension(trainingFilePath) + ".adf");
	annFile->saveANN(neuralNet);
	annFile->close();
	delete trainingFile;
	delete annFile;
	delete neuralNet;
}

void doTrain(int argc, char *argv[]) {
	if(argc < 3)
		throw Error(1, "The action train takes one argument minimum");
	if (argc > 4)
		throw Error(1, "The action train only takes two arguments maximum");
	//Default values of these arguments
	unsigned int nbHLayers = 4, nbHNeurons = 0;
	if (argc > 3) {
		nbHLayers = atoi(argv[2]);
		if (nbHLayers < 0)
			throw Error(1, "You can't have a negative number of hidden layers");
		if (argc == 5) {
			nbHNeurons = atoi(argv[3]);
			if (nbHNeurons < 0)
				throw Error(1, "You can't have a negative number of neurons per hidden layer");
		}
	}
	train(argv[2], nbHLayers, nbHNeurons);
}

void doTest(int argc, char *argv[]) {
	if (argc < 4)
		throw Error(1, "The action test takes two argument minimum");
	File *annFile = new File(string(argv[2]));
	ANN *neuralNet = annFile->readANN();
	//Testing all the inputs
	for(int i = 0; i < argc - 3; i++)
		cout << neuralNet->probe(stringToInput(string(argv[3 + i]))) << endl;
	annFile->close();
	delete annFile;
	delete neuralNet;
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
			system("pause");
			return 1;
		}
	}
	system("pause");
    return 0;
}