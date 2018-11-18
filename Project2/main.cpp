#include "common.h"
#include "ANN.h"
#include "File.h"

using namespace std;

void printUsage(char *progName) {
    cout << "Invalid parameters!" << endl;
    cout << "Usage:" << endl;
    cout << progName << " train file [nbHiddenLayers] [nbHiddenNeuronsPerLayer]" << endl;
    cout << progName << " test file input" << endl;
	system("pause");
    exit(0);
}

void printError(Error e) {
    cout << "Error " << e.code << ": " << e.msg << endl;
	system("pause");
    exit(0);
}

void train(char *trainingFilePath, unsigned int nbHLayers = 4, unsigned int nbHiddenNeurons = 0) {
    try {
        File *trainingFile = new File(string(trainingFilePath));
        TrainingSet *trainingSet = trainingFile->readTrainingSet();
        trainingFile->close();
        ANN *neuralNet = new ANN(trainingSet, nbHLayers, nbHiddenNeurons, 1);
        File *annFile = new File(stripExtension(trainingFilePath) + ".adf");
        annFile->saveANN(neuralNet);
        annFile->close();
        delete trainingFile;
        delete annFile;
        delete neuralNet;
    } catch(Error e) {
        printError(e);
    }
}

void test(const char *annFilePath, const char *input) {
    try {
        File *annFile = new File(string(annFilePath));
        ANN *neuralNet = annFile->readANN();
		cout << neuralNet->probe(stringToInput(string(input))) << endl;
        annFile->close();
		delete annFile;
		delete neuralNet;
    } catch(Error e) {
        printError(e);
    }
}

int main(int argc, char *argv[]) {
    clear();
    if(argc == 1)
        printUsage(argv[0]);
    if(string(argv[1]) == "train") {
        if(argc = 3)
            train(argv[2]);
        else if(argc = 4)
            train(argv[2], atoi(argv[3]));
        else if(argc = 5)
            train(argv[2], atoi(argv[3]), atoi(argv[4]));
        else
            printUsage(argv[0]);
    } else if(string(argv[1]) == "test") {
        if(argc != 4)
            printUsage(argv[0]);
        test(argv[2], argv[3]);
    } else {
        printUsage(argv[0]);
    }
	system("pause");
    return 0;
}