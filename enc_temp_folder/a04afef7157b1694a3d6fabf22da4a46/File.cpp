#include "common.h"
#include "ANN.h"
#include "File.h"

//Public:
File::File(string path) : path(path) {}

File::~File() {
    close();
}

void File::open(fstream::openmode openMode) {
    file.open(path.c_str(), openMode);
    if(!file.is_open())
        throw Error(0, "Couldn't open the file " + path);
    nbLines = countLines();
}

void File::close() {
    file.close();
}

TrainingSet *File::readTrainingSet() {
    cout << "Reading the training set of " << path << endl;
	//Openning the file in read-only mode
	open();
    //Starting progress bar thread
    unsigned int progress = 0;
    thread progressBarThread(&progressBar, &progress, nbLines);
    TrainingSet *ret = new TrainingSet;
    char *buffer = new char[256];
    file.getline(buffer, 256);
    check();
    int nbPossibleOutputs = atoi(buffer), maxLength = 0, len;
    float output;
    for(int i = 0;; i++) {
        //Reading possible outputs
        if(file.eof())
            break;
        file.getline(buffer, 256, ' ');
        check();
        output = 1 / (float)nbPossibleOutputs * (float)i;
        ret->outputMap[output] = string(buffer);
        file.getline(buffer, 256);
        check();
        //Progress bar
        progress++;
        while(1) {
            //Reading sample
            file.getline(buffer, 256);
            if(buffer[0] == '}')
                break;
            check();
            len = strlen(buffer);
            TrainingSample curr;
            curr.output.push_back(output);
            if(maxLength < len)
                maxLength = len;
            for(int j = 0; j < len; j++)
                curr.input.push_back(unify(buffer[j], 256));
            ret->samples.push_back(curr);
            //Progress bar
            progress++;
        }
    }
    ret->maxInputSize = maxLength;
	progressBarThread.join();
	random_shuffle(ret->samples.begin(), ret->samples.end());
    cout << "\rDone!" << endl;
    delete[] buffer;
    return ret;
}

void File::saveANN(ANN *ann) {
	cout << "Saving neural network in " << path << endl;
	//Openning the file in write-only/trunc mode
	open(fstream::out | fstream::trunc);
	unsigned int progress = 0;
    int nbLayers = ann->layerSizes.size();
	//Writing header
    file << nbLayers << ' ' << ann->learningRate << endl;
    for(int i = 0; i < nbLayers; i++)
        file << ann->layerSizes[i] << ' ';
    file << endl;
	//Writing output map
	file << ann->outputMap.size() << endl;
	for (map<float, string>::iterator it = ann->outputMap.begin(); it != ann->outputMap.end(); it++)
		file << it->first << ' ' << it->second << endl;
	//Starting progress bar
	thread progressBarThread(&progressBar, &progress, (ann->nbHLayers * ann->layerSizes[1] + ann->layerSizes[ann->nbHLayers + 1]));
	//Writing neurons
    for(int i = 1; i < nbLayers; i++) {
        for(int j = 0; j < ann->layerSizes[i]; j++) {
            ANN::Neuron *curr = ann->neurons[i][j];
            for(int k = 0; k < curr->weights.size(); k++)
                file << curr->weights[k] << ' ';
            file << endl << curr->bias << endl;
			//Progress bar
			progress++;
        }
    }
	progressBarThread.join();
	cout << "Done!" << endl;
}

ANN *File::readANN() {
	cout << "Reading neural network in " << path << endl;
	//Openning the file in read-only mode
	open();
	unsigned int progress = 0;
    string buffer;
	int nbLayers, nbPossibleOuputs;
	float learningRate, tmp;
	//Reading header
    file >> buffer;
    nbLayers = atoi(buffer.c_str());
	file >> buffer;
	learningRate = stof(buffer);
    vector<int> layerSizes;
    for(int i = 0; i < nbLayers; i++) {
        file >> buffer;
        layerSizes.push_back(atoi(buffer.c_str()));
    }
    ANN *ret = new ANN(layerSizes, learningRate);
	//Reading output map
	file >> buffer;
	nbPossibleOuputs = atoi(buffer.c_str());
	for (int i = 0; i < nbPossibleOuputs; i++) {
		file >> buffer;
		tmp = stof(buffer);
		file >> buffer;
		ret->outputMap[tmp] = buffer;
	}
	//Starting progress bar
	thread progressBarThread(&progressBar, &progress, (ret->nbHLayers * layerSizes[1] + layerSizes[nbLayers - 1]));
	//Reading neurons
    for(int i = 1; i < nbLayers; i++) {
        for(int j = 0; j < layerSizes[i]; j++) {
            ANN::Neuron *curr = ret->neurons[i][j];
            for(int k = 0; k < layerSizes[i - 1]; k++) {
                file >> buffer;
                curr->weights[k] = stof(buffer);
            }
            file >> buffer;
            curr->bias = stof(buffer);
			//Progress bar
			progress++;
        }
    }
	progressBarThread.join();
	cout << "Done!" << endl;
    return ret;
}

//Private:
void File::check() {
    if(file.rdstate() & (fstream::failbit | fstream::badbit))
        throw Error(1, "A problem has occured while reading " + path + '!');
}

int File::countLines() {
    int ret = 0;
    string garbage;
    while(getline(file, garbage))
        ret++;
    file.clear();
    file.seekg(0, fstream::beg);
    return ret;
}