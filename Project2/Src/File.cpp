#include "common.h"
#include "ANN.h"
#include "WordClassifier.h"
#include "File.h"

//Public:
File::File(string path) : m_path(path) {}

File::~File() {
    close();
}

void File::open(fstream::openmode openMode) {
    m_file.open(m_path.c_str(), openMode);
    if(!m_file.is_open())
        throw Error(0, "Couldn't open the file " + m_path);
    m_nbLines = countLines();
}

void File::close() {
    m_file.close();
}

TrainingSet *File::readTrainingSet() {
    cout << "Reading the training set of " << m_path << endl;
	open(); //Openning the file in read-only mode
    unsigned int progress = 0;
    thread progressBarThread(&progressBar, &progress, m_nbLines); //Starting the progress bar's thread
    TrainingSet *ret = new TrainingSet;
	string buffer;
    getline(m_file, buffer);
    check();
    int nbPossibleOutputs = stoi(buffer), maxLength = 0, len;
    float output;
    for(int i = 0;; i++) {
        //Reading possible outputs
        if(m_file.eof())
            break;
		getline(m_file, buffer);
        check();
        output = 1 / (float)nbPossibleOutputs * (float)i;
        ret->outputMap[output] = buffer;
		getline(m_file, buffer);
        check();
        progress++; //Progress bar
        while(1) {
			getline(m_file, buffer); //Reading sample
            if(buffer[0] == '}')
                break;
            check();
			len = buffer.length();
            TrainingSample curr(len, 1);
            curr.output.push_back(output);
            if(maxLength < len)
                maxLength = len;
			curr.input = stringToInput(buffer);
            ret->samples.push_back(curr);
            progress++; //Progress bar
        }
    }
    ret->maxInputSize = maxLength;
	ret->computeNbSamples();
	random_shuffle(ret->samples.begin(), ret->samples.end());
	progressBarThread.join();
    return ret;
}

void File::saveANN(ANN *ann) {
	cout << "Saving the neural network in " << m_path << endl;
	open(fstream::out | fstream::trunc); //Openning the file in write-only/trunc mode
	unsigned int progress = 0;
    int nbLayers = ann->m_layerSizes.size();
	//Writing header
    m_file << nbLayers << ' ' << ann->m_learningRate << endl;
    for(int i = 0; i < nbLayers; i++)
        m_file << ann->m_layerSizes[i] << ' ';
    m_file << endl;
	//Writing output map
	m_file << ann->m_outputMap.size() << endl;
	for (map<float, string>::iterator it = ann->m_outputMap.begin(); it != ann->m_outputMap.end(); it++)
		m_file << it->first << ' ' << it->second << endl;
	//Starting progress bar
	thread progressBarThread(&progressBar, &progress, (ann->m_nbHLayers * ann->m_layerSizes[1] + ann->m_layerSizes[ann->m_nbHLayers + 1]));
	//Writing neurons
    for(int i = 1; i < nbLayers; i++) {
        for(int j = 0; j < ann->m_layerSizes[i]; j++) {
            ANN::Neuron *curr = ann->m_neurons[i][j];
            for(int k = 0; k < curr->weights.size(); k++)
                m_file << curr->weights[k] << ' ';
            m_file << endl << curr->bias << endl;
			progress++; //Progress bar
        }
    }
	progressBarThread.join();
}

ANN *File::readANN() {
	cout << "Reading neural network in " << m_path << endl;
	open(); //Openning the file in read-only mode
	unsigned int progress = 0;
    string buffer;
	int nbLayers, nbPossibleOuputs;
	float learningRate, tmp;
	//Reading header
    m_file >> buffer;
    nbLayers = stoi(buffer);
	m_file >> buffer;
	learningRate = stof(buffer);
    vector<unsigned int> layerSizes;
    for(int i = 0; i < nbLayers; i++) {
        m_file >> buffer;
        layerSizes.push_back(stoi(buffer));
    }
    ANN *ret = new ANN(layerSizes, learningRate);
	//Reading output map
	m_file >> buffer;
	nbPossibleOuputs = stoi(buffer);
	for (int i = 0; i < nbPossibleOuputs; i++) {
		m_file >> buffer;
		tmp = stof(buffer);
		m_file >> buffer;
		ret->m_outputMap[tmp] = buffer;
	}
	//Starting progress bar
	thread progressBarThread(&progressBar, &progress, (ret->m_nbHLayers * layerSizes[1] + layerSizes[nbLayers - 1]));
	//Reading neurons
    for(int i = 1; i < nbLayers; i++) {
        for(int j = 0; j < layerSizes[i]; j++) {
            ANN::Neuron *curr = ret->m_neurons[i][j];
            for(int k = 0; k < layerSizes[i - 1]; k++) {
                m_file >> buffer;
                curr->weights[k] = stof(buffer);
            }
            m_file >> buffer;
            curr->bias = stof(buffer);
			progress++; //Progress bar
        }
    }
	progressBarThread.join();
    return ret;
}

void File::saveWordClassifier(WordClassifier *wordClassifier) {
	cout << "Saving the word classifier:" << endl;
	unsigned int nbNeuralNets = wordClassifier->getNbNeuralNetworks();
	open(fstream::out | fstream::trunc); //Openning the file in write-only/trunc mode
	m_file << nbNeuralNets << ' ' << wordClassifier->m_maxInputLength << endl;
	//Saving all the neural networks in seperate files
	for (int i = 0; i < nbNeuralNets; i++) {
		unsigned int inputLength = wordClassifier->m_neuralNetworks[i]->m_layerSizes[0];
		string currPath = stripExtension(m_path) + to_string(inputLength) + ".adf";
		File *currFile = new File(currPath);
		currFile->saveANN(wordClassifier->m_neuralNetworks[i]);
		delete currFile;
		//Writing the current neural network's file's path and the associated length of words
		m_file << currPath << ' ' << inputLength << endl;
	}
}

WordClassifier *File::readWordClassifier() {
	open(); //Openning the file in read-only mode
	string buffer;
	unsigned int nbNeuralNets, maxInputLength;
	m_file >> buffer;
	nbNeuralNets = stoi(buffer);
	m_file >> buffer;
	maxInputLength = stoi(buffer);
	WordClassifier *ret = new WordClassifier(maxInputLength, countLines() - 1);
	//Reading all the neural networks
	for (int i = 0; i < nbNeuralNets; i++) {
		m_file >> buffer;
		string currPath = buffer;
		m_file >> buffer;
		unsigned int inputLength = stoi(buffer);
		File *currFile = new File(currPath);
		ret->m_neuralNetworks[inputLength - 1] = currFile->readANN();
		delete currFile;
	}
	return ret;
}

//Private:
void File::check() {
	//Checking if any error happened while using the file
    if(m_file.rdstate() & (fstream::failbit | fstream::badbit))
        throw Error(0, "A problem has occured while reading/writing " + m_path + '!');
}

int File::countLines() {
    int ret = 0;
    string garbage;
	m_file.seekg(0, fstream::beg);
    while(getline(m_file, garbage))
        ret++;
    m_file.clear();
	m_file.seekg(0, fstream::beg);
    return ret;
}