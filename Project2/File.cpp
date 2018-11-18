#include "common.h"
#include "ANN.h"
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
	//Openning the file in read-only mode
	open();
    //Starting progress bar thread
    unsigned int progress = 0;
    thread progressBarThread(&progressBar, &progress, m_nbLines);
    TrainingSet *ret = new TrainingSet;
    char *buffer = new char[256];
    m_file.getline(buffer, 256);
    check();
    int nbPossibleOutputs = atoi(buffer), maxLength = 0, len;
    float output;
    for(int i = 0;; i++) {
        //Reading possible outputs
        if(m_file.eof())
            break;
		m_file.getline(buffer, 256, ' ');
        check();
        output = 1 / (float)nbPossibleOutputs * (float)i;
        ret->outputMap[output] = string(buffer);
		m_file.getline(buffer, 256);
        check();
        //Progress bar
        progress++;
        while(1) {
            //Reading sample
			m_file.getline(buffer, 256);
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
	random_shuffle(ret->samples.begin(), ret->samples.end());
    delete[] buffer;
	progressBarThread.join();
    return ret;
}

void File::saveANN(ANN *ann) {
	cout << "Saving neural network in " << m_path << endl;
	//Openning the file in write-only/trunc mode
	open(fstream::out | fstream::trunc);
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
			//Progress bar
			progress++;
        }
    }
	progressBarThread.join();
}

ANN *File::readANN() {
	cout << "Reading neural network in " << m_path << endl;
	//Openning the file in read-only mode
	open();
	unsigned int progress = 0;
    string buffer;
	int nbLayers, nbPossibleOuputs;
	float learningRate, tmp;
	//Reading header
    m_file >> buffer;
    nbLayers = atoi(buffer.c_str());
	m_file >> buffer;
	learningRate = stof(buffer);
    vector<int> layerSizes;
    for(int i = 0; i < nbLayers; i++) {
        m_file >> buffer;
        layerSizes.push_back(atoi(buffer.c_str()));
    }
    ANN *ret = new ANN(layerSizes, learningRate);
	//Reading output map
	m_file >> buffer;
	nbPossibleOuputs = atoi(buffer.c_str());
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
			//Progress bar
			progress++;
        }
    }
	progressBarThread.join();
    return ret;
}

//Private:
void File::check() {
    if(m_file.rdstate() & (fstream::failbit | fstream::badbit))
        throw Error(1, "A problem has occured while reading " + m_path + '!');
}

int File::countLines() {
    int ret = 0;
    string garbage;
    while(getline(m_file, garbage))
        ret++;
    m_file.clear();
	m_file.seekg(0, fstream::beg);
    return ret;
}