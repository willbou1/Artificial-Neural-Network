#include "common.h"
#include "ANN.h"
#include "WordClassifier.h"

///Public:
WordClassifier::WordClassifier(unsigned int maxInputLength, unsigned int nbNeuralNetworks) :
	m_maxInputLength(maxInputLength), m_nbNeuralNetworks(nbNeuralNetworks),
	m_neuralNetworks(vector<ANN *>(maxInputLength)) {}

WordClassifier::WordClassifier(const TrainingSet *trainingSet, unsigned int nbHLayers, unsigned int nbHiddenNeurons, float learningRate) :
	m_maxInputLength(trainingSet->maxInputSize) , m_nbNeuralNetworks(0)
{
	////Classifying training samples (words) by length
	vector<TrainingSet*> trainingSets = classifyTrainingSamples(trainingSet);
	//Creating a neural network for each length of words and training it
	m_neuralNetworks = vector<ANN *>(m_maxInputLength);
	for (int i = 0; i < m_maxInputLength; i++) {
		if (trainingSets[i]) {
			cout << "Generating a neural network for the words of " << trainingSets[i]->maxInputSize << " characters (" << i + 1 << '/' << m_nbNeuralNetworks << "):" << endl;
			m_neuralNetworks[trainingSets[i]->maxInputSize - 1] = new ANN(trainingSets[i], nbHLayers, nbHiddenNeurons, learningRate);
		}
	}
	//Deleting training sets
	for (int i = 0; i < m_maxInputLength; i++) {
		if (trainingSets[i])
			delete trainingSets[i];
	}
}

WordClassifier::~WordClassifier() {
	//Deleting neural networks
	for (int i = 0; i < m_maxInputLength; i++) {
		if (m_neuralNetworks[i])
			delete m_neuralNetworks[i];
	}
}

unsigned int WordClassifier::getMaxInputLength() const {
	return m_maxInputLength;
}

unsigned int WordClassifier::getNbNeuralNetworks() const {
	unsigned int ret = 0;
	for (int i = 0; i < m_maxInputLength; i++) {
		if (m_neuralNetworks[i])
			ret++;
	}
	return ret;
}

string WordClassifier::probe(const string &input) {
	//Checking the length of the input and probing the associated neural network
	return m_neuralNetworks[input.length()]->probe(stringToInput(input));
}

//Private:
vector<TrainingSet*> WordClassifier::classifyTrainingSamples(const TrainingSet *trainingSet) {
	vector<TrainingSet*> trainingSets(m_maxInputLength); //Vector of training sets, one for each length of words
	//Classifying training samples (words) by length
	for (int i = 0; i < trainingSet->nbSamples; i++) {
		unsigned int sampleLength = trainingSet->samples[i].inputSize;
		TrainingSet *&currSet = trainingSets[sampleLength - 1];
		if (!currSet) {
			currSet = new TrainingSet(trainingSet->outputMap, sampleLength);
			m_nbNeuralNetworks++;
		}
		currSet->samples.push_back(trainingSet->samples[i]);
	}
	//Computing the number of samples for each training set
	for (int i = 0; i < m_maxInputLength; i++)
		trainingSets[i]->computeNbSamples();
	return trainingSets;
}