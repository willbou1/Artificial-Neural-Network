#ifndef WORDCLASSIFIER_H
#define WORDCLASSIFIER_H

using namespace std;

class WordClassifier {
public:
	WordClassifier(unsigned int maxInputLength, unsigned int nbNeuralNetworks);
	WordClassifier(const TrainingSet *trainingSet, unsigned int nbHLayers, unsigned int nbHiddenNeurons, float learningRate = 0.5);
	~WordClassifier();
	unsigned int getMaxInputLength() const;
	unsigned int getNbNeuralNetworks() const;
	string probe(const string &input);
private:
	vector<TrainingSet*> classifyTrainingSamples(const TrainingSet *trainingSet);
	vector<ANN *> m_neuralNetworks;
	unsigned int m_maxInputLength;
	unsigned int m_nbNeuralNetworks;
	friend class File;
};

#endif
