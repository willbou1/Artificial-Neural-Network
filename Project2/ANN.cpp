#include "common.h"
#include "ANN.h"
#include "File.h"

ANN::Neuron::Neuron(unsigned int prevLayerSize) : activation(0) {
    //Initializing everything randomly if this isn’t an input neuron
    if(prevLayerSize) {
        for(int i = 0; i < prevLayerSize; i++)
            weights.push_back(randToOne());
        bias = randToOne();
    }
}

//Public:
ANN::ANN(unsigned int nbHLayers, unsigned int nbInputNeurons, unsigned int nbHiddenNeurons, unsigned int nbOutputNeurons, float learningRate) : m_nbHLayers(nbHLayers), m_learningRate(learningRate) {
	initLayerSizes(nbInputNeurons, nbHiddenNeurons, nbOutputNeurons);
    initNeurons();
}

ANN::ANN(const TrainingSet *trainingSet, unsigned int nbHLayers, unsigned  nbHiddenNeurons = 0, float learningRate) : m_nbHLayers(nbHLayers), m_learningRate(learningRate) {
    if(!nbHiddenNeurons)
        nbHiddenNeurons = (int)(1.5 * (float)trainingSet->maxInputSize);
	initLayerSizes(trainingSet->maxInputSize, nbHiddenNeurons, 1);
	initNeurons();
    train(trainingSet, 200);
}

ANN::ANN(const vector<int> &layerSizes, float learningRate) : m_layerSizes(layerSizes), m_learningRate(learningRate) {
	m_nbHLayers = layerSizes.size() - 2;
	initNeurons();
}

ANN::~ANN() {
    for(int i = 0; i < m_nbHLayers + 2; i++) {
        for(int j = 0; j < m_layerSizes[i]; j++)
            delete m_neurons[i][j];
    }
}

void ANN::initLayerSizes(unsigned int nbInputNeurons, unsigned int nbHiddenNeurons, unsigned int nbOutputNeurons) {
	//Initializing a vector with all the layers' size
	m_layerSizes.push_back(nbInputNeurons);
	for (int i = 0; i < m_nbHLayers; i++)
		m_layerSizes.push_back(nbHiddenNeurons);
	m_layerSizes.push_back(nbOutputNeurons);
}

void ANN::initNeurons() {
    //Creating the neurons
    vector<Neuron*> temp;
    for(int i = 0; i < m_layerSizes[0]; i++)
        temp.push_back(new Neuron(0));
	m_neurons.push_back(temp);
    temp.clear();
    for(int i = 1; i < m_nbHLayers + 2; i++) {
        for(int j = 0; j < m_layerSizes[i]; j++)
            temp.push_back(new Neuron(m_layerSizes[i - 1]));
		m_neurons.push_back(temp);
        temp.clear();
    }
}

void ANN::train(const TrainingSet *trainingSet, unsigned int nbIterations) {
	m_outputMap = trainingSet->outputMap;
	unsigned int progress = 0, nbSamples = trainingSet->samples.size();
	cout << "Training the neural network" << endl;
	thread progressBarThread(&progressBar, &progress, nbSamples * nbIterations);
    for(int i = 0; i < nbSamples * nbIterations; i++) {
        propagate(trainingSet->samples[i % nbSamples].input);
		backpropagate(trainingSet->samples[i % nbSamples].output, m_nbHLayers + 1);
		progress++;
	}
	progressBarThread.join();
	cout << "Done!" << endl;
}

string ANN::probe(const vector<float> &inputs) {
	if (!m_outputMap.size())
		return string();
	propagate(inputs);
	return mapOutput(m_neurons[m_nbHLayers + 1][0]->activation);
}

//Private:
void ANN::propagate(const vector<float> &inputs) {
	//Setting the first layer's neurons' activations to the value of the inputs
	for (int i = 0; i < m_layerSizes[0]; i++) {
		if (i < inputs.size())
			m_neurons[0][i]->activation = inputs[i];
		else
			m_neurons[0][i]->activation = 0;
	}
	//Propagating the input layer through the other layers
	for (int i = 1; i < m_nbHLayers + 2; i++) {
		for (int j = 0; j < m_layerSizes[i]; j++) {
			Neuron *curr = m_neurons[i][j];
			float activation = curr->bias;
			for (int k = 0; k < m_layerSizes[i - 1]; k++)
				activation += m_neurons[i - 1][k]->activation * curr->weights[k];
			curr->activation = activate(activation, i);
		}
	}
}

void ANN::propagate(unsigned int startingLayer) {
	for (int i = startingLayer; i < m_nbHLayers + 2; i++) {
		for (int j = 0; j < m_layerSizes[i]; j++) {
			Neuron *curr = m_neurons[i][j];
			float activation = curr->bias;
			for (int k = 0; k < m_layerSizes[i - 1]; k++)
				activation += m_neurons[i - 1][k]->activation * curr->weights[k];
			curr->activation = activate(activation, i);
		}
	}
}

void ANN::backpropagate(const vector<float> &outputs, unsigned int layer) {
	//Initializing a multidimensional vector of gradients
	static vector<vector<vector<float>>> gradients;
	gradients = vector<vector<vector<float>>>(m_nbHLayers + 2);
	for (int i = 0; i < m_nbHLayers + 2; i++)
		gradients[i] = vector<vector<float>>(m_layerSizes[i]);
	float dBase, derivative;
	//Calculating the gradients of the output layer
	int nbOutputNeurons = m_layerSizes[m_nbHLayers + 1];
	for (int i = 0; i < nbOutputNeurons; i++) {
		//dBase (dC/dz) = dz(cost(act)) = da(cost) * dz(act)
		dBase = (m_neurons[m_nbHLayers + 1][i]->activation - outputs[i]);
		dBase *= (1 / (float)(m_layerSizes[m_nbHLayers] + 1));
		for (int j = 0; j < m_layerSizes[m_nbHLayers]; j++) {
			//derivative (dC/dw) = dBase * dw(wSums)
			derivative = dBase * m_neurons[m_nbHLayers][j]->activation;
			gradients[m_nbHLayers + 1][i].push_back(derivative);
		}
		//derivative (dC/db) = dBase * db(wSums)
		gradients[m_nbHLayers + 1][i].push_back(dBase);
	}
	//Updating weights of output neurons
	if (layer == m_nbHLayers + 1) {
		for (int i = 0; i < nbOutputNeurons; i++)
			updateWeights(gradients[m_nbHLayers + 1][i], m_neurons[m_nbHLayers + 1][i]);
		//Repropagating
		propagate(m_nbHLayers + 1);
		//Backpropagate layers before layer
		backpropagate(outputs, layer - 1);
		return;
	}
	//Calculating the gradients of the hidden layers and updating the weights
	for (int i = m_nbHLayers; i > 0; i--) {
		for (int j = 0; j < m_layerSizes[i]; j++) {
			//dBase (dC/da) where a is the activation of this neuron
			dBase = 0;
			for (int k = 0; k < m_layerSizes[i + 1]; k++)
				dBase += gradients[i + 1][k][j] * m_neurons[i + 1][k]->weights[j];
			//dBase (dC/dz) = dBase * dz(wSums)
			dBase *= (1 / (float)(m_layerSizes[i - 1] + 1));
			for (int k = 0; k < m_layerSizes[i - 1]; k++) {
				//derivative (dC/dw) = dBase * dw(wSums)
				derivative = dBase * m_neurons[i - 1][k]->activation;
				gradients[i][j].push_back(derivative);
			}
			gradients[i][j].push_back(dBase);
		}
		//Updating weights
		if (layer == i) {
			for (int j = 0; j < m_layerSizes[i]; j++)
				updateWeights(gradients[i][j], m_neurons[i][j]);
			//Repropagate
			propagate(i);
			//Backpropagate layers before layer
			backpropagate(outputs, layer - 1);
			return;
		}
	}
}

string ANN::mapOutput(float output) {
    if(m_outputMap.size() == 0)
        return "";
    float smallestDistance = 1;
    string ret;
    for(map<float, string>::iterator it = m_outputMap.begin(); it != m_outputMap.end(); it++) {
        float curr = it->first;
        if(output > curr)
            curr = output - curr;
        else
            curr -= output;
        if(curr < smallestDistance) {
            smallestDistance = curr;
            ret = it->second;
        }
    }
    return ret;
}

float ANN::activate(float input, int layer) {
    return input / (m_layerSizes[layer - 1] + 1);
}

void ANN::updateWeights(const vector<float> &gradient, Neuron *neuron) {
	float updatedWeight, updatedBias;
	for (int i = 0; i < neuron->weights.size(); i++) {
		updatedWeight = neuron->weights[i] + -gradient[i] * m_learningRate;
		//Making sure that the updated weight sits between 0 and 1
		if (updatedWeight < 0)
			neuron->weights[i] = 0;
		else if(updatedWeight > 1)
			neuron->weights[i] = 1;
		else
			neuron->weights[i] = updatedWeight;
	}
	updatedBias = neuron->bias + (-gradient.back() * m_learningRate);
	//Making sure that the updated bias sits between 0 and 1
	if (updatedBias < 0)
		neuron->bias = 0;
	else if (updatedBias > 1)
		neuron->bias = 1;
	else
		neuron->bias = updatedBias;
}

float ANN::findCost(const vector<float> &outputs) {
	float ret = 0;
	vector<Neuron*> &outputLayer = m_neurons[m_nbHLayers + 1];
	for (int i = 0; i < outputLayer.size(); i++)
		ret += (0.5 * (outputLayer[i]->activation - outputs[i]) * (outputLayer[i]->activation - outputs[i]));
	return ret;
}