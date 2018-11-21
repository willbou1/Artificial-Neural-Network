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

ANN::Neuron::Neuron(const Neuron &a) : activation(a.activation), bias(a.bias) {
	weights = vector<float>(a.weights);
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
    train(trainingSet, 1);
}

ANN::ANN(const vector<unsigned int> &layerSizes, float learningRate) : m_layerSizes(layerSizes), m_learningRate(learningRate) {
	m_nbHLayers = layerSizes.size() - 2;
	initNeurons();
}

ANN::ANN(const ANN &a) : m_nbHLayers(a.m_nbHLayers), m_learningRate(a.m_learningRate) {
	//Copying neurons
	m_neurons = vector<vector<Neuron*>>(a.m_nbHLayers + 2);
	for (int i = 0; i < a.m_nbHLayers + 2; i++) {
		for (int j = 0; j < a.m_layerSizes[i]; j++) {
			m_neurons[i].push_back(new Neuron(*(a.m_neurons[i][j])));
		}
	}
	m_layerSizes = vector<unsigned int>(a.m_layerSizes);
	m_outputMap = map<float, string>(a.m_outputMap);
}

ANN::~ANN() {
    for(int i = 0; i < m_nbHLayers + 2; i++) {
        for(int j = 0; j < m_layerSizes[i]; j++)
            delete m_neurons[i][j];
    }
}

unsigned int ANN::getNbInputNeurons() {
	if (!m_layerSizes.empty())
		return m_layerSizes[0];
	return 0;
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
	vector<ANN> optimizations; //Stores the optimized config of the neural network for each sample
	cout << "Training the neural network" << endl;
	thread progressBarThread(&progressBar, &progress, nbSamples * nbIterations);
    for(int i = 0; i < nbIterations; i++) {
		for (int j = 0; j < nbSamples; j++) {
			ANN optimization = *this;
			//Update weights until the cost doesn't change anymore
			float cost = 1;
			while (1) {
				optimization.propagate(trainingSet->samples[j].input);
				optimization.backpropagate(trainingSet->samples[j].output, m_nbHLayers + 1);
				float new_cost = optimization.findCost(trainingSet->samples[j].output);
				if (new_cost == cost)
					break;
				cost = new_cost;
			}
			optimizations.push_back(optimization);
			progress++;
		}
		//Calculate the average of the weights and biases between all the optimizations
		updateNetwork(optimizations);
		optimizations.clear();
	}
	progressBarThread.join();
}

string ANN::probe(const vector<float> &inputs) {
	//Propagating input through the neural network
	//Returning the string of the possible output which is the closest to the real output
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
			m_neurons[0][i]->activation = 0.5;
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
	//Propagateing, but only starting from startingLayer
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
	//Finding the string of the possible output which is the closest to the output of the neural network
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
	//Dividing the weighted sum by the number of neurons in the previous layer and adding 1 to take into account the bias
	///The maximum value that a weighted sum can give is when all the weights, all the activations and the bias are equal to 1
	//Ex: s = w1*a1 + w2*a2 + b = 1*1 + 1*1 + 1 = 3
    return input / (m_layerSizes[layer - 1] + 1);
}

void ANN::updateWeights(const vector<float> &gradient, Neuron *neuron) {
	//Making small changes to the weights and the bias that are proportional to dC/dw or dC/db
	float updatedWeight, updatedBias;
	for (int i = 0; i < neuron->weights.size(); i++) {
		updatedWeight = neuron->weights[i] + (-gradient[i] * m_learningRate * 10000);
		//Making sure that the updated weight sits between 0 and 1
		if (updatedWeight < 0)
			neuron->weights[i] = 0;
		else if(updatedWeight > 1)
			neuron->weights[i] = 1;
		else
			neuron->weights[i] = updatedWeight;
	}
	updatedBias = neuron->bias + (-gradient.back() * m_learningRate * 1000);
	//Making sure that the updated bias sits between 0 and 1
	if (updatedBias < 0)
		neuron->bias = 0;
	else if (updatedBias > 1)
		neuron->bias = 1;
	else
		neuron->bias = updatedBias;
}

float ANN::findCost(const vector<float> &outputs) {
	//C = 0.5(a - t)^2 where a is the activation of an output neuron and t, what it's supposed to be
	//When there's multiple output neurons, the cost will be the sum of every output neuron's cost
	float ret = 0;
	vector<Neuron*> &outputLayer = m_neurons[m_nbHLayers + 1];
	for (int i = 0; i < outputLayer.size(); i++)
		ret += (0.5 * (outputLayer[i]->activation - outputs[i]) * (outputLayer[i]->activation - outputs[i]));
	return ret;
}

void ANN::updateNetwork(const vector<ANN> &optimizations) {
	//Cycling through all the neurons to update their weights and bias with their average between all the optimizations/configs
	unsigned int nbSamples = optimizations.size();
	for (int i = 1; i < m_nbHLayers; i++) { //Cycling through the layers
		for (int j = 0; j < m_layerSizes[i]; j++) { //Cycling through the neurons of a layer
			for (int k = 0; k < m_layerSizes[i - 1]; k++) { //Cycling through the weights of a neuron
				//Calculating the average for the weight
				float average = 0;
				for (int l = 0; l < nbSamples; l++)
					average += optimizations[l].m_neurons[i][j]->weights[k];
				m_neurons[i][j]->weights[k] = average / nbSamples;
			}
			//Calculating the average for the bias
			float average = 0;
			for (int l = 0; l < nbSamples; l++)
				average += optimizations[l].m_neurons[i][j]->bias;
			m_neurons[i][j]->bias = average / nbSamples;
		}
	}
}