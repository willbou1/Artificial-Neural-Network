#include "common.h"
#include "ANN.h"
#include "File.h"

ANN::Neuron::Neuron(unsigned int prevLayerSize) : activation(0) {
    //Initializing everything randomly if this isn’t an input neuron
    if(prevLayerSize) {
        for(int i = 0; i < prevLayerSize; i++)
            weights.push_back(randToOne());
        bias = randToOne() * 2 - 1;
    }
}

//Public:
ANN::ANN(unsigned int nbHLayers, unsigned int nbInputNeurons, unsigned int nbHiddenNeurons, unsigned int nbOutputNeurons, float learningRate) : nbHLayers(nbHLayers), learningRate(learningRate) {
	initLayerSizes(nbInputNeurons, nbHiddenNeurons, nbOutputNeurons);
    initNeurons();
}

ANN::ANN(const TrainingSet *trainingSet, unsigned int nbHLayers, unsigned  nbHiddenNeurons = 0, float learningRate) : nbHLayers(nbHLayers), learningRate(learningRate) {
    if(!nbHiddenNeurons)
        nbHiddenNeurons = (int)(1.5 * (float)trainingSet->maxInputSize);
	initLayerSizes(trainingSet->maxInputSize, nbHiddenNeurons, 1);
	initNeurons();
    train(trainingSet, 1000);
}

ANN::ANN(const vector<int> &layerSizes, float learningRate) : layerSizes(layerSizes), learningRate(learningRate) {
    nbHLayers = layerSizes.size() - 2;
	initNeurons();
}

ANN::~ANN() {
    for(int i = 0; i < nbHLayers + 2; i++) {
        for(int j = 0; j < layerSizes[i]; j++)
            delete neurons[i][j];
    }
}

void ANN::initLayerSizes(unsigned int nbInputNeurons, unsigned int nbHiddenNeurons, unsigned int nbOutputNeurons) {
	//Initializing a vector with all the layers' size
	layerSizes.push_back(nbInputNeurons);
	for (int i = 0; i < nbHLayers; i++)
		layerSizes.push_back(nbHiddenNeurons);
	layerSizes.push_back(nbOutputNeurons);
}

void ANN::initNeurons() {
    //Creating the neurons
    vector<Neuron*> temp;
    for(int i = 0; i < layerSizes[0]; i++)
        temp.push_back(new Neuron(0));
    neurons.push_back(temp);
    temp.clear();
    for(int i = 1; i < nbHLayers + 2; i++) {
        for(int j = 0; j < layerSizes[i]; j++)
            temp.push_back(new Neuron(layerSizes[i - 1]));
        neurons.push_back(temp);
        temp.clear();
    }
}

void ANN::train(const TrainingSet *trainingSet, unsigned int nbIterations) {
    outputMap = trainingSet->outputMap;
	unsigned int progress = 0, nbSamples = trainingSet->samples.size();
	cout << "Training the neural network" << endl;
	thread progressBarThread(&progressBar, &progress, nbSamples * nbIterations);
    for(int i = 0; i < nbSamples * nbIterations; i++) {
        propagate(trainingSet->samples[i % nbSamples].input);
		backpropagate(trainingSet->samples[i % nbSamples].output, nbHLayers + 1);
		progress++;
	}
	progressBarThread.join();
	cout << "Done!" << endl;
}

string ANN::probe(const vector<float> &inputs) {
	if (!outputMap.size())
		return string();
	propagate(inputs);
	return mapOutput(neurons[nbHLayers + 1][0]->activation);
}

//Private:
void ANN::propagate(const vector<float> &inputs) {
	//Setting the first layer's neurons' activations to the value of the inputs
	for (int i = 0; i < layerSizes[0]; i++) {
		if (i < inputs.size())
			neurons[0][i]->activation = inputs[i];
		else
			neurons[0][i]->activation = 0;
	}
	//Propagating the input layer through the other layers
	for (int i = 1; i < nbHLayers + 2; i++) {
		for (int j = 0; j < layerSizes[i]; j++) {
			Neuron *curr = neurons[i][j];
			float activation = curr->bias;
			for (int k = 0; k < layerSizes[i - 1]; k++)
				activation += neurons[i - 1][k]->activation * curr->weights[k];
			curr->activation = activate(activation, i);
		}
	}
}

void ANN::propagate(unsigned int startingLayer) {
	for (int i = startingLayer; i < nbHLayers + 2; i++) {
		for (int j = 0; j < layerSizes[i]; j++) {
			Neuron *curr = neurons[i][j];
			float activation = curr->bias;
			for (int k = 0; k < layerSizes[i - 1]; k++)
				activation += neurons[i - 1][k]->activation * curr->weights[k];
			curr->activation = activate(activation, i);
		}
	}
}

void ANN::backpropagate(const vector<float> &outputs, unsigned int layer) {
	//Initializing a multidimensional vector of gradients
	vector<vector<vector<float>>> gradients(nbHLayers + 2);
	for (int i = 0; i < nbHLayers + 2; i++)
		gradients[i] = vector<vector<float>>(layerSizes[i]);
	float dBase, derivative;
	//Calculating the gradients of the output layer
	int nbOutputNeurons = layerSizes[nbHLayers + 1];
	for (int i = 0; i < nbOutputNeurons; i++) {
		//dBase (dC/dz) = dz(cost(act)) = da(cost) * dz(act)
		dBase = (outputs[i] - neurons[nbHLayers + 1][i]->activation);
		dBase *= (1 / (float)(layerSizes[nbHLayers] + 1));
		for (int j = 0; j < layerSizes[nbHLayers]; j++) {
			//derivative (dC/dw) = dBase * dw(wSums)
			derivative = dBase * neurons[nbHLayers][j]->activation;
			gradients[nbHLayers + 1][i].push_back(derivative);
		}
		//derivative (dC/db) = dBase * db(wSums)
		gradients[nbHLayers + 1][i].push_back(dBase);
	}
	//Updating weights of input neurons
	if (layer == nbHLayers + 1) {
		for (int i = 0; i < nbOutputNeurons; i++)
			updateWeights(gradients[nbHLayers + 1][i], neurons[nbHLayers + 1][i]);
		//Repropagating
		propagate(nbHLayers + 1);
		//Continue where we left
		backpropagate(outputs, layer - 1);
		return;
	}
	//Calculating the gradients of the hidden layers and updating the weights
	for (int i = nbHLayers; i > 0; i--) {
		for (int j = 0; j < layerSizes[i]; j++) {
			//dBase (dC/da) where a is the activation of this neuron
			dBase = 0;
			for (int k = 0; k < layerSizes[i + 1]; k++)
				dBase += gradients[i + 1][k][j] * neurons[i + 1][k]->weights[j];
			//dBase (dC/dz) = dBase * dz(wSums)
			dBase *= (1 / (float)(layerSizes[i - 1] + 1));
			for (int k = 0; k < layerSizes[i - 1]; k++) {
				//derivative (dC/dw) = dBase * dw(wSums)
				derivative = dBase * neurons[i - 1][k]->activation;
				gradients[i][j].push_back(derivative);
			}
			gradients[i][j].push_back(dBase);
		}
		//Updating weights
		if (layer == i) {
			for (int j = 0; j < layerSizes[i]; j++)
				updateWeights(gradients[i][j], neurons[i][j]);
			//Repropagate
			propagate(i);
			//Continue where we left
			backpropagate(outputs, layer - 1);
			return;
		}
	}
}

string ANN::mapOutput(float output) {
    if(outputMap.size() == 0)
        return "";
    float smallestDistance = 1;
    string ret;
    for(map<float, string>::iterator it = outputMap.begin(); it != outputMap.end(); it++) {
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
    if(input <= 0)
        return 0;
    return input / (layerSizes[layer - 1] + 1);
}

void ANN::updateWeights(const vector<float> &gradient, Neuron *neuron) {
	for (int i = 0; i < neuron->weights.size(); i++)
		neuron->weights[i] += -gradient[i] * learningRate * 0.01;
	neuron->bias += -gradient.back() * learningRate * 0.0001;
}

float ANN::findCost(const vector<float> &outputs) {
	float ret = 0;
	vector<Neuron*> &outputLayer = neurons[nbHLayers + 1];
	for (int i = 0; i < outputLayer.size(); i++)
		ret += (0.5 * (outputs[i] - outputLayer[i]->activation) * (outputs[i] - outputLayer[i]->activation));
	return ret;
}