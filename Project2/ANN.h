#ifndef ANN_H
#define ANN_H

using namespace std;

class ANN {
    struct Neuron {
        Neuron(unsigned int prevLayerSize);
        float activation;
        vector<float> weights;
        float bias;
    };
    
public:
    ANN(unsigned int nbHLayers, unsigned int nbInputNeurons, unsigned int nbHiddenNeurons, unsigned int nbOutputNeurons, float learningRate = 0.5);
    ANN(const TrainingSet *trainingSet, unsigned int nbHLayers, unsigned int nbHiddenNeurons, float learningRate = 0.5);
    ANN(const vector<int> &layerSizes, float learningRate = 0.5);
    ~ANN();
	void initLayerSizes(unsigned int nbInputNeurons, unsigned int nbHiddenNeurons, unsigned int nbOutputNeurons);
    void initNeurons();
	void train(const TrainingSet *trainingSet, unsigned int nbIterations = 4);
    string probe(const vector<float> &input);
    
private:
    string mapOutput(float output);
    void propagate(const vector<float> &inputs);
	void propagate(unsigned int startingLayer);
    void backpropagate(const vector<float> &outputs, unsigned int layer);
    float activate(float input, int layer);
	void updateWeights(const vector<float> &gradient, Neuron *neuron);
	float findCost(const vector<float> &outputs);
    vector<vector<Neuron*> > m_neurons;
    unsigned int m_nbHLayers;
	float m_learningRate;
    vector<int> m_layerSizes;
    map<float, string> m_outputMap;
    friend class File;
};

#endif