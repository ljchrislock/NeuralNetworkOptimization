#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <fstream>
#include <ctime>
#include <functional>

// Activation Functions as described exactly in the assignment. 
// Bipolar Sigmoid Activation Function
double bipolarSigmoid(double x)
{
    return 2.0 / (1.0 + std::exp(-x)) - 1.0;
}
// Unipolar Sigmoid Activation Function
double unipolarSigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}
// ReLU Activation Function
double relu(double x)
{

    return std::max(0.0, x);
}

class Particle
{
public:
    std::vector<double> position;     // Current position (weights and biases)
    std::vector<double> velocity;     // velocity
    std::vector<double> bestPosition; // Best position found by this particle
    double bestFitness;               // Best fitness value for this particle

    // Constructor to initialize a particle with random position and velocity
    Particle(int dimensions)
    {
        position.resize(dimensions); //dont forget to resize (note for myself because I forgot in the last lab :)
        velocity.resize(dimensions);
        bestPosition.resize(dimensions);
        bestFitness = std::numeric_limits<double>::infinity(); // Start with worst fitness

        for (int i = 0; i < dimensions; i++)
        {
            position[i] = randomValue();
            velocity[i] = randomValue() * 0.1; // Small initial velocity
        }
    }

private:
    // Helper function to generate a random value in the range [-1, 1] using srand like below. 
    static double randomValue()
    {
        return ((double)std::rand() / RAND_MAX) * 2 - 1;
    }
};

// Neural Network Class
class NeuralNetwork
{
public:
    std::vector<int> layerSizes;                            // Number of neurons in each layer (5 in each layer, there are two hidden!)
    std::vector<std::vector<std::vector<double>>> weights;  // Weights(the[layer])(the[neuron])(and the [prevNeuron]) all help make up this vector
    std::vector<std::vector<double>> biases;                // Biases[layer][neuron]
    std::vector<std::function<double(double)>> activations; // Activation functions per layer

    NeuralNetwork(const std::vector<int> &sizes, const std::vector<std::function<double(double)>> &activationFuncs)
    {
        layerSizes = sizes;
        activations = activationFuncs;

        for (size_t i = 1; i < sizes.size(); i++)
        {
            weights.push_back(std::vector<std::vector<double>>(sizes[i], std::vector<double>(sizes[i - 1])));
            biases.push_back(std::vector<double>(sizes[i]));

            for (size_t neuron = 0; neuron < sizes[i]; neuron++)
            {
                biases[i - 1][neuron] = randomValue(); // Initialize biases
                for (size_t prevNeuron = 0; prevNeuron < sizes[i - 1]; prevNeuron++)
                {
                    weights[i - 1][neuron][prevNeuron] = randomValue(); // Initialize weights
                }
            }
        }
    }

    double forward(const std::vector<double> &input)
    {
        std::vector<double> currentActivations = input;

        for (size_t layer = 1; layer < layerSizes.size(); layer++)
        {
            std::vector<double> nextActivations(layerSizes[layer]);

            for (size_t neuron = 0; neuron < layerSizes[layer]; neuron++)
            {
                double weightedSum = biases[layer - 1][neuron];
                for (size_t prevNeuron = 0; prevNeuron < layerSizes[layer - 1]; prevNeuron++)
                {
                    weightedSum += currentActivations[prevNeuron] * weights[layer - 1][neuron][prevNeuron];
                }
                nextActivations[neuron] = activations[layer - 1](weightedSum);
            }

            currentActivations = nextActivations;
        }

        return currentActivations[0]; // Return output of the last layer
    }

    void setParameters(const std::vector<double> &parameters)
    {
        size_t index = 0;
        for (size_t layer = 1; layer < layerSizes.size(); layer++)
        {
            for (size_t neuron = 0; neuron < layerSizes[layer]; neuron++)
            {
                for (size_t prevNeuron = 0; prevNeuron < layerSizes[layer - 1]; prevNeuron++)
                {
                    weights[layer - 1][neuron][prevNeuron] = parameters[index++];
                }
                biases[layer - 1][neuron] = parameters[index++];
            }
        }
    }

private:
    double randomValue() const
    {
        return ((double)std::rand() / RAND_MAX) * 2 - 1; // Random value in range [-1, 1]
    }
};


class ParticleSwarmOptimizer
{
public:
    ParticleSwarmOptimizer(int swarmSize, int dimensions, NeuralNetwork &nn)
        : swarmSize(swarmSize), dimensions(dimensions), neuralNetwork(nn)
    {
        globalBestFitness = std::numeric_limits<double>::infinity();

        for (int i = 0; i < swarmSize; i++)
        {
            swarm.emplace_back(dimensions); //vector item, like push_back, but rather inserting at the end while constructing in place. 
        }
    }

    std::vector<double> optimize(const std::vector<double> &inputs, const std::vector<double> &targets, int iterations)
    {
        for (int iteration = 0; iteration < iterations; iteration++)
        {
            for (Particle &particle : swarm)
            {
                double fitness = evaluateFitness(inputs, targets, particle.position);

                // Update personal best
                if (fitness < particle.bestFitness)
                {
                    particle.bestFitness = fitness;
                    particle.bestPosition = particle.position;
                }

                // Update global best
                if (fitness < globalBestFitness)
                {
                    globalBestFitness = fitness;
                    globalBestPosition = particle.position;
                }
            }

            // Update particle velocities and positions
            for (Particle &particle : swarm)
            {
                for (int d = 0; d < dimensions; d++)
                {
                    double r1 = randomValue();
                    double r2 = randomValue();
                    particle.velocity[d] = inertiaWeight * particle.velocity[d] +
                                           cognitiveWeight * r1 * (particle.bestPosition[d] - particle.position[d]) +
                                           socialWeight * r2 * (globalBestPosition[d] - particle.position[d]);

                    particle.position[d] += particle.velocity[d];
                }
            }
        }

        return globalBestPosition;
    }

    double getGlobalBestFitness() const
    {
        return globalBestFitness;
    }

private:
    int swarmSize;
    int dimensions;
    NeuralNetwork &neuralNetwork;
    std::vector<Particle> swarm; // vector of type particle for our swarm.
    std::vector<double> globalBestPosition;
    double globalBestFitness;

    const double inertiaWeight = 0.7;
    const double cognitiveWeight = 1.4;
    const double socialWeight = 1.4;

    static double randomValue()
    {
        return ((double)std::rand() / RAND_MAX); // Random value in range [0, 1]
    }

    double evaluateFitness(const std::vector<double> &inputs, const std::vector<double> &targets, const std::vector<double> &params)
    {
        neuralNetwork.setParameters(params);
        double mse = 0.0;

        for (size_t i = 0; i < inputs.size(); i++)
        {
            double predicted = neuralNetwork.forward({inputs[i]});
            mse += std::pow(predicted - targets[i], 2);
        }

        return mse / inputs.size();
    }
};

int main()
{
    std::srand(static_cast<unsigned int>(std::time(0))); // random number generator.

    // Generate training data for sine function
    std::vector<double> inputs, targets;
    for (double x = -3 * M_PI; x <= 3 * M_PI; x += 0.05)
    {
        inputs.push_back(x); //pushing that data into our vector for training.
        targets.push_back(std::sin(x)); //as well as pushing that into our target vector. 
    }

    // Define the neural network architecture
    //Below is the orginal architecture, and the one below that is the one I have been using to optimize the network. 
    //std::vector<int> architecture = {1, 5, 5, 1}; //1 input, the second layer has 5 neurons, the third layer has 5 neurons, and the output layer has 1 neuron. There are two hidden layers being laid out here.
    std::vector<int> architecture = {1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 1};

    // Choose activation functions for each layer
    std::vector<std::function<double(double)>> activations = {
        bipolarSigmoid, // First hidden layer
        bipolarSigmoid, // Second hidden layer
        bipolarSigmoid, // Third hidden layer
        bipolarSigmoid, // Fourth hidden layer
        bipolarSigmoid, // Fifth hidden layer
        bipolarSigmoid, // Sixth hidden layer
        bipolarSigmoid, // Seventh hidden layer
        bipolarSigmoid, // Eighth hidden layer
        bipolarSigmoid, // Ninth hidden layer
        bipolarSigmoid, // Tenth hidden layer
        bipolarSigmoid  // Output layer
        
    };

    NeuralNetwork my_network(architecture, activations); //my instance of the NeuralNetwork clas, passing it our architecture to follow as well as its activation functions.

    // Initialize Particle Swarm Optimizer
    int totalParameters = 0;
    for (size_t i = 1; i < architecture.size(); i++)
    {   //total parameters is the total number of weights and biases in the network.
        totalParameters += architecture[i] * architecture[i - 1]; // Weights
        totalParameters += architecture[i];                       // Biases
    }

    ParticleSwarmOptimizer pso(30, totalParameters, my_network); //instance of the particle swarm optimizer, pass it the number of particles, number of parameters, and instance of my neural network

    // Optimize network parameters
    std::vector<double> bestParameters = pso.optimize(inputs, targets, 5000); //change the last number for the iterations!
    my_network.setParameters(bestParameters);

    //data writting and outputting basic code. 
    std::ofstream outputFile("Assignment_4_output.csv");
    outputFile << "Domain,Predicted,Actual\n";
    for (size_t i = 0; i < inputs.size(); i++)
    {
        double predicted = my_network.forward({inputs[i]});
        outputFile << inputs[i] << "," << predicted << "," << targets[i] << "\n";
    }
    outputFile.close();

    std::cout << "Training complete. Results saved to 'Assignment_4_output.csv'.\n";

    return 0;
}
