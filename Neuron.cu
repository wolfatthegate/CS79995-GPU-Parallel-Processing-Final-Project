#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

enum neuralNetworkLayerType { inputLayer = 0, hiddenLayer = 1, outputLayer = 2 }; 

double applyActivationFunction(double weightedSum) {
	// activation function is a sigmoid function
	return (1.0 / (1 + exp(-1.0 * weightedSum)));  
}

double derivative(double output) {
	return output * (1.0 - output); 
}

struct neuron {
	
	double threshold = 0; 
	double weight[2] = {0, 0}; 
	double output = 0; 
	double error = 0; 
	neuralNetworkLayerType layerType; 
};

void forwardProp(double input[], struct neuron *neurons) {
	double weightedSum = 0; 
	for( int i = 0; i < (int) sizeof (neurons); i++){
		switch (neurons[i].layerType) {
			case 0: // input layer
				neurons[i].output = input[i];  
				break;
			case 1: // hidden layer
				weightedSum = neurons[i].threshold + 
								  neurons[i].weight[0] * neurons[0].output + 
		    		              neurons[i].weight[1] * neurons[1].output;
				neurons[i].output = applyActivationFunction(weightedSum); 
				break; 
			case 2: // output layer
				weightedSum = neurons[i].threshold + 
	    		                  neurons[i].weight[0] * neurons[2].output + 
	    		                  neurons[i].weight[1] * neurons[3].output;
		    	neurons[i].output = applyActivationFunction(weightedSum); 
				break; 
		}
	}
}

void backpropError(double targetResult, struct neuron *neurons){
	// calculating for output neurons
	neurons[4].error = (targetResult - neurons[4].output) * derivative(neurons[4].output);
	neurons[4].threshold = neurons[4].threshold + LEARNING_RATE * neurons[4].error;
	neurons[4].weight[0] = neurons[4].weight[0] + LEARNING_RATE * neurons[4].error * neurons[2].output; 
	neurons[4].weight[1] = neurons[4].weight[1] + LEARNING_RATE * neurons[4].error * neurons[3].output; 
	
	// calculating for hidden layer 1 
	neurons[3].error = (neurons[4].weight[1] * neurons[4].error) * derivative(neurons[3].output);
	neurons[3].threshold = neurons[3].threshold + LEARNING_RATE * neurons[3].error;
	neurons[3].weight[0] = neurons[3].weight[0] + LEARNING_RATE * neurons[3].error * neurons[0].output;
	neurons[3].weight[1] = neurons[3].weight[1] + LEARNING_RATE * neurons[3].error * neurons[1].output;

	// calculating for hidden layer 2 
	neurons[2].error = (neurons[4].weight[0] * neurons[4].error) * derivative(neurons[2].output);
	neurons[2].threshold = neurons[2].threshold + LEARNING_RATE * neurons[2].error;
	neurons[2].weight[0] = neurons[2].weight[0] + LEARNING_RATE * neurons[2].error * neurons[0].output;
	neurons[2].weight[1] = neurons[2].weight[1] + LEARNING_RATE * neurons[2].error * neurons[1].output;

}

void setNeurons(struct neuron *neurons){

	srand((long)time(NULL)); /* initialize rand() */
	for (int i = 0; i < 2; i ++){
		neurons[i].threshold = 0.5 - (rand()/(double)RAND_MAX); 
		neurons[i].weight[0] =  0.5 - (rand()/(double)RAND_MAX); 
		neurons[i].weight[1] =  0.5 - (rand()/(double)RAND_MAX); 
		neurons[i].layerType = inputLayer; 
	}

	for (int i = 2; i < 4; i ++){
		neurons[i].threshold = 0.5 - (rand()/(double)RAND_MAX); 
		neurons[i].weight[0] =  0.5 - (rand()/(double)RAND_MAX); 
		neurons[i].weight[1] =  0.5 - (rand()/(double)RAND_MAX); 
		neurons[i].layerType = hiddenLayer; 
	}

	neurons[4].threshold = 0.5 - (rand()/(double)RAND_MAX); 
	neurons[4].weight[0] =  0.5 - (rand()/(double)RAND_MAX); 
	neurons[4].weight[1] =  0.5 - (rand()/(double)RAND_MAX); 
	neurons[4].layerType = outputLayer; 
}

void printTrainingData(struct neuron *neurons){
	
	printf("[(I: %.2f), (I: %.2f), ", neurons[0].output, neurons[1].output); 
	printf("(H: %.2f, %.2f, %.2f, %.5f), ", neurons[2].weight[0], neurons[2].weight[1], neurons[2].threshold, neurons[2].output);
	printf("(H: %.2f, %.2f, %.2f, %.5f), ", neurons[3].weight[0], neurons[3].weight[1], neurons[3].threshold, neurons[3].output);
	printf("(O: %.2f, %.2f, %.2f, %.5f)]\n ", neurons[4].weight[0], neurons[4].weight[1], neurons[4].threshold, neurons[4].output);
}

void printResult(double result[]) {
	printf("    Input 1    |    Input 2    | Target Result |  Result    \n");
	printf("-------------------------------------------------------------\n");
	for(int i = 0; i < 4; i++ ) {
		for(int j = 0; j < 2; j++) {
			printf("    %.5f    |", TRAINING_DATA[i][0][j]); 
		}
		printf("    %.5f    |   %.5f   \n", TRAINING_DATA[i][1][0], result[i]);
	}
}

void trainOnCPU(struct neuron *neurons){
	
	double result[] = {0, 0, 0, 0}; 

	for(int i = 0; i < TD_X; i++) {   // TD_X - Traning Data Dimension X 
		forwardProp(TRAINING_DATA[i][0], neurons);
		result[i] = neurons[4].output; // get output
	}
	printResult(result); 

	// training 100 * 100 = 10,000 trainings 
	printf("Training..\n");

	for(int i = 0; i < NUMB_OF_EPOCHS; i++) {   
		if(i%100 == 0) {
				// printf("[epoch %d ]\n", i);
		}
		for(int j = 0; j < TD_X; j++) {  // TD_X - Traning Data Dimension X 
			forwardProp(TRAINING_DATA[j][0], neurons);
			backpropError(TRAINING_DATA[j][1][0], neurons);		
			if(i%100 == 0) {
					// printTrainingData(neurons); 
			}
		}
	}
	
	printf("[done training]\n");

	// forward propagation after the training
	for(int i = 0; i < TD_X; i++) {
		forwardProp(TRAINING_DATA[i][0], neurons);
		result[i] = neurons[4].output; // get output
	}
	printResult(result); 
}

void trainOnCPU2(struct neuron *neurons, struct neuron *neurons2, struct neuron *neurons3, struct neuron *neurons4){
	
	double result[] = {0, 0, 0, 0}; 

	for(int i = 0; i < TD_X; i++) {   // TD_X - Traning Data Dimension X 
		forwardProp(TRAINING_DATA[i][0], neurons);
		result[i] = neurons[4].output; // get output
	}
	printResult(result); 

	// training 100 * 100 = 10,000 trainings 
	printf("Training..\n");

	for(int i = 0; i < NUMB_OF_EPOCHS; i++) {   			
		forwardProp(TRAINING_DATA[0][0], neurons);
		backpropError(TRAINING_DATA[0][1][0], neurons);		
	}
	forwardProp(TRAINING_DATA[0][0], neurons);	
	result[0] = neurons[4].output;
	
	for(int i = 0; i < NUMB_OF_EPOCHS; i++) {   
		forwardProp(TRAINING_DATA[1][0], neurons2);
		backpropError(TRAINING_DATA[1][1][0], neurons2);	
	}
	forwardProp(TRAINING_DATA[1][0], neurons2);	
	result[1] = neurons2[4].output;

	for(int i = 0; i < NUMB_OF_EPOCHS; i++) {   		
		forwardProp(TRAINING_DATA[2][0], neurons3);
		backpropError(TRAINING_DATA[2][1][0], neurons3);			
	}
	forwardProp(TRAINING_DATA[2][0], neurons3);	
	result[2] = neurons3[4].output;

	for(int i = 0; i < NUMB_OF_EPOCHS; i++) {   		
		forwardProp(TRAINING_DATA[3][0], neurons4);
		backpropError(TRAINING_DATA[3][1][0], neurons4);		
	}
	forwardProp(TRAINING_DATA[3][0], neurons4);	
	result[3] = neurons4[4].output;


	printResult(result); 
}

void printNetworkInfo(){

	// the number of inputs, hidden layers and output layers are set. 
	// the number of iterations and learning rate can be vary. 
	printf("Number of inputs: %d\n", 2); 
	printf("Number of hidden layers: %d\n", 2); 
	printf("Number of output: %d\n", 1);
	printf("Number of iterations: %d\n", NUMB_OF_EPOCHS);
	printf("Learning Rate: %.2f\n", LEARNING_RATE);

}
