/* 
 * Edjust NUMB_OF_EPOCHS for the iterations
 */

#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <algorithm>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define CHECK(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define LEARNING_RATE 0.25 
#define NUMB_OF_EPOCHS 100000

#define TD_X 4 // training data in x- dimension
#define TD_Y 2 // training data in y- dimension
#define TD_Z 2 // training data in z- dimension   

double TRAINING_DATA[TD_X][TD_Y][TD_Z] = {{{0,0},{0}}, {{0,1},{1}},
					          	 		 {{1,0},{1}}, {{1,1},{1}}};

void _printResult_(double result[]); 
void _printNetworkInfo_(); 
double applyActivationFunction(double weightedSum); 
void __setNeurons__(vector<float> &neurons, int size); 
void __forwardProp__(double input[], vector<float> &neurons, int nOfneurons, int size); 

__device__ double __applyActivationFunction__(double weightedSum) {
	// activation function is a sigmoid function (GPU function)
	return (1.0 / (1 + exp(-1.0 * weightedSum)));  
}

__device__ double derivative(double output) {
	// the derivative of the sigmoid function (GPU function)
	return output * (1.0 - output); 
}

__device__ void forwardPropagate(double input[], float * neurons){
	int size = 5; // Each neuron has 5 variables. 
	int nOfneurons = 5; 
	double weightedSum = 0.0; 
	for( int i = 0; i < nOfneurons; i++){
		switch (i) {
			case 0: case 1: // input layer
				neurons[(i * size) + 3] = input[i];  
				break;
			case 2: case 3: // hidden layer
				weightedSum = neurons[(i * size) + 0] + 
								  neurons[(i * size) + 1] * neurons[(0 * size) + 3] + 
		    		              neurons[(i * size) + 2] * neurons[(1 * size) + 3];
				neurons[(i * size) + 3] = __applyActivationFunction__(weightedSum); 
				break; 
			case 4: // output layer
				weightedSum = neurons[(i * size) + 0] + 
	    		                  neurons[(i * size) + 1] * neurons[(2 * size) + 3] + 
	    		                  neurons[(i * size) + 2] * neurons[(3 * size) + 3];
		    	neurons[(i * size) + 3] = __applyActivationFunction__(weightedSum); 
				break; 
		}
	}
}

__device__ void backPropagate(double targetResult, float *neurons, double learning_rate) {
	int size = 5; // Each neuron has 5 variables.  

	// calculating for output neurons
	neurons[(4 * size) + 4] = (targetResult - neurons[(4 * size) + 3]) * derivative(neurons[(4 * size) + 3]);
	neurons[(4 * size) + 0] = neurons[(4 * size) + 0] + learning_rate * neurons[(4 * size) + 4];
	neurons[(4 * size) + 1] = neurons[(4 * size) + 1] + learning_rate * neurons[(4 * size) + 4] * neurons[(2 * size) + 3]; 
	neurons[(4 * size) + 2] = neurons[(4 * size) + 2] + learning_rate * neurons[(4 * size) + 4] * neurons[(3 * size) + 3]; 
	
	// calculating for a neuron in hidden layer 
	neurons[(3 * size) + 4] = (neurons[(4 * size) + 2] * neurons[(4 * size) + 4]) * derivative(neurons[(3 * size) + 3]);
	neurons[(3 * size) + 0] = neurons[(3 * size) + 0] + learning_rate * neurons[(3 * size) + 4];
	neurons[(3 * size) + 1] = neurons[(3 * size) + 1] + learning_rate * neurons[(3 * size) + 4] * neurons[(0 * size) + 3];
	neurons[(3 * size) + 2] = neurons[(3 * size) + 2] + learning_rate * neurons[(3 * size) + 4] * neurons[(1 * size) + 3];

	// calculating for a neuron hidden layer 
	neurons[(2 * size) + 4] = (neurons[(4 * size) + 1] * neurons[(4 * size) + 4]) * derivative(neurons[(2 * size) + 3]);
	neurons[(2 * size) + 0] = neurons[(2 * size) + 0] + learning_rate * neurons[(2 * size) + 4];
	neurons[(2 * size) + 1] = neurons[(2 * size) + 1] + learning_rate * neurons[(2 * size) + 4] * neurons[(0 * size) + 3];
	neurons[(2 * size) + 2] = neurons[(2 * size) + 2] + learning_rate * neurons[(2 * size) + 4] * neurons[(1 * size) + 3];
}

__global__ void trainNeurons(float *Neuronset0, float *Neuronset1, float *Neuronset2, float *Neuronset3,int epochs, double learning_rate) {

	// This should be the same as host TRAINING_DATA
	double TRAINING_DATA[TD_X][TD_Y][TD_Z] = {{{0,0},{0}}, 
											  {{0,1},{1}},
					          	 		 	  {{1,0},{1}}, 
					          	 		 	  {{1,1},{1}}};

	// There should be 4 threads for four inputs and four outputs
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(idx == 0){ // thread for the input 0, 0 
		for(int i = 0; i < epochs; i++) {
			forwardPropagate(TRAINING_DATA[idx][0], Neuronset0); 	          				 	
			backPropagate(TRAINING_DATA[idx][1][0], Neuronset0, learning_rate); 
		}
	}
	if(idx == 1){ // thread for the input 0, 1 
		for(int i = 0; i < epochs; i++) {
			forwardPropagate(TRAINING_DATA[idx][0], Neuronset1); 	          				 	
			backPropagate(TRAINING_DATA[idx][1][0], Neuronset1, learning_rate); 
		}
	}
	if(idx == 2){ // thread for the input 1, 0 
		for(int i = 0; i < epochs; i++) {
			forwardPropagate(TRAINING_DATA[idx][0], Neuronset2); 	          				 	
			backPropagate(TRAINING_DATA[idx][1][0], Neuronset2, learning_rate); 
		}
	}
	if(idx == 3){ // thread for the input 1, 1 
		for(int i = 0; i < epochs; i++) {
			forwardPropagate(TRAINING_DATA[idx][0], Neuronset3); 	          				 	
			backPropagate(TRAINING_DATA[idx][1][0], Neuronset3, learning_rate); 
		}
	}
}

int main(void){

	// set up device
	int dev = 0; 
	cudaDeviceProp deviceProp; 
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev)); 

 	_printNetworkInfo_(); 

 	double result[] = {0, 0, 0, 0}; 
 	int N = 5; // number of neurons
 	int V = 5; // number of variables 
 	size_t nBytes = N * V * sizeof(float); 

 	// declare and initialize neurons as pointers
 	// malloc device global memory
	
 	vector<float> host_neurons0(N * V);
 	vector<float> host_neurons1(N * V);
 	vector<float> host_neurons2(N * V);
 	vector<float> host_neurons3(N * V);

	__setNeurons__(host_neurons0, N * V); // initialize neurons 

	for(int i = 0; i < TD_X; i++) {   // TD_X - Traning Data Dimension X 
		__forwardProp__(TRAINING_DATA[i][0], host_neurons0, N, V);
		result[i] = host_neurons0[23]; // get output of the output neuron. 
	}
	_printResult_(result); 

	float *dev_neuronset0; 
	float *dev_neuronset1; 
	float *dev_neuronset2; 
	float *dev_neuronset3; 

	cudaMalloc(&dev_neuronset0, nBytes);
	cudaMalloc(&dev_neuronset1, nBytes);
	cudaMalloc(&dev_neuronset2, nBytes);
	cudaMalloc(&dev_neuronset3, nBytes);

	// train network from CPU. 
	float GPUtime; 
	cudaEvent_t start, stop; 

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0); 

	printf("Training..\n"); 
	dim3 block(1);
	dim3 grid(4); 

	// Copy the array to device
	cudaMemcpy(dev_neuronset0, host_neurons0.data(), nBytes, cudaMemcpyHostToDevice);  
	cudaMemcpy(dev_neuronset1, host_neurons0.data(), nBytes, cudaMemcpyHostToDevice);  
	cudaMemcpy(dev_neuronset2, host_neurons0.data(), nBytes, cudaMemcpyHostToDevice);  
	cudaMemcpy(dev_neuronset3, host_neurons0.data(), nBytes, cudaMemcpyHostToDevice);  

	// Kernel Launch 
	trainNeurons <<< grid, block >>> (dev_neuronset0, dev_neuronset1, dev_neuronset2, dev_neuronset3, NUMB_OF_EPOCHS, LEARNING_RATE); 
	// Sync point
	cudaDeviceSynchronize(); 

	// Copy back the network
	cudaMemcpy(host_neurons0.data(), dev_neuronset0, nBytes, cudaMemcpyDeviceToHost); 
	cudaMemcpy(host_neurons1.data(), dev_neuronset1, nBytes, cudaMemcpyDeviceToHost); 
	cudaMemcpy(host_neurons2.data(), dev_neuronset2, nBytes, cudaMemcpyDeviceToHost); 
	cudaMemcpy(host_neurons3.data(), dev_neuronset3, nBytes, cudaMemcpyDeviceToHost); 

	__forwardProp__(TRAINING_DATA[0][0], host_neurons0, N, V);
	result[0] = host_neurons0[23]; // get output of the output neuron. 

	__forwardProp__(TRAINING_DATA[1][0], host_neurons1, N, V);
	result[1] = host_neurons1[23]; // get output of the output neuron. 

	__forwardProp__(TRAINING_DATA[2][0], host_neurons2, N, V);
	result[2] = host_neurons2[23]; // get output of the output neuron. 

	__forwardProp__(TRAINING_DATA[3][0], host_neurons3, N, V);
	result[3] = host_neurons3[23]; // get output of the output neuron. 

	printf("[done training]\n"); 
	_printResult_(result); 

	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop); 
	cudaEventElapsedTime(&GPUtime, start, stop); 
	

	printf("Compute time on GPU: %3.6f ms \n", GPUtime); 

	cudaFree(dev_neuronset0); 
	cudaFree(dev_neuronset1); 
	cudaFree(dev_neuronset2); 
	cudaFree(dev_neuronset3); 
	return(1);
}

void __setNeurons__(vector<float> &neurons, int size){

	srand((long)time(NULL)); /* initialize rand() */
	for (int i = 0; i < size; i ++){
		if( i%5 == 0 || i%5 == 1 || i%5 == 2)
			neurons[i] = 0.5 - (rand()/(double)RAND_MAX); 
	}
}

void __forwardProp__(double input[], vector<float> &neurons, int nOfneurons, int size) {
	double weightedSum = 0; 
	for( int i = 0; i < nOfneurons; i++){
		switch (i) {
			case 0: case 1: // input layer
				neurons[(i * size) + 3] = input[i];  
				break;
			case 2: case 3: // hidden layer
				weightedSum = neurons[(i * size) + 0] + 
								  neurons[(i * size) + 1] * neurons[(0 * size) + 3] + 
		    		              neurons[(i * size) + 2] * neurons[(1 * size) + 3];
				neurons[(i * size) + 3] = applyActivationFunction(weightedSum); 
				break; 
			case 4: // output layer
				weightedSum = neurons[(i * size) + 0] + 
	    		                  neurons[(i * size) + 1] * neurons[(2 * size) + 3] + 
	    		                  neurons[(i * size) + 2] * neurons[(3 * size) + 3];
		    	neurons[(i * size) + 3] = applyActivationFunction(weightedSum); 
				break; 
		}
	}
}

double applyActivationFunction(double weightedSum) {
	// activation function is a sigmoid function
	return (1.0 / (1 + exp(-1.0 * weightedSum)));  
}

void _printNetworkInfo_(){

	// the number of inputs, hidden layers and output layers are set. 
	// the number of iterations and learning rate can be vary. 
	printf("Number of inputs: %d\n", 2); 
	printf("Number of hidden layers: %d\n", 2); 
	printf("Number of output: %d\n", 1);
	printf("Number of iterations: %d\n", NUMB_OF_EPOCHS);
	printf("Learning Rate: %.2f\n", LEARNING_RATE);

}

void _printResult_(double result[]) {
	printf("    Input 1    |    Input 2    | Target Result |  Result    \n");
	printf("-------------------------------------------------------------\n");
	for(int i = 0; i < 4; i++ ) {
		for(int j = 0; j < 2; j++) {
			printf("    %.5f    |", TRAINING_DATA[i][0][j]); 
		}
		printf("    %.5f    |   %.5f   \n", TRAINING_DATA[i][1][0], result[i]);
	}
}
