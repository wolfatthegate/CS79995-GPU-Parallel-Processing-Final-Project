/* 
 * 
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

double TRAINING_DATA[TD_X][TD_Y][TD_Z] = {{{0,0},{0}},
					          	 {{0,1},{1}},
					          	 {{1,0},{1}},
					          	 {{1,1},{1}}}; 

void trainOnCPU(struct neuron *neurons);
void printNetworkInfo();

#include "Neuron.cu"

int main(void){

	// set up device
	int dev = 0; 
	cudaDeviceProp deviceProp; 
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev)); 
 
 	printNetworkInfo(); 

 	// declare and initialize neurons 
	struct neuron neurons[5]; 
	setNeurons(neurons);

	// train network from CPU. 
	float CPUtime; 
	cudaEvent_t start, stop; 

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0); 

	trainOnCPU(neurons); 

	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop); 
	cudaEventElapsedTime(&CPUtime, start, stop); 

	printf("Compute time on CPU: %3.6f ms\n", CPUtime); 

	return(1);
}

