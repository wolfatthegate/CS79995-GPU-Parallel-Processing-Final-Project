/*
 * backProgGPU_optimized.cu
 *
 * Optimized version of backProgGPU.cu.
 *
 * Same learning problem (logical OR), same 2-2-1 network, same 25-float
 * network layout, same parallelization strategy (one CUDA thread trains one
 * copy of the network on one training sample) and the same weight-update
 * order as the original, so the two programs can be compared directly.
 *
 * What changed and why (see README.md, section 6):
 *   1. Register-resident training. Each thread loads its network once, runs
 *      every epoch in registers and writes the result back once. The original
 *      reads and writes global memory roughly 60 times per epoch.
 *   2. Single precision throughout, with the __expf intrinsic in the sigmoid.
 *      The original promotes every float to double and calls double exp().
 *   3. One straight-line fused forward+backward step. No loop over neurons,
 *      no switch, no index arithmetic inside the epoch loop.
 *   4. Training data lives in __constant__ memory and is copied from the one
 *      host array, so it is defined in a single place.
 *   5. One contiguous device buffer and one copy in each direction, instead of
 *      four allocations and eight copies.
 *   6. One block of NUMB_OF_NETWORKS threads (one warp) instead of four
 *      single-thread blocks. The kernel uses a grid-stride loop, so it works
 *      unchanged for any number of networks.
 *   7. Every CUDA call is checked, the kernel is timed separately from the
 *      memory copies, and main returns 0 on success.
 *
 * Build:  nvcc -O3 backProgGPU_optimized.cu -o bpgpu_opt.out
 * Run:    ./bpgpu_opt.out [seed]
 *         Without a seed the generator is seeded with the current time, as in
 *         the original. Passing a seed makes runs repeatable.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err__));                               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define LEARNING_RATE  0.25f
#define NUMB_OF_EPOCHS 100000

#define TD_X 4 // training data in x- dimension (number of samples)
#define TD_Y 2 // training data in y- dimension (inputs / target)
#define TD_Z 2 // training data in z- dimension (values per row)

#define NUMB_OF_NEURONS  5
#define VARS_PER_NEURON  5                                   // threshold, weight[0], weight[1], output, error
#define NETWORK_SIZE     (NUMB_OF_NEURONS * VARS_PER_NEURON) // 25 floats per network
#define NUMB_OF_NETWORKS TD_X                                // one network per training sample

// Field offsets inside one neuron. Neuron n, field f is at index IDX(n, f).
// This is the same layout as backProgGPU.cu.
#define THRESHOLD 0
#define WEIGHT0   1
#define WEIGHT1   2
#define OUTPUT    3
#define ERROR     4
#define IDX(n, f) ((n) * VARS_PER_NEURON + (f))

// The only definition of the training data. It is copied to device constant
// memory at run time, so there is no second copy to keep in sync.
float TRAINING_DATA[TD_X][TD_Y][TD_Z] = {{{0, 0}, {0}},
                                         {{0, 1}, {1}},
                                         {{1, 0}, {1}},
                                         {{1, 1}, {1}}};

__constant__ float dev_trainingData[TD_X][TD_Y][TD_Z];

/* ------------------------------------------------------------------------ */
/* Math shared by host and device                                           */
/* ------------------------------------------------------------------------ */

// Trainable state of the network: the three non-input neurons' bias and two
// weights. Nine scalars, so the compiler keeps the whole thing in registers.
struct Network {
    float t2, w20, w21; // hidden neuron 2
    float t3, w30, w31; // hidden neuron 3
    float t4, w40, w41; // output neuron 4
};

// Values produced by one training step, written back once at the end.
struct Activations {
    float h2, h3, y;    // outputs of neurons 2, 3, 4
    float e2, e3, e4;   // error terms of neurons 2, 3, 4
};

__host__ __device__ __forceinline__ float sigmoid(float z) {
#ifdef __CUDA_ARCH__
    return 1.0f / (1.0f + __expf(-z)); // fast single-precision intrinsic on the GPU
#else
    return 1.0f / (1.0f + expf(-z));
#endif
}

// One epoch for one sample: forward pass followed by backpropagation.
// The update order is identical to the original backPropagate(): the output
// neuron's weights are updated first, and the hidden error terms then use the
// already-updated w40 / w41.
__host__ __device__ __forceinline__
void trainStep(Network &n, Activations &a, float x0, float x1, float target, float lr)
{
    // forward propagation
    a.h2 = sigmoid(n.t2 + n.w20 * x0 + n.w21 * x1);
    a.h3 = sigmoid(n.t3 + n.w30 * x0 + n.w31 * x1);
    a.y  = sigmoid(n.t4 + n.w40 * a.h2 + n.w41 * a.h3);

    // output neuron
    a.e4   = (target - a.y) * (a.y * (1.0f - a.y));
    n.t4  += lr * a.e4;
    n.w40 += lr * a.e4 * a.h2;
    n.w41 += lr * a.e4 * a.h3;

    // hidden neuron 3
    a.e3   = (n.w41 * a.e4) * (a.h3 * (1.0f - a.h3));
    n.t3  += lr * a.e3;
    n.w30 += lr * a.e3 * x0;
    n.w31 += lr * a.e3 * x1;

    // hidden neuron 2
    a.e2   = (n.w40 * a.e4) * (a.h2 * (1.0f - a.h2));
    n.t2  += lr * a.e2;
    n.w20 += lr * a.e2 * x0;
    n.w21 += lr * a.e2 * x1;
}

/* ------------------------------------------------------------------------ */
/* Device                                                                   */
/* ------------------------------------------------------------------------ */

// Thread `net` trains network copy `net` on training sample `net % TD_X` for
// `epochs` iterations. All networks are independent, so there is no
// synchronization. The grid-stride loop lets any launch size cover any
// number of networks.
__global__ void trainNetworks(float *networks, int numNetworks, int epochs, float lr)
{
    for (int net = blockIdx.x * blockDim.x + threadIdx.x;
         net < numNetworks;
         net += gridDim.x * blockDim.x)
    {
        float *g = networks + net * NETWORK_SIZE;

        const int   s      = net % TD_X;
        const float x0     = dev_trainingData[s][0][0];
        const float x1     = dev_trainingData[s][0][1];
        const float target = dev_trainingData[s][1][0];

        // load once: global memory -> registers
        Network n;
        n.t2 = g[IDX(2, THRESHOLD)]; n.w20 = g[IDX(2, WEIGHT0)]; n.w21 = g[IDX(2, WEIGHT1)];
        n.t3 = g[IDX(3, THRESHOLD)]; n.w30 = g[IDX(3, WEIGHT0)]; n.w31 = g[IDX(3, WEIGHT1)];
        n.t4 = g[IDX(4, THRESHOLD)]; n.w40 = g[IDX(4, WEIGHT0)]; n.w41 = g[IDX(4, WEIGHT1)];

        Activations a = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        // the whole training loop runs on registers
        for (int i = 0; i < epochs; i++) {
            trainStep(n, a, x0, x1, target, lr);
        }

        // store once: registers -> global memory, same fields the original
        // kernel leaves behind (weights, last outputs and last error terms)
        g[IDX(0, OUTPUT)] = x0;
        g[IDX(1, OUTPUT)] = x1;
        g[IDX(2, THRESHOLD)] = n.t2; g[IDX(2, WEIGHT0)] = n.w20; g[IDX(2, WEIGHT1)] = n.w21;
        g[IDX(2, OUTPUT)]    = a.h2; g[IDX(2, ERROR)]   = a.e2;
        g[IDX(3, THRESHOLD)] = n.t3; g[IDX(3, WEIGHT0)] = n.w30; g[IDX(3, WEIGHT1)] = n.w31;
        g[IDX(3, OUTPUT)]    = a.h3; g[IDX(3, ERROR)]   = a.e3;
        g[IDX(4, THRESHOLD)] = n.t4; g[IDX(4, WEIGHT0)] = n.w40; g[IDX(4, WEIGHT1)] = n.w41;
        g[IDX(4, OUTPUT)]    = a.y;  g[IDX(4, ERROR)]   = a.e4;
    }
}

/* ------------------------------------------------------------------------ */
/* Host                                                                     */
/* ------------------------------------------------------------------------ */

// Random weights and biases in [-0.5, 0.5], same fields and same rand()
// consumption order as __setNeurons__ in the original.
static void setNeurons(float *network, unsigned seed)
{
    srand(seed);
    for (int i = 0; i < NETWORK_SIZE; i++) {
        int field = i % VARS_PER_NEURON;
        if (field == THRESHOLD || field == WEIGHT0 || field == WEIGHT1) {
            network[i] = 0.5f - (float)(rand() / (double)RAND_MAX);
        }
    }
}

// Host-side forward pass over the 25-float layout. Returns the network output.
static float forwardPropHost(const float input[], float *network)
{
    network[IDX(0, OUTPUT)] = input[0];
    network[IDX(1, OUTPUT)] = input[1];
    for (int i = 2; i <= 3; i++) {
        float z = network[IDX(i, THRESHOLD)] +
                  network[IDX(i, WEIGHT0)] * network[IDX(0, OUTPUT)] +
                  network[IDX(i, WEIGHT1)] * network[IDX(1, OUTPUT)];
        network[IDX(i, OUTPUT)] = sigmoid(z);
    }
    float z = network[IDX(4, THRESHOLD)] +
              network[IDX(4, WEIGHT0)] * network[IDX(2, OUTPUT)] +
              network[IDX(4, WEIGHT1)] * network[IDX(3, OUTPUT)];
    network[IDX(4, OUTPUT)] = sigmoid(z);
    return network[IDX(4, OUTPUT)];
}

static void printNetworkInfo(void)
{
    printf("Number of inputs: %d\n", 2);
    printf("Number of hidden layers: %d\n", 2);
    printf("Number of output: %d\n", 1);
    printf("Number of iterations: %d\n", NUMB_OF_EPOCHS);
    printf("Learning Rate: %.2f\n", (double)LEARNING_RATE);
}

static void printResult(const float result[])
{
    printf("    Input 1    |    Input 2    | Target Result |  Result    \n");
    printf("-------------------------------------------------------------\n");
    for (int i = 0; i < TD_X; i++) {
        for (int j = 0; j < 2; j++) {
            printf("    %.5f    |", (double)TRAINING_DATA[i][0][j]);
        }
        printf("    %.5f    |   %.5f   \n", (double)TRAINING_DATA[i][1][0], (double)result[i]);
    }
}

int main(int argc, char **argv)
{
    unsigned seed = (argc > 1) ? (unsigned)strtoul(argv[1], NULL, 10) : (unsigned)time(NULL);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaFree(0)); // create the context now so it is not timed later

    printNetworkInfo();
    printf("Seed: %u\n", seed);

    const size_t nBytes = (size_t)NUMB_OF_NETWORKS * NETWORK_SIZE * sizeof(float);
    float *host_networks = (float *)calloc((size_t)NUMB_OF_NETWORKS * NETWORK_SIZE, sizeof(float));
    if (host_networks == NULL) {
        fprintf(stderr, "calloc failed\n");
        return EXIT_FAILURE;
    }

    // initialize the first network and show its untrained outputs
    setNeurons(host_networks, seed);
    float result[TD_X];
    for (int i = 0; i < TD_X; i++) {
        result[i] = forwardPropHost(TRAINING_DATA[i][0], host_networks);
    }
    printResult(result);

    // every network starts from the same initial weights
    for (int k = 1; k < NUMB_OF_NETWORKS; k++) {
        memcpy(host_networks + k * NETWORK_SIZE, host_networks, NETWORK_SIZE * sizeof(float));
    }

    // device memory: one buffer for all networks, training data in constant memory
    float *dev_networks = NULL;
    CUDA_CHECK(cudaMalloc(&dev_networks, nBytes));
    CUDA_CHECK(cudaMemcpyToSymbol(dev_trainingData, TRAINING_DATA, sizeof(TRAINING_DATA)));

    cudaEvent_t startTotal, startKernel, stopKernel, stopTotal;
    CUDA_CHECK(cudaEventCreate(&startTotal));
    CUDA_CHECK(cudaEventCreate(&startKernel));
    CUDA_CHECK(cudaEventCreate(&stopKernel));
    CUDA_CHECK(cudaEventCreate(&stopTotal));

    printf("Training..\n");
    CUDA_CHECK(cudaEventRecord(startTotal, 0));

    CUDA_CHECK(cudaMemcpy(dev_networks, host_networks, nBytes, cudaMemcpyHostToDevice));

    // one block holding all networks: with 4 networks that is a single warp
    dim3 block(NUMB_OF_NETWORKS < 128 ? NUMB_OF_NETWORKS : 128);
    dim3 grid((NUMB_OF_NETWORKS + block.x - 1) / block.x);

    CUDA_CHECK(cudaEventRecord(startKernel, 0));
    trainNetworks<<<grid, block>>>(dev_networks, NUMB_OF_NETWORKS, NUMB_OF_EPOCHS, LEARNING_RATE);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stopKernel, 0));

    CUDA_CHECK(cudaMemcpy(host_networks, dev_networks, nBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stopTotal, 0));
    CUDA_CHECK(cudaEventSynchronize(stopTotal));

    printf("[done training]\n");

    // evaluate each trained network on the sample it was trained on
    for (int i = 0; i < TD_X; i++) {
        result[i] = forwardPropHost(TRAINING_DATA[i][0], host_networks + i * NETWORK_SIZE);
    }
    printResult(result);

    float kernelTime = 0.0f, totalTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernelTime, startKernel, stopKernel));
    CUDA_CHECK(cudaEventElapsedTime(&totalTime, startTotal, stopTotal));
    printf("Kernel time on GPU: %3.6f ms \n", kernelTime);
    printf("Compute time on GPU: %3.6f ms (copies + kernel)\n", totalTime);

    CUDA_CHECK(cudaEventDestroy(startTotal));
    CUDA_CHECK(cudaEventDestroy(startKernel));
    CUDA_CHECK(cudaEventDestroy(stopKernel));
    CUDA_CHECK(cudaEventDestroy(stopTotal));
    CUDA_CHECK(cudaFree(dev_networks));
    free(host_networks);
    return 0;
}
