# CS79995 GPU Parallel Processing: Final Project

## Backpropagation Neural Network on CPU and GPU (CUDA)

This project builds a small feed-forward neural network from scratch and trains it with the **backpropagation** algorithm. It comes in two versions:

| Version | File(s) | Where training runs |
|---|---|---|
| CPU | [backProgCPU.cu](backProgCPU.cu) + [Neuron.cu](Neuron.cu) | Host (CPU), one thread |
| GPU | [backProgGPU.cu](backProgGPU.cu) | Device (NVIDIA GPU), one CUDA thread per training sample |

Each program prints the network's outputs before and after training, along with how long training took. The goal is to compare how the same learning problem behaves and performs on a CPU and on a GPU.

---

## 1. The learning problem

The network learns the logical **OR** function. There are four training samples, defined in `TRAINING_DATA` in both programs:

| Input 1 | Input 2 | Target |
|:---:|:---:|:---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

`TRAINING_DATA` is a `[4][2][2]` array (`TD_X × TD_Y × TD_Z`). `TRAINING_DATA[i][0]` holds the two inputs of sample `i`, and `TRAINING_DATA[i][1][0]` holds its target.

---

## 2. Network architecture

The network has a fixed **2 → 2 → 1** topology with five neurons:

```
   Input layer        Hidden layer        Output layer

   [n0] x1 ───┬──────▶ [n2] ──┐
              │  ╲  ╱          ├──────▶ [n4]  ──▶ y
              │   ╳            │
              │  ╱  ╲          │
   [n1] x2 ───┴──────▶ [n3] ──┘
```

- **Input neurons (n0, n1)** pass the input values through unchanged.
- **Hidden neurons (n2, n3)** each receive both inputs.
- **Output neuron (n4)** receives the outputs of both hidden neurons.

Every non-input neuron has two weights and a bias. The code calls the bias the `threshold`.

### Neuron state

| Field | Meaning |
|---|---|
| `threshold` | Bias term added to the weighted sum |
| `weight[0]`, `weight[1]` | Weights on the neuron's two incoming connections |
| `output` | Activation computed during the forward pass |
| `error` | Error term (delta) computed during the backward pass |

The CPU version stores this in a `struct neuron`. The GPU version flattens it into a `float` array with **5 values per neuron** (so 25 floats per network), laid out like this:

```
index within a neuron:  0          1          2          3        4
                        threshold  weight[0]  weight[1]  output   error
```

Neuron `i`'s field `f` is stored at `neurons[i * 5 + f]`. For example, the network's final output (neuron 4, field 3) is at index `23`.

### Hyperparameters

Both programs set these with `#define` at the top of the file:

| Constant | Value | Meaning |
|---|---|---|
| `LEARNING_RATE` | `0.25` | Step size for weight updates |
| `NUMB_OF_EPOCHS` | `100000` | Number of training iterations |

Weights and biases start with random values in `[-0.5, 0.5]`, and the generator is seeded with the current time. As a result, the "before training" output differs on every run.

---

## 3. The algorithm

### Activation function

Every hidden and output neuron uses the **sigmoid** function:

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma'(z) = \sigma(z)\,\bigl(1 - \sigma(z)\bigr)$$

The derivative is computed from the neuron's output that was already calculated (`output * (1 - output)`), so the exponential does not need to be evaluated again.

### Forward propagation

For each hidden neuron `h ∈ {n2, n3}` and for the output neuron `n4`:

```
z = threshold + weight[0] * a + weight[1] * b
output = sigmoid(z)
```

Here `(a, b)` are the two inputs for the hidden neurons, and the two hidden outputs for the output neuron.

### Backpropagation (online / stochastic gradient descent)

After each forward pass, the program compares the output with the target and updates the weights immediately:

1. **Output neuron**
   ```
   δ4 = (target − y) · y(1 − y)
   threshold4 += η · δ4
   weight4[k] += η · δ4 · hidden_output[k]
   ```
2. **Hidden neurons** (n3, then n2). Each hidden neuron's error is sent back through the output neuron's weight on that connection.
   ```
   δh = (weight4[h] · δ4) · outh(1 − outh)
   thresholdh += η · δh
   weighth[k] += η · δh · input[k]
   ```

Note: the output neuron's weights are updated *before* the hidden deltas are computed, so the hidden deltas use the already-updated `weight4`. Textbook backprop uses the old weights. With a small learning rate the difference barely matters, but it is a deviation from the standard algorithm.

---

## 4. CPU implementation

**Files:** [backProgCPU.cu](backProgCPU.cu), [Neuron.cu](Neuron.cu)

[Neuron.cu](Neuron.cu) is not compiled separately. [backProgCPU.cu](backProgCPU.cu) pulls it in with `#include "Neuron.cu"` after defining the constants and `TRAINING_DATA` that it depends on. Neuron.cu contains:

| Function | Purpose |
|---|---|
| `applyActivationFunction` / `derivative` | Sigmoid and its derivative |
| `setNeurons` | Randomly initializes weights and biases and assigns layer types |
| `forwardProp` | Runs one forward pass for one input sample |
| `backpropError` | Computes the deltas and updates weights and biases |
| `trainOnCPU` | Main training routine (described below) |
| `trainOnCPU2` | Alternative routine that trains four separate networks, one per sample, the same way the GPU version does (not called by default) |
| `printResult`, `printTrainingData`, `printNetworkInfo` | Output helpers |

**Training flow (`trainOnCPU`):**

1. Evaluate the untrained network on all four samples and print the table.
2. For each of `NUMB_OF_EPOCHS` epochs, loop over all four samples and run forward propagation followed by backpropagation.
3. Evaluate the trained network on all four samples and print the table.

This is standard training. **A single network** learns all four samples, so it ends up approximating the full OR function.

The program times training with CUDA events (`cudaEventRecord` / `cudaEventElapsedTime`). That is why it is a `.cu` file and needs `nvcc` and a CUDA device even though all the work runs on the CPU.

---

## 5. GPU implementation

**File:** [backProgGPU.cu](backProgGPU.cu), a standalone file.

### Device functions

| Function | Qualifier | Purpose |
|---|---|---|
| `__applyActivationFunction__` | `__device__` | Sigmoid on the GPU |
| `derivative` | `__device__` | Sigmoid derivative on the GPU |
| `forwardPropagate` | `__device__` | Forward pass over the flattened 25-float network |
| `backPropagate` | `__device__` | Backward pass and weight update over the flattened network |
| `trainNeurons` | `__global__` | Training kernel |

### Host functions

`__setNeurons__` (random initialization), `__forwardProp__` (host-side evaluation), `applyActivationFunction`, `_printNetworkInfo_` and `_printResult_`.

### Parallelization strategy

The GPU version parallelizes **across training samples**:

1. The host initializes one network (`host_neurons0`) and prints its untrained outputs.
2. The host allocates **four copies** of the network in device global memory (`dev_neuronset0..3`) and copies the same initial weights into each one.
3. The kernel is launched as `trainNeurons<<<4, 1>>>`: 4 blocks of 1 thread each.
4. Thread `idx` (0–3) trains network copy `idx` on **only training sample `idx`**, running `NUMB_OF_EPOCHS` forward and backward passes. The four threads run at the same time and never communicate.
5. After `cudaDeviceSynchronize()`, the four networks are copied back to the host. Each one is evaluated on its own sample and the results are printed.

The training data is also defined as a local array inside the kernel, because the host `TRAINING_DATA` global is not accessible from the device.

### How this differs from the CPU version

The GPU version trains **four independent networks**, and each one fits a single input/target pair. The CPU version (`trainOnCPU`) trains **one network** on all four samples. Keep this in mind when you compare them:

- The GPU results show that each network reaches its own target. They do **not** give one model that computes OR for every input.
- The CPU equivalent of the GPU approach is `trainOnCPU2` in [Neuron.cu](Neuron.cu). Use it instead of `trainOnCPU` for a like-for-like timing comparison. Its signature takes four neuron arrays, so `main` needs to declare and initialize them.
- The GPU timer covers host↔device memory copies, the kernel, host-side evaluation and `printf`. The CPU timer covers training plus the before/after evaluation and printing.

Each GPU thread still runs its epochs one after another, and the kernel uses only 4 threads. This workload is far too small to occupy a GPU, so kernel launch and memory transfer overhead dominate the run time. The project is best read as a demonstration of mapping backpropagation onto CUDA, not as a fast implementation.

---

## 6. Building and running

### Requirements

- An NVIDIA GPU with a working CUDA driver. **Both** programs call `cudaGetDeviceProperties` and use CUDA events.
- The CUDA Toolkit (`nvcc`).

### GPU version

```bash
nvcc backProgGPU.cu -o bpgpu.out
./bpgpu.out
```

### CPU version

Compile only `backProgCPU.cu`. It already includes `Neuron.cu`.

```bash
nvcc backProgCPU.cu -o bpcpu.out
./bpcpu.out
```

### Output

Each program prints:

1. The CUDA device in use.
2. The network configuration (inputs, hidden neurons, outputs, iterations, learning rate).
3. A results table **before** training, with random outputs.
4. `Training..` and then `[done training]`.
5. A results table **after** training. Each result should be close to its target (near 0 for `0,0` and near 1 for the others).
6. The elapsed time in milliseconds (`Compute time on CPU` / `Compute time on GPU`).

The results table looks like this:

```
    Input 1    |    Input 2    | Target Result |  Result
-------------------------------------------------------------
    0.00000    |    0.00000    |    0.00000    |   0.00122
    0.00000    |    1.00000    |    1.00000    |   0.98760
    1.00000    |    0.00000    |    1.00000    |   0.99674
    1.00000    |    1.00000    |    1.00000    |   0.97543
```

### Experimenting

- To change the training length or step size, edit `NUMB_OF_EPOCHS` or `LEARNING_RATE` at the top of either `.cu` file.
- To train a different 2-input boolean function (for example AND), change the targets in `TRAINING_DATA`. In the GPU version, change them in **both** the host global and the copy inside the `trainNeurons` kernel.
- XOR is a good test for the CPU version, because a single network has to use its hidden layer to solve it. Depending on the random initialization, it may need more epochs. In the GPU version, each network only ever sees one sample, so it will reach any target you set, including XOR's, without really learning the function.

---

## 7. Known limitations

- **Fixed topology.** The 2-2-1 structure is hard-coded throughout: the indices, the two-element weight arrays and the explicit per-neuron backprop steps.
- **Out-of-bounds loop in `forwardProp` (CPU).** The loop bound is `sizeof(neurons)`. That is the size of a *pointer* (8 on 64-bit systems), not the number of neurons (5), so the loop reads past the end of the array. It should loop to `5`.
- **Mixed precision.** The GPU version stores the network as `float` but computes in `double`, and the CPU version uses `double` throughout. Results can differ slightly between the two.
- **Exit code.** Both `main` functions `return(1)`, which the shell treats as failure even when the run succeeds.
- **Unchecked CUDA calls.** Only device setup uses `CHECK(...)`. The `cudaMalloc`, `cudaMemcpy` and kernel launch results are not checked.
- **Unused code.** The `CUDA_CALL` macro and several STL includes/usings (`vector` is used only by the GPU version) are not needed.

---

## Repository layout

```
.
├── README.md          This file
├── backProgCPU.cu     CPU entry point (includes Neuron.cu)
├── Neuron.cu          CPU neuron struct, forward/backward pass, training and printing helpers
└── backProgGPU.cu     Standalone GPU version: device functions, training kernel and host driver
```
