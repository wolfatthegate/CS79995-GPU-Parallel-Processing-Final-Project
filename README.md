# CS79995-GPU-Parallel-Processing-Final-Project
Final Project - Back Propagation Neural Network

There are two version of the program, one is in CPU version (backProgCPU.cu), the other one is in GPU version (backProgGPU.cu). 

backProgCPU.cu needs a companion file Neuron.cu to be able to compile the program. 
Neuron.cu includes all the necessary functions that backProgCPU.cu needs to compile. 

backProgGPU.cu is a stand alone file where all the functions are in the same file. 

To compile the file, use the following command. 
<code>
nvcc backProgGPU.cu -o bpgpu.out
</code>


Then run the program by executing the following command. 
<code>
./bpgpu.out
</code>
