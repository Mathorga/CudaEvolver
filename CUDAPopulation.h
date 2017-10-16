#ifndef __CUDA_POPULATION__
#define __CUDA_POPULATION__

#include "CUDAGenome.h"

class CUDAPopulation {
public:
    enum {
        MINIMIZE = 0,
        MAXIMIZE = 1
    };

    CUDAPopulation(unsigned int popSize, unsigned int genNum, CUDAGenome &genome, int objective = MAXIMIZE);

    void initialize();
    void evolve();
    __global__ void step();

    CUDAGenome *best();
    CUDAGenome *worst();

private:
    __device__ void evaluate();
    __device__ CUDAGenome *select();

private:
    bool initialized;
    unsigned int size;
    unsigned int genNumber;
    unsigned int currentGen;
    CUDAGenome **individuals;
    CUDAGenome **d_individuals;
};

#endif
