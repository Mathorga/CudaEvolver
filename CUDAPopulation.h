#ifndef __CUDA_POPULATION__
#define __CUDA_POPULATION__

#include "CUDAGenome.h"

class CUDAPopulation {
public:
    enum Objective {
        MINIMIZE = 0,
        MAXIMIZE = 1
    };

    CUDAPopulation(unsigned int popSize, unsigned int genNum, CUDAGenome *genome, Objective obj = MAXIMIZE);

    void initialize();
    // __device__ void evolve();
    __device__ void step();

    CUDAGenome *best();
    CUDAGenome *worst();

    __host__ __device__ unsigned int getSize() {
        return size;
    }
    __host__ __device__ unsigned int getGenNumber() {
        return genNumber;
    }

private:
    __device__ void evaluate();
    // Selects an individual from the population.
    __device__ CUDAGenome *select();
    // Implements ascending odd-even transposition sort on the individuals of the population.
    __device__ void sort();
    __device__ void scale();

private:
    Objective aim;
    bool initialized;
    unsigned int size;
    unsigned int genNumber;
    unsigned int currentGen;
    CUDAGenome **individuals;
    CUDAGenome **d_individuals;
};

__global__ void evolve(CUDAPopulation *pop);
__global__ void step(CUDAPopulation *pop);

#endif
