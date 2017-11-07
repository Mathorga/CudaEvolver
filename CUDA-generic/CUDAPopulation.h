#ifndef __CUDA_POPULATION__
#define __CUDA_POPULATION__

#include "CUDAGenome.h"

class CUDAPopulation {
public:
    CUDAGenome **individuals;
    CUDAGenome **offspring;
    CUDAGenome *tmp;

    CUDAPopulation(unsigned int popSize, unsigned int genNum);
    // Scales the individuals' scores to fitnesses.
    __device__ void scale();
    __device__ void step();

    CUDAGenome *best();
    CUDAGenome *worst();

    __host__ __device__ unsigned int getSize() {
        return size;
    }
    __host__ __device__ unsigned int getGenNumber() {
        return genNumber;
    }
    __host__ __device__ CUDAGenome *getIndividual(unsigned int index) {
        return individuals[index];
    }
    __host__ __device__ CUDAGenome **getIndividuals() {
        return individuals;
    }
    __host__ __device__ CUDAGenome ***getIndividualsAddress() {
        return &individuals;
    }
    __host__ __device__ CUDAGenome **getIndividualAddress(unsigned int index) {
        return &individuals[index];
    }
private:
    // Performs fitness-proportionate selection to get an individual from the population.
    __device__ CUDAGenome *select();

private:
    bool initialized;
    unsigned int size;
    unsigned int genNumber;
    unsigned int currentGen;
};

__global__ void evolve();

#endif
