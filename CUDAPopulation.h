#ifndef __CUDA_POPULATION__
#define __CUDA_POPULATION__

#include "CUDAGenome.h"
#include "CUDAPathGenome.h"

class CUDAPopulation {
public:
    enum Objective {
        MINIMIZE,
        MAXIMIZE
    };
    enum SortBasis {
        SCORE,
        FITNESS
    };

    CUDAGenome **individuals;
    CUDAGenome **offspring;
    CUDAGenome *original;

    CUDAPopulation(unsigned int popSize, unsigned int genNum, Objective obj = MAXIMIZE);
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
    // Implements ascending odd-even transposition sort on the individuals of the population.
    __device__ void scale();
    __device__ void sort();

private:
    Objective aim;
    bool initialized;
    unsigned int size;
    unsigned int genNumber;
    unsigned int currentGen;
};

// Perform an avaluation on the elements of the given population.
__global__ void evaluate(CUDAPopulation *pop);

// Perform an evolution step (selection, crossover, mutation, replacement) on the given population.
__global__ void step(CUDAPopulation *pop);

#endif
