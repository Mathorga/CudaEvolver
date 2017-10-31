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
    Objective aim;
    bool initialized;
    unsigned int size;
    unsigned int genNumber;
    unsigned int currentGen;
};

// Performs an avaluation on the elements of the given population.
__global__ void evaluate(CUDAPopulation *pop);

// Implements ascending odd-even transposition sort on the individuals of the population.
__global__ void sort(CUDAPopulation *pop);

// Performs an evolution step (selection, crossover, mutation, replacement) on the given population.
__global__ void step(CUDAPopulation *pop);

// Outputs the best individual of the specified population to the specified file.
__global__ void outputBest(CUDAPopulation *pop, char *string);

// Outputs the worst individual of the specified population to the specified file.
__global__ void outputWorst(CUDAPopulation *pop, char *string);

#endif
