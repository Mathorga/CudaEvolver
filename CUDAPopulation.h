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

    CUDAPopulation(unsigned int popSize, unsigned int genNum, CUDAGenome *genome, Objective obj = MAXIMIZE);

    void initialize();
    __device__ void step();

    CUDAGenome *best();
    CUDAGenome *worst();

    __host__ __device__ unsigned int getSize() {
        return size;
    }
    __host__ __device__ unsigned int getGenNumber() {
        return genNumber;
    }
    CUDAGenome *getHostIndividual(unsigned int index) {
        return individuals[index];
    }
    CUDAGenome **getHostIndividuals() {
        return individuals;
    }
    __device__ CUDAGenome **getDeviceIndividuals() {
        return d_individuals;
    }
    __device__ CUDAGenome *getDeviceIndividual(unsigned int index) {
        return d_individuals[index];
    }
    CUDAGenome ***getHostIndividualsAddress() {
        return &individuals;
    }
    CUDAGenome **getHostIndividualAddress(unsigned int index) {
        return &individuals[index];
    }
    CUDAGenome ***getDeviceIndividualsAddress() {
        return &d_individuals;
    }
    CUDAGenome **getDeviceIndividualAddress(unsigned int index) {
        return &d_individuals[index];
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
    CUDAGenome **individuals;
    CUDAGenome **d_individuals;
    CUDAGenome **offspring;
};

// Evolve the given population from start to finish.
void evolve(CUDAPopulation *pop, dim3 genomeSize);

// Perform an avaluation on the elements of the given population.
__global__ void evaluate(CUDAPopulation *pop);

// Perform an evolution step (selection, crossover, mutation, replacement) on the given population.
__global__ void step(CUDAPopulation *pop);

#endif
