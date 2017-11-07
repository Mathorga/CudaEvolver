#include "CUDAPopulation.h"
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>




__global__ void evolve(CUDAPopulation *pop) {

}





CUDAPopulation::CUDAPopulation(unsigned int popSize, unsigned int genNum) {
    genNumber = genNum;
    currentGen = 0;
    initialized = false;
    size = popSize;
    individuals = (CUDAGenome **) malloc(size * sizeof(CUDAGenome *));
    offspring = (CUDAGenome **) malloc(size * sizeof(CUDAGenome *));
}

__device__ void CUDAPopulation::step() {
    // Select.
    // printf("Selection\n");
    CUDAGenome *partner = select();
    __syncthreads();

    // Crossover.
    // printf("Crossover\n");
    individuals[blockIdx.x]->crossover(partner, &(offspring[blockIdx.x]));
    __syncthreads();

    // Mutate.
    // printf("Mutation\n");
    offspring[blockIdx.x]->mutate();
    __syncthreads();

    // Overwrite the old individual with the new one.
    if (threadIdx.x == 0) {
        individuals[blockIdx.x]->dealloc();
        *(individuals[blockIdx.x]) = *(offspring[blockIdx.x]);
        *(offspring[blockIdx.x]) = *(individuals[blockIdx.x]->clone());
        // printf("%u\n", offspring[blockIdx.x]->getXSize());
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // TODO Copy the best from the old pop to the new one.
    }
    __syncthreads();
}

__device__ CUDAGenome *CUDAPopulation::select() {
    float totalFitness = 0.0;
    float previousProb = 0.0;

    // Threads of the same block select the same genome by generating the same pseudo-random number.
    curandState_t state;
    curand_init((unsigned long) clock(), blockIdx.x, 0, &state);
    float random = curand_uniform(&state);

    // Calculate the total fitness.
    for (unsigned int i = 0; i < size; i++) {
        totalFitness += individuals[i]->getFitness();
    }

    // Calculate the probability for each individual.
    for (unsigned int i = 0; i < size - 1; i++) {
        float prob = previousProb + (individuals[i]->getFitness() / totalFitness);
        if (random < prob) {
            return individuals[i];
        } else {
            previousProb += prob;
        }
    }
    return individuals[size - 1];
}

__device__ void CUDAPopulation::scale() {
    individuals[blockIdx.x]->scale(individuals[size - 1]->getScore());
}
