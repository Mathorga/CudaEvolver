#include "CUDAPopulation.h"
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>

__global__ void evaluate(CUDAPopulation *pop) {
    pop->individuals[blockIdx.x]->evaluate();
    pop->scale();
}

__global__ void sort() {
    if (blockIdx.x == 0) {
        int l;
        CUDAGenome *tmp = (CUDAGenome *) malloc(sizeof(CUDAGenome));

        if (size % 2 == 0) {
            l = size / 2;
        } else {
            l = (size / 2) + 1;
        }

        for (int i = 0; i < l; i++) {
            // Even phase.
            if (!(threadIdx.x & 1) && (threadIdx.x < (size - 1))) {
                if (individuals[threadIdx.x]->getFitness() > individuals[threadIdx.x + 1]->getFitness()) {
                    CUDAGenome *tmp = individuals[threadIdx.x];
                    individuals[threadIdx.x] = individuals[threadIdx.x + 1];
                    individuals[threadIdx.x + 1] = tmp;
                }
            }
            __syncthreads();

            // Odd phase.
            if ((threadIdx.x & 1) && (threadIdx.x < (size - 1))) {
                if (individuals[threadIdx.x]->getFitness() > individuals[threadIdx.x + 1]->getFitness()) {
                    CUDAGenome *tmp = individuals[threadIdx.x];
                    individuals[threadIdx.x] = individuals[threadIdx.x + 1];
                    individuals[threadIdx.x + 1] = tmp;
                }
            }
            __syncthreads();
        }
    }
}

__global__ void step(CUDAPopulation *pop) {
    pop->step();
}

__global__ void outputBest(CUDAPopulation *pop, *string) {
    if (blockIdx.x == 0) {
        // Output the last (best) individual.
        pop->individuals[pop->getSize() - 1]->output(fileName);
    }
}

__global__ void outputWorst(CUDAPopulation *pop, char *string) {
    if (blockIdx.x == 0) {
        // Output the first (worst) individual.
        pop->individuals[0]->output(string);
    }
}



CUDAPopulation::CUDAPopulation(unsigned int popSize, unsigned int genNum, Objective obj) {
    // printf("Starting creation\n");
    genNumber = genNum;
    currentGen = 0;
    initialized = false;
    size = popSize;
    individuals = (CUDAGenome **) malloc(size * sizeof(CUDAGenome *));
    offspring = (CUDAGenome **) malloc(size * sizeof(CUDAGenome *));
}

__device__ void CUDAPopulation::step() {
    // printf("\n");
    // for (unsigned int i = 0; i < 5; i++) {
    //     printf("x:%u\ty:%u\n", ((CUDAPathGenome *) individuals[blockIdx.x])->path[i].x, ((CUDAPathGenome *) individuals[blockIdx.x])->path[i].y);
    // }

    // Create a temporary population.
    CUDAGenome *ind = (CUDAGenome *) malloc(sizeof(CUDAGenome));
    memcpy(ind, individuals[blockIdx.x], sizeof(CUDAGenome));

    // Select.
    // printf("Selection\n");
    CUDAGenome *partner = select();
    __syncthreads();

    // if (threadIdx.x == 0) {
    //     printf("\n");
    //     for (unsigned int i = 0; i < 5; i++) {
    //         printf("x:%u\ty:%u\n", ((CUDAPathGenome *) individuals[blockIdx.x])->path[i].x, ((CUDAPathGenome *) individuals[blockIdx.x])->path[i].y);
    //     }
    // }

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
        individuals[blockIdx.x] = offspring[blockIdx.x];

        // for (unsigned int i = 0; i < ((CUDAPathGenome *) individuals[blockIdx.x])->getChecksNum(); i++) {
        //     printf("x:%u\ty:%u\n", ((CUDAPathGenome *) individuals[blockIdx.x])->path[i].x, ((CUDAPathGenome *) individuals[blockIdx.x])->path[i].y);
        // }
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Copy the best from the old pop to the new one.
        // TODO.
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
