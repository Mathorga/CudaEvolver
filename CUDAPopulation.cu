#include "CUDAPopulation.h"
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>

void evolve(CUDAPopulation *pop, dim3 genomeSize) {
    dim3 popSize(pop->getSize());
    printf("Population size:%u\n", pop->getSize());

    // Copy the population on the device.
    CUDAPopulation *d_pop;
    cudaMalloc(&d_pop, sizeof(CUDAPopulation));
    cudaMemcpy(d_pop, pop, sizeof(CUDAPopulation), cudaMemcpyHostToDevice);
    printf("Copied the population on the device\n");

    CUDAGenome **d_individuals;
    d_individuals = (CUDAGenome **) malloc(pop->getSize() * sizeof(CUDAGenome *));
    for (unsigned int i = 0; i < pop->getSize(); i++) {
        pop->individuals[0]->allocateCopySingle(&(d_individuals[i]), &(pop->individuals[i]), cudaMemcpyHostToDevice);
    }
    printf("Copied the individuals on the device\n");

    CUDAGenome **tmpD_individuals;
    cudaMalloc(&tmpD_individuals, pop->getSize() * sizeof(CUDAGenome *));
    cudaMemcpy(tmpD_individuals, d_individuals, pop->getSize() * sizeof(CUDAGenome *), cudaMemcpyHostToDevice);

    cudaMemcpy(&(d_pop->individuals), &tmpD_individuals, sizeof(CUDAGenome **), cudaMemcpyHostToDevice);


    // Evolve.
    printf("Starting evolution loop\n");
    for (unsigned int i = 0; i < pop->getGenNumber(); i++) {
        evaluate<<<popSize, genomeSize>>>(d_pop);
        cudaDeviceSynchronize();
        step<<<popSize, genomeSize>>>(d_pop);
        cudaDeviceSynchronize();
    }

    // TODO Copy the population back to the host.
    cudaMemcpy(pop, d_pop, sizeof(CUDAPopulation *), cudaMemcpyDeviceToHost);
}

__global__ void evaluate(CUDAPopulation *pop) {
    if (threadIdx.x == 0) {
        printf("Started evaluation of individual %d\n", blockIdx.x);
        printf("Address:%p\n", pop->individuals[blockIdx.x]);
        printf("Size:%u\n", pop->individuals[blockIdx.x]->getXSize());
        printf("checksNumber:%d\n", ((CUDAPathGenome *) (pop->individuals[blockIdx.x]))->getChecksNum());
    }
    pop->individuals[blockIdx.x]->evaluate();
}

__global__ void step(CUDAPopulation *pop) {
    pop->step();
}

CUDAPopulation::CUDAPopulation(unsigned int popSize, unsigned int genNum, CUDAGenome *genome, Objective obj) {
    printf("Starting creation\n");
    genNumber = genNum;
    currentGen = 0;
    initialized = false;
    size = popSize;
    individuals = (CUDAGenome **) malloc(size * sizeof(CUDAGenome *));
    printf("Allocated individuals on host\n");

    // genome->allocateIndividuals(d_individuals, size);
    // cudaMalloc(&d_individuals, size * sizeof(CUDAGenome *));

    printf("Allocated individuals' pointer on device\n");
    for (unsigned int i = 0; i < size; i++) {
        individuals[i] = genome->clone();
        printf("Cloned individual %d\n", i);
        // cudaMalloc(&d_individuals[i], sizeof(CUDAGenome *));
        // printf("Allocated individual %d on device\n", i);
    }
    // cudaMemcpy(d_individuals, individuals, size * sizeof(CUDAGenome *), cudaMemcpyHostToDevice);
    // printf("Copied individuals' reference from host to device\n");
}

void CUDAPopulation::initialize() {
    printf("Starting initialization\n");
    if (!initialized) {
        for (unsigned int i = 0; i < size; i++) {
            individuals[i]->initialize();
            printf("Initialized individual %d on host\n", i);
        }
        initialized = true;
    }
}

__device__ void CUDAPopulation::step() {
    // Create a temporary population.
    CUDAGenome *ind = (CUDAGenome *) malloc(sizeof(CUDAGenome));
    memcpy(ind, individuals[blockIdx.x], sizeof(CUDAGenome));

    // Select.
    CUDAGenome *partner = select();
    __syncthreads();

    // Crossover.
    if (threadIdx.x == 0) {
        offspring[blockIdx.x] = (CUDAGenome *) malloc(sizeof(CUDAGenome *));
    }
    __syncthreads();
    individuals[blockIdx.x]->crossover(partner, offspring[blockIdx.x]);

    // Mutate.
    offspring[blockIdx.x]->mutate();
    __syncthreads();

    // Overwrite the old individual with the new one.
    if (threadIdx.x == 0) {
        individuals[blockIdx.x] = offspring[blockIdx.x];
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Copy the best from the old pop to the new one.
        // TODO.
    }
}

__device__ CUDAGenome *CUDAPopulation::select() {
    float totalFitness = 0.0;
    float previousProb = 0.0;

    if (threadIdx.x == 0) {
        scale();
        sort();
    }
    __syncthreads();

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

__device__ void CUDAPopulation::sort() {
    int l;
    CUDAGenome *tmp = (CUDAGenome *) malloc(sizeof(CUDAGenome));

    if (size % 2 == 0) {
        l = size / 2;
    } else {
        l = (size / 2) + 1;
    }

    for (int i = 0; i < l; i++) {
        // Even phase.
        if (!(blockIdx.x & 1) && (blockIdx.x < (size - 1))) {
            if (individuals[blockIdx.x]->getFitness() > individuals[blockIdx.x + 1]->getFitness()) {
                CUDAGenome *tmp = individuals[blockIdx.x];
                individuals[blockIdx.x] = individuals[blockIdx.x + 1];
                individuals[blockIdx.x + 1] = tmp;
            }
        }
        __syncthreads();

        // Odd phase.
        if ((blockIdx.x & 1) && (blockIdx.x < (size - 1))) {
            if (individuals[blockIdx.x]->getFitness() > individuals[blockIdx.x + 1]->getFitness()) {
                CUDAGenome *tmp = individuals[blockIdx.x];
                individuals[blockIdx.x] = individuals[blockIdx.x + 1];
                individuals[blockIdx.x + 1] = tmp;
            }
        }
        __syncthreads();
    }
}
