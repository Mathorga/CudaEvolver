#include "CUDAPopulation.h"
#include <curand.h>
#include <curand_kernel.h>

__global__ void evolve(CUDAPopulation *pop) {
    for (unsigned int i = 0; i < pop->getGenNumber(); i++) {
        pop->step();
        __syncthreads();
    }
}

__global__ void step(CUDAPopulation *pop) {
    pop->step();
}

CUDAPopulation::CUDAPopulation(unsigned int popSize, unsigned int genNum, CUDAGenome *genome, Objective obj) {
    genNumber = genNum;
    currentGen = 0;
    initialized = false;
    size = popSize;
    individuals = (CUDAGenome **) malloc(size * sizeof(CUDAGenome *));
    cudaMalloc(&d_individuals, size * sizeof(CUDAGenome *));
    for (unsigned int i = 0; i < size; i++) {
        individuals[i] = genome->clone();
        cudaMalloc(&d_individuals[i], sizeof(CUDAGenome));
    }
    cudaMemcpy(d_individuals, individuals, size * sizeof(CUDAGenome *), cudaMemcpyHostToDevice);
}

void CUDAPopulation::initialize() {
    if (!initialized) {
        for (unsigned int i = 0; i < size; i++) {
            individuals[i]->initialize();
            cudaMemcpy(d_individuals[i], individuals[i], sizeof(CUDAGenome), cudaMemcpyHostToDevice);
        }
        initialized = true;
    }
}

// __device__ void CUDAPopulation::evolve() {
//     initialize();
//     dim3 gridSize(size);
//     dim3 blockSize(individuals[0]->getXSize(), individuals[0]->getYSize(), individuals[0]->getZSize());
//     for (unsigned int i = 0; i < genNumber; i++) {
//         step<<<gridSize, blockSize>>>();
//     }
// }

__device__ void CUDAPopulation::step() {
    // Create a temporary population.
    CUDAGenome *ind = (CUDAGenome *) malloc(sizeof(CUDAGenome));
    memcpy(ind, d_individuals[blockIdx.x], sizeof(CUDAGenome));

    // Evaluate.
    evaluate();
    __syncthreads();

    // Select.
    CUDAGenome *partner = select();
    __syncthreads();

    // Crossover.
    CUDAGenome *offspring = (CUDAGenome *) malloc(sizeof(CUDAGenome *));
    individuals[blockIdx.x]->crossover(partner, offspring);

    // Mutate.
    offspring->mutate();

    // Synchronize.
    __syncthreads();

    // Overwrite the old individual with the new one.
    if (threadIdx.x == 0) {
        d_individuals[blockIdx.x] = offspring;
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Copy the best from the old pop to the new one.
        // TODO.
    }
}

__device__ void CUDAPopulation::evaluate() {
    d_individuals[blockIdx.x]->evaluate();
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
