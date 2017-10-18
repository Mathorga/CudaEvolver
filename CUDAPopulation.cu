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

    // Select.
    __syncthreads();
    CUDAGenome *parent1 = select();
    CUDAGenome *parent2 = select();

    // Crossover.
    CUDAGenome *child = parent1->crossover(parent2);

    // Mutate.
    child->mutate();

    // Synchronize.
    __syncthreads();

    // Overwrite the old individual with the new one.
    // d_individuals[blockIdx.x] = child;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Copy the best from the old pop to the new one.
        // TODO.
    }
}

__device__ void CUDAPopulation::evaluate() {
    d_individuals[blockIdx.x]->evaluate();
}

__device__ CUDAGenome *CUDAPopulation::select() {
    if (threadIdx.x == 0) {
        sort();
        scale();
    }
    __syncthreads();

    // TODO Implement so that threads of the same block select the same genome.
    curandState_t state;
    curand_init((unsigned long) clock(), blockIdx.x, 0, &state);
    unsigned int random = curand(&state);


    return individuals[0];
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
            if (individuals[blockIdx.x]->getScore() > individuals[blockIdx.x + 1]->getScore()) {
                CUDAGenome *tmp = individuals[blockIdx.x];
                individuals[blockIdx.x] = individuals[blockIdx.x + 1];
                individuals[blockIdx.x + 1] = tmp;
            }
        }
        __syncthreads();

        // Odd phase.
        if ((blockIdx.x & 1) && (blockIdx.x < (size - 1))) {
            if (individuals[blockIdx.x]->getScore() > individuals[blockIdx.x + 1]->getScore()) {
                CUDAGenome *tmp = individuals[blockIdx.x];
                individuals[blockIdx.x] = individuals[blockIdx.x + 1];
                individuals[blockIdx.x + 1] = tmp;
            }
        }
        __syncthreads();
    }
}
