#include "CUDAPopulation.h"

__global__ void evolve(CUDAPopulation *pop) {
    for (unsigned int i = 0; i < pop->getGenNumber(); i++) {
        pop->step();
        __syncthreads();
    }
}

__global__ void step(CUDAPopulation *pop) {
    pop->step();
}

CUDAPopulation::CUDAPopulation(unsigned int popSize, unsigned int genNum, CUDAGenome *genome, int objective) {
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
    CUDAGenome *ind;
    // cudaMalloc(&ind, sizeof(CUDAGenome));
    memcpy(ind, d_individuals[blockIdx.x], sizeof(CUDAGenome));

    // Evaluate.
    evaluate();

    // Select.
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
    // TODO Implement so that threads of the same block select the same genome.
    return individuals[0];
}
