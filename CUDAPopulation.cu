#include "CUDAPopulation.h"

CUDAPopulation::CUDAPopulation(unsigned int popSize, unsigned int genNum, CUDAGenome &genome, int objective = MAXIMIZE) {
    genNumber = genNum;
    currentGen = 0;
    initialized = false;
    size = popSize;
    individuals = (CUDAGenome **) malloc(size * sizeof(CUDAGenome *));
    cudaMalloc(&d_individuals, size * sizeof(CUDAGenome *));
    cudaMemcpy(d_individuals, individuals, size * sizeof(CUDAGenome *), cudaMemcpyHostToDevice);
    for (unsigned int i = 0; i < size; i++) {
        individuals[i] = genome.clone();
        cudaMalloc(&d_individuals[i], sizeof(CUDAGenome));
    }
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

void CUDAPopulation::evolve() {
    initialize();
    dim3 gridSize(size);
    dim3 blockSize(individuals[0]->xSize(), individuals[0]->ySize(), individuals[0]->zSize());
    for (unsigned int i = 0; i < genNumber; i++) {
        step<<<gridSize, blockSize>>>();
    }
}

__global__ void CUDAPopulation::step() {
    // Evaluate.
    evaluate();
    for () {
        // Select.
        CUDAGenome *parent1 = select();
        CUDAGenome *parent2 = select();
        // Crossover.
        CUDAGenome *child = crossover(parent1, parent2);
        // Mutate.
        child->mutate();
        // Add the child to the new pop.
        // TODO.
    }
    // Copy the best from the old pop to the new one.
    // TODO.
}

__device__ void CUDAPopulation::evaluate() {
    d_individuals[blockIdx.x]->evaluate();
}

__device__ CUDAGenome *CUDAPopulation::select() {
    // TODO.
}

__device__ CUDAGenome *CUDAPopulation::crossover() {
    // TODO.
}
