#ifndef __CUDA_PATH_GENOME_UTILS__
#define __CUDA_PATH_GENOME_UTILS__

#include "CUDAPopulation.h"
#include "CUDAPathGenome.h"

__global__ void createCUDAPathGenomePopulation(CUDAPopulation *pop, CUDAPathGenome::_Point2D *checks, unsigned int checksNum) {
    if (threadIdx.x == 0) {
        CUDAPathGenome *genome = new CUDAPathGenome(checks, checksNum);
        CUDAPathGenome *child = new CUDAPathGenome(checks, checksNum);

        genome->initialize();

        pop->individuals[blockIdx.x] = genome;
        pop->offspring[blockIdx.x] = child;
        pop->tmp = genome->clone();
    }
}

#endif
