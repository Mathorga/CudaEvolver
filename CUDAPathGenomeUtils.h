#ifndef __CUDA_PATH_GENOME_UTILS__
#define __CUDA_PATH_GENOME_UTILS__

#include "CUDAPopulation.h"
#include "CUDAPathGenome.h"

__global__ void createCUDAPathGenomePopulation(CUDAPopulation *pop, CUDAPathGenome::_Point2D *checks, unsigned int checksNum) {
    if (threadIdx.x == 0) {
        CUDAPathGenome *genome = new CUDAPathGenome(checks, checksNum);

        genome->initialize();

        // printf("\n");
        // for (unsigned int i = 0; i < checksNum; i++) {
        //     printf("x:%u\ty:%u\n", genome->path[i].x, genome->path[i].y);
        // }

        pop->individuals[blockIdx.x] = genome;

        // printf("\n");
        // for (unsigned int i = 0; i < checksNum; i++) {
        //     printf("x:%u\ty:%u\n", ((CUDAPathGenome *) pop->individuals[blockIdx.x])->path[i].x, ((CUDAPathGenome *) pop->individuals[blockIdx.x])->path[i].y);
        // }

        pop->offspring[blockIdx.x] = genome;
    }
}

#endif
