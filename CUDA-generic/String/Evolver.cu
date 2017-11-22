#include "Evolver.h"
#include <curand.h>
#include <curand_kernel.h>


template <typename T>
__global__ void evolve(T *pop, unsigned int genNum) {
    curandState_t singleState;
    curandState_t coupleState;
    extern __shared__ T family[];
    T *tmpFamily = &family[blockDim.x];
    int *fitters = (int *) &tmpFamily[blockDim.x];

    // Initialize the inter-block random number generator.
    // Different threads get different randoms.
    curand_init((unsigned long) clock(), blockIdx.x, threadIdx.x, &singleState);

    // Initialize the intra-block random number generator.
    // Pairs of threads get the same randoms.
    curand_init((unsigned long) clock(), blockIdx.x, threadIdx.x % (blockDim.x / 2), &coupleState);

    // Copy the family to shared memory.
    family[threadIdx.x] = pop[IDX(blockIdx.x, threadIdx.x, blockDim.x)];

    for (unsigned int g = 0; g < genNum; g++) {
        // Evaluation.
        evaluate(family);
        __syncthreads();

        // Selection.
        select(family, tmpFamily, fitters, &singleState);
        __syncthreads();

        // Crossover.
        if (curand_uniform(&coupleState) < CROSS_PROB) {
            crossover(family, tmpFamily, fitters, &singleState);
        }
        __syncthreads();

        // Mutation.
        mutate(family, tmpFamily, &singleState);
        __syncthreads();
    }

    // Sort the family.
    sort(family, tmpFamily);
    __syncthreads();

    // Copy the family back to global memory.
    pop[IDX(blockIdx.x, threadIdx.x, blockDim.x)] = family[threadIdx.x];
    __syncthreads();
}
