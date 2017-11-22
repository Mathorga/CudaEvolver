#ifndef __INDIVIDUAL__
#define __INDIVIDUAL__

#include "../Evolver.h"

#define STRING_LENGTH 4
#define OBJECTIVE_STRING "luka"

struct Individual {
    float score = 0.0;
    char string[STRING_LENGTH];
};

void initialize(Individual *pop, unsigned int popSize) {
    for (int i = 0; i < popSize; i++) {
        for (int j = 0; i < STRING_LENGTH; i++) {
            pop[i].string[j] = 'a' + (rand() % 26);
        }
    }
}

template <typename T>
__device__ void evaluate(T *family) {
    family[threadIdx.x].score = 0;
    for (int i = 0; i < STRING_LENGTH; i++) {
        if (family[threadIdx.x].string[i] == OBJECTIVE_STRING[i]) {
            family[threadIdx.x].score++;
        }
    }
}

template <typename T>
__device__ void select(T *family, T *tmpFamily, int *fitters, curandState_t *state) {
    int random = (int) (curand_uniform(state) * blockDim.x);
    fitters[threadIdx.x] = (family[threadIdx.x].score > family[random].score) ? threadIdx.x : random;
    tmpFamily[threadIdx.x] = family[threadIdx.x];
}

template <typename T>
__device__ void crossover(T *family, T *tmpFamily, int *fitters, curandState_t *state) {
    T parent1 = family[fitters[threadIdx.x]];
    T parent2 = family[fitters[(threadIdx.x + 1) % blockDim.x]];
    int midPoint = curand_uniform(state) * STRING_LENGTH;
    // Pick from parent 1.
    for (int i = 0; i <= midPoint; i++) {
        tmpFamily[threadIdx.x].string[i] = parent1.string[i];
    }

    // Pick from parent 2.
    for (int i = midPoint + 1; i < STRING_LENGTH; i++) {
        tmpFamily[threadIdx.x].string[i] = parent2.string[i];
    }
    family[threadIdx.x] = tmpFamily[threadIdx.x];
}

template <typename T>
__device__ void mutate(T *family, T *tmpFamily, curandState_t *state, float mutProb) {
    for (int i = 0; i < STRING_LENGTH; i++) {
        if (curand_uniform(state) <= mutProb) {
            family[threadIdx.x].string[i] = 'a' + (curand_uniform(state) * 26);
        }
    }
}

template <typename T>
__device__ void sort(T *family, T *tmpFamily) {
    for (int i = 0; i < blockDim.x / 2; i++) {
        // Even phase.
        if (!(threadIdx.x & 1) && threadIdx.x < gridDim.x - 1) {
            if (family[threadIdx.x].score < family[threadIdx.x + 1].score) {
                // Swap.
                tmpFamily[threadIdx.x] = family[threadIdx.x];
                family[threadIdx.x] = family[threadIdx.x + 1];
                family[threadIdx.x + 1] = tmpFamily[threadIdx.x];
            }
        }
        __syncthreads();

        // Odd phase.
        if ((threadIdx.x & 1) && threadIdx.x < gridDim.x * blockDim.x - 1) {
            if (family[threadIdx.x].score < family[threadIdx.x + 1].score) {
                // Swap.
                tmpFamily[threadIdx.x] = family[threadIdx.x];
                family[threadIdx.x] = family[threadIdx.x + 1];
                family[threadIdx.x + 1] = tmpFamily[threadIdx.x];
            }
        }
        __syncthreads();
    }
}

#endif
