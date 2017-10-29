#include "CUDAPathGenome.h"
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>

__global__ void createCUDAPathGenome(CUDAGenome **genome, CUDAPathGenome::_Point2D *checks, unsigned int checksNum) {
    if (threadIdx.x == 0) {
        *genome = new CUDAPathGenome(checks, checksNum);
    }
}

__device__ void CUDAPathGenome::initialize() {
    if (threadIdx.x == 0) {
        // printf("Initializing\n");
        curandState_t state;
        curand_init((unsigned long) clock(), blockIdx.x, threadIdx.x, &state);

        // Create a copy of the checks array.
        _Point2D *checksCopy = (_Point2D *) malloc(checksNumber * sizeof(_Point2D));
        for (unsigned int i = 0; i < checksNumber; i++) {
            checksCopy[i] = checks[i];
            // printf("x:%d\ty:%d\n", checks[i].x, checks[i].y);
        }
        // printf("Created a copy of checks\n");

        // Randomly initialize path;
        for (unsigned int i = 0; i < checksNumber; i++) {
            int index = curand(&state) % (checksNumber - i);
            path[i] = checksCopy[index];
            for (unsigned int j = index; j < checksNumber - i; j++) {
                checksCopy[j] = checksCopy[j + 1];
            }
        }
        // printf("Initialized path\n");
    }
}

__device__ void CUDAPathGenome::evaluate() {
    // printf("Evaluating individual %d\n", blockIdx.x);
    __shared__ float *tmpDists;
    tmpDists = (float *) malloc(checksNumber * sizeof(float));

    int bSize = blockDim.x / 2;

    // Calculate distances between each check.
    float dx = (float) path[(threadIdx.x + 1) % checksNumber].x - (float) path[threadIdx.x].x;
    float dy = (float) path[(threadIdx.x + 1) % checksNumber].y - (float) path[threadIdx.x].y;
    tmpDists[threadIdx.x] = sqrtf(powf(dx, 2) + powf(dy, 2));
    __syncthreads();

    // Perform reduction to compute the sum of the distances.
    while (bSize > 0) {
        if (threadIdx.x < bSize) {
            tmpDists[threadIdx.x] += tmpDists[threadIdx.x + bSize];
        }
        bSize /= 2;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        score = tmpDists[0];
        // printf("Individual %d score: %f\n", blockIdx.x, score);
    }
}

__device__ void CUDAPathGenome::crossover(CUDAGenome *partner, CUDAGenome **offspring) {
    CUDAPathGenome *child = (CUDAPathGenome *) (*offspring);
    CUDAPathGenome *mate = (CUDAPathGenome *) partner;

    // _Point2D *tmpPath = (_Point2D *) malloc(checksNumber * sizeof(_Point2D));
    unsigned int midPoint = 0;

    curandState_t state;
    curand_init((unsigned long) clock(), 0, 0, &state);
    midPoint = curand(&state) % (checksNumber - 1);

    // Pick from parent 1.
    if (threadIdx.x <= midPoint) {
        child->path[threadIdx.x] = path[threadIdx.x];
    }
    __syncthreads();

    // Pick from parent 2.
    if (threadIdx.x == 0) {
        for (unsigned int i = midPoint + 1; i < checksNumber; ) {
            for (unsigned int j = 0; j < checksNumber; j++) {
                bool insert = true;
                for (unsigned int k = 0; k <= midPoint; k++) {
                    if (mate->path[j].id == child->path[k].id) {
                        insert = false;
                        break;
                    }
                }
                if (insert) {
                    child->path[i] = mate->path[j];
                    i++;
                }
            }
        }
    }
    __syncthreads();
}

__device__ void CUDAPathGenome::mutate() {
    // TODO.
}

__device__ void CUDAPathGenome::scale(float baseScore) {
    if (threadIdx.x == 0) {
        fitness = (baseScore - score) + 1;
    }
}

__host__ __device__ CUDAGenome *CUDAPathGenome::clone() {
    return new CUDAPathGenome(checks, checksNumber);
}

__host__ __device__ void CUDAPathGenome::print() {
    #ifdef __CUDA_ARCH__

    if (threadIdx.x == 0) {
        for (unsigned int i = 0; i < checksNumber; i++) {
            printf("x:%u\ty:%u\tid:%d\n", path[i].x, path[i].y, path[i].id);
        }
    }

    #else

    for (unsigned int i = 0; i < checksNumber; i++) {
        printf("x:%u\ty:%u\tid:%d\n", path[i].x, path[i].y, path[i].id);
    }

    #endif
};

__host__ __device__ virtual void output(char *string) {
    for (int i = 0; i < COORD_DIGITS; i++) {
        string[threadIdx.x * POINT_DIGITS + i] = itoa(path[threadIdx.x].x / powf(10, COORD_DIGITS - i));
    }
}


__host__ __device__ CUDAPathGenome::CUDAPathGenome(_Point2D *checkArray, unsigned int checksNum, unsigned int fieldSize) : CUDAGenome(checksNum) {
    checksNumber = checksNum;
    checks = (_Point2D *) malloc(checksNum * sizeof(_Point2D));
    path = (_Point2D *) malloc(checksNum * sizeof(_Point2D));
    distances = (float *) malloc(checksNum * sizeof(float));
    for (unsigned int i = 0; i < checksNum; i++) {
        checks[i] = checkArray[i];
        _Point2D newCheck;
        path[i] = newCheck;
        // printf("Passed Checks\n");
        // printf("x:%d\ty:%d\n", checkArray[i].x, checkArray[i].y);
        distances[i] = 0.0;
    }
    // print();
}
