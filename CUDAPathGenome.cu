#include "CUDAPathGenome.h"
#include <curand.h>
#include <curand_kernel.h>

void CUDAPathGenome::initialize() {
    // TODO.
}

__device__ void CUDAPathGenome::evaluate() {
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
    }
}

__device__ CUDAGenome *CUDAPathGenome::crossover(CUDAGenome *partner, CUDAGenome *offspring) {
    CUDAPathGenome *child = (CUDAPathGenome *) offspring;
    CUDAPathGenome *mate = (CUDAPathGenome *) partner;
    __shared__ _2DDot *tmpPath;
    tmpPath = (_2DDot *) malloc(checksNumber * sizeof(_2DDot));
    unsigned int midPoint = 0;

    if (threadIdx.x == 0) {
        midPoint = GARandomInt(0, checksNumber - 1);
    }
    __syncthreads();

    // Pick from parent 1.
    if (threadIdx.x <= midPoint) {
        tmpPath[threadIdx.x] = getCheck(threadIdx.x);
    }
    __syncthreads();

    // Pick from parent 2.
    if (threadIdx.x == 0) {
        for (unsigned int i = midPoint + 1; i < checksNumber; ) {
            for (unsigned int j = 0; j < checksNumber; j++) {
                bool insert = true;
                for (unsigned int k = 0; k <= midPoint; k++) {
                    if (mate->getCheck(j).id == tmpPath[k].id) {
                        insert = false;
                        break;
                    }
                }
                if (insert) {
                    tmpPath[i] = mate->getCheck(j);
                    i++;
                }
            }
        }
        child = new CUDAPathGenome(checksNumber);
    }
    __syncthreads();
    child



    return child;
}

__device__ void CUDAPathGenome::mutate() {
    // TODO.
}

__device__ void CUDAPathGenome::scale(float baseScore) {
    if (threadIdx == 0) {
        fitness = (baseScore - score) + 1;
    }
}

CUDAGenome *CUDAPathGenome::clone() {
    return new CUDAPathGenome(checksNumber);
}

CUDAPathGenome::CUDAPathGenome(unsigned int checksNum) : CUDAGenome(checksNum) {
    checksNumber = checksNum;
    cudaMalloc(&checks, checksNum * sizeof(CUDAPathGenome::_2DDot));
    cudaMalloc(&path, checksNum * sizeof(CUDAPathGenome::_2DDot));
    cudaMemset(checks, 0, checksNum * sizeof(CUDAPathGenome::_2DDot));
    cudaMemset(path, 0, checksNum * sizeof(CUDAPathGenome::_2DDot));
}
