#include "CUDAPathGenome.h"
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>

void CUDAPathGenome::initialize() {
    printf("Initializing\n");
    srand(time(NULL));
    // Create a copy of the checks array.
    _2DDot *checksCopy = (_2DDot *) malloc(checksNumber * sizeof(_2DDot));
    for (unsigned int i = 0; i < checksNumber; i++) {
        checksCopy[i] = checks[i];
    }
    printf("Created a copy of checks\n");

    // Randomly initialize path;
    for (unsigned int i = 0; i < checksNumber; i++) {
        int index = rand() % (checksNumber - i);
        path[i] = checksCopy[index];
        for (unsigned int j = index; j < checksNumber - i; j++) {
            checksCopy[j] = checksCopy[j + 1];
        }
    }
    printf("Initialized path on the host\n");

    // Copy the initialized path on the device copy.
    // cudaMalloc(&d_checks, checksNumber * sizeof(_2DDot));
    // cudaMalloc(&d_path, checksNumber * sizeof(_2DDot));
    // cudaMalloc(&d_distances, checksNumber * sizeof(float));
    // cudaMemcpy(d_checks, checks, checksNumber * sizeof(_2DDot), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_path, path, checksNumber * sizeof(_2DDot), cudaMemcpyHostToDevice);
    // printf("Copied path on the device\n");
}

__device__ void CUDAPathGenome::evaluate() {
    printf("Evaluating individual %d\n", blockIdx.x);
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
        printf("Individual %d score: %f\n", blockIdx.x, score);
    }
}

__device__ void CUDAPathGenome::crossover(CUDAGenome *partner, CUDAGenome *offspring) {
    CUDAPathGenome *child = (CUDAPathGenome *) offspring;
    CUDAPathGenome *mate = (CUDAPathGenome *) partner;
    _2DDot *tmpPath = (_2DDot *) malloc(checksNumber * sizeof(_2DDot));
    unsigned int midPoint = 0;

    if (threadIdx.x == 0) {
        curandState_t state;
        curand_init((unsigned long) clock(), blockIdx.x, 0, &state);
        midPoint = curand(&state) % (checksNumber - 1);
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
    }
    __syncthreads();
    child->setPath(tmpPath);
}

__device__ void CUDAPathGenome::mutate() {
    // TODO.
}

__device__ void CUDAPathGenome::scale(float baseScore) {
    if (threadIdx.x == 0) {
        fitness = (baseScore - score) + 1;
    }
}

CUDAGenome *CUDAPathGenome::clone() {
    return new CUDAPathGenome(checks, checksNumber);
}

void CUDAPathGenome::allocateCopySingle(CUDAGenome **deviceIndividual, CUDAGenome **hostIndividual, cudaMemcpyKind direction) {
    cudaMalloc(deviceIndividual, sizeof(CUDAPathGenome));
    cudaMemcpy(*deviceIndividual, *hostIndividual, sizeof(CUDAPathGenome), direction);

    cudaMalloc(((CUDAPathGenome *) (*deviceIndividual))->getDeviceChecksAddress(), (*deviceIndividual)->getXSize() * sizeof(_2DDot));
    cudaMalloc(((CUDAPathGenome *) (*deviceIndividual))->getDevicePathAddress(), (*deviceIndividual)->getXSize() * sizeof(_2DDot));
    // cudaMalloc(((CUDAPathGenome *) (*deviceIndividual))->getDeviceDistancesAddress(), (*deviceIndividual)->getXSize() * sizeof(float));
    cudaMemcpy(((CUDAPathGenome *) (*deviceIndividual))->getDeviceChecks(),
               ((CUDAPathGenome *) (*hostIndividual))->getHostChecks(),
               (*hostIndividual)->getXSize() * sizeof(_2DDot),
               cudaMemcpyHostToDevice);
    cudaMemcpy(((CUDAPathGenome *) (*deviceIndividual))->getDevicePath(),
               ((CUDAPathGenome *) (*hostIndividual))->getHostPath(),
               (*hostIndividual)->getXSize() * sizeof(_2DDot),
               cudaMemcpyHostToDevice);
}

void CUDAPathGenome::allocateCopyMultiple(CUDAGenome ***deviceIndividuals, CUDAGenome ***hostIndividuals, unsigned int count, cudaMemcpyKind direction) {
    cudaMalloc(deviceIndividuals, count * sizeof(CUDAPathGenome *));
    cudaMemcpy(*deviceIndividuals, *hostIndividuals, count * sizeof(CUDAPathGenome *), direction);
}

CUDAPathGenome::CUDAPathGenome(_2DDot *checkArray, unsigned int checksNum) : CUDAGenome(checksNum) {
    checksNumber = checksNum;
    checks = (_2DDot *) malloc(checksNum * sizeof(_2DDot));
    path = (_2DDot *) malloc(checksNum * sizeof(_2DDot));
    distances = (float *) malloc(checksNum * sizeof(float));
    for (unsigned int i = 0; i < checksNum; i++) {
        checks[i] = checkArray[i];
        path[i] = checkArray[i];
        distances[i] = 0.0;
    }

    // cudaMalloc(&d_checks, checksNum * sizeof(_2DDot));
    // cudaMalloc(&d_path, checksNum * sizeof(_2DDot));
    // cudaMalloc(&d_distances, checksNum * sizeof(float));
    // cudaMemcpy(d_checks, checkArray, checksNum * sizeof(_2DDot), cudaMemcpyHostToDevice);
}
