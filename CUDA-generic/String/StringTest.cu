/*
**************************************************************************
StringTest.cu

This file implements an example of resolution to the objective string
problem using CUDA genetic algorithm.
Copyright (C) 2017  Luka Micheletti

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>
**************************************************************************
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "../../hpc.h"
#include "Individual.h"

#define BLOCK_SIZE 32

#define cudaCheckError() {                                                                                  \
            cudaError_t e = cudaGetLastError();                                                             \
            if (e != cudaSuccess) {                                                                         \
                printf("Cuda failure %s(%d): %d(%s)\n", __FILE__, __LINE__ - 1, e, cudaGetErrorString(e));  \
                exit(0);                                                                                    \
            }                                                                                               \
        }




int main(int argc, char const *argv[]) {
    Individual *population;
    Individual *d_population;

    unsigned int popSize = 4096;
    unsigned int famNumber = 32;
    unsigned int genNumber = 1000;
    float crossProb = 0.5;
    float mutProb = 0.001;
    double startTime = 0.0;
    double endTime = 0.0;

    if (argc > 5) {
        printf("Usage: %s [popSize [genNumber [crossProb [mutProb]]]\n", argv[0]);
        return -1;
    }
    if (argc > 1) {
        popSize = atoi(argv[3]);
    }
    if (argc > 2) {
        genNumber = atoi(argv[4]);
    }
    if (argc > 3) {
        crossProb = atof(argv[5]);
    }
    if (argc > 4) {
        mutProb = atof(argv[6]);
    }


    const size_t size = popSize * sizeof(Individual);

    dim3 members(popSize / famNumber);
    dim3 families(famNumber);
    size_t familySize = members.x * sizeof(Individual);
    size_t intArraySize = members.x * sizeof(int);
    size_t sharedMemSize = 2 * familySize + intArraySize;
    printf("total shared mem / block:\t%zuB\n", sharedMemSize);
    printf("family size:\t\t\t%zuB\n", familySize);
    printf("int array size:\t\t\t%zuB\n", intArraySize);
    srand(time(0));

    // Create the host population.
    population = (Individual *) malloc(size);

    // Create the device populations.
    cudaMalloc(&d_population, size);

    // Initialize the population.
    initialize(population, popSize);

    // Copy the host population to the device.
    cudaMemcpy(d_population, population, size, cudaMemcpyHostToDevice);






    // ***Execution.***
    printf("Execution:\n");
    startTime = hpc_gettime();

    evolve<<<families, members, sharedMemSize>>>(d_population, genNumber, crossProb, mutProb);
    cudaDeviceSynchronize();

    endTime = hpc_gettime();
    printf("Execution time: %fs\n\n", endTime - startTime);





    // Copy the device population back to the host.
    cudaMemcpy(population, d_population, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < famNumber; i++) {
        printf("Best of family %d: ", i);
        printf("%f", population[i * members.x].score);
        printf("\t%s\n", population[i * members.x].string);
    }




    free(population);
    cudaFree(d_population);
    return 0;
}
