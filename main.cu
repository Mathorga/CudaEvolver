#include <stdio.h>
#include <stdlib.h>
#include "Chromosome.h"

__global__ void print() {
    printf("Hello, I am thread %d of block %d\n", threadIdx.x, blockIdx.x);
}

int main(int argc, char const *argv[]) {
    GaChromosome dna = new GaChromosome();
    // Create the initial population.

    // Calculate the fitness of the population.

    //
    print<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}
