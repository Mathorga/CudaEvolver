#include <stdio.h>
#include <stdlib.h>
#include "Population.h"

__global__ void print() {
    printf("Hello, I am thread %d of block %d\n", threadIdx.x, blockIdx.x);
}

int main(int argc, char const *argv[]) {
    // Create the initial population.
    Population* pool = new Population();
    //for (int i = 0; i < 100; i++) {
    //}
    // Calculate the fitness of the population.

    print<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}
