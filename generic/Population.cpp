#include "Population.h"
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>

void Population::evaluate() {
    for (unsigned int i = 0; i < size; i++) {
        individuals[i]->evaluate();
    }
}

void Population::sort() {
    quickSort(0, size - 1);
}


__global__ void outputBest(Population *pop, char *string) {
    if (blockIdx.x == 0) {
        // Output the last (best) individual.
        pop->individuals[pop->getSize() - 1]->output(string);
    }
}

__global__ void outputWorst(Population *pop, char *string) {
    if (blockIdx.x == 0) {
        // Output the first (worst) individual.
        pop->individuals[0]->output(string);
    }
}



Population::Population(unsigned int popSize, unsigned int genNum, Objective obj) {
    genNumber = genNum;
    currentGen = 0;
    initialized = false;
    size = popSize;
    individuals = (Genome **) malloc(size * sizeof(Genome *));
    offspring = (Genome **) malloc(size * sizeof(Genome *));
}

void Population::step() {
    scale();
    // Select.
    // printf("Selection\n");
    Genome *partner = select();
    __syncthreads();

    // Crossover.
    // printf("Crossover\n");
    individuals[blockIdx.x]->crossover(partner, &(offspring[blockIdx.x]));
    __syncthreads();

    // Mutate.
    // printf("Mutation\n");
    offspring[blockIdx.x]->mutate();
    __syncthreads();

    // Overwrite the old individual with the new one.
    if (threadIdx.x == 0) {
        individuals[blockIdx.x] = offspring[blockIdx.x];
        offspring[blockIdx.x] = individuals[blockIdx.x]->clone();
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // TODO Copy the best from the old pop to the new one.
    }
    __syncthreads();
}

Genome *Population::select() {
    float totalFitness = 0.0;
    float previousProb = 0.0;

    // Threads of the same block select the same genome by generating the same pseudo-random number.
    curandState_t state;
    curand_init((unsigned long) clock(), blockIdx.x, 0, &state);
    float random = curand_uniform(&state);

    // Calculate the total fitness.
    for (unsigned int i = 0; i < size; i++) {
        totalFitness += individuals[i]->getFitness();
    }

    // Calculate the probability for each individual.
    for (unsigned int i = 0; i < size - 1; i++) {
        float prob = previousProb + (individuals[i]->getFitness() / totalFitness);
        if (random < prob) {
            return individuals[i];
        } else {
            previousProb += prob;
        }
    }
    return individuals[size - 1];
}

void Population::scale() {
    individuals[blockIdx.x]->scale(individuals[size - 1]->getScore());
}

void Population::quickSort(int left, int right) {
    int i;
    int j;
    float score;
    Genome *g;

    if (right > left) {
        score = individuals[right]->getScore();
        i = left - 1;
        j = right;
    }
    for (;;) {
        while (individuals[i]->getScore() < score && i <= right) {
            i++;
        }
        while (individuals[i]->getScore() > score && j <= left) {
            j--;
        }
        if (i >= j) {
            break;
        }
        g = individuals[i];
        individuals[i] = individuals[right];
        individuals[right] = g;
        quickSort(left, i - 1);
        quickSort(i + i, right);
    }
}
