#include "Population.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void Population::initialize() {
    for (unsigned int i = 0; i < size; i++) {
        individuals[i]->initialize();
    }
}

void Population::evaluate() {
    for (unsigned int i = 0; i < size; i++) {
        individuals[i]->evaluate();
    }
}

void Population::sort() {
    quickSort(0, size - 1);
}



Population::Population(unsigned int popSize, unsigned int genNum, float mutationRate, Genome *original) {
    genNumber = genNum;
    mutRate = mutationRate;
    currentGen = 0;
    initialized = false;
    size = popSize;
    individuals = new Genome *[size];
    offspring = new Genome *[size];
    for (unsigned int i = 0; i < size; i++) {
        individuals[i] = original->clone();
        offspring[i] = original->clone();
    }
}

void Population::step() {
    // Evaluate.
    evaluate();

    // Sort.
    sort();

    // Scale.
    scale();

    // Select and crossover.
    for (unsigned int i = 0; i < size; i++) {
        Genome *parent1 = select();
        Genome *parent2 = select();

        parent1->crossover(parent2, &(offspring[i]));
    }

    // Mutate.
    for (unsigned int i = 0; i < size; i++) {
        offspring[i]->mutate(mutRate);
    }

    // Keep the best individual from the old population if better than the best of the current.
    if (individuals[0]->getFitness() > offspring[0]->getFitness()) {
        offspring[0] = individuals[0];
    }

    // Overwrite the old individuals with the new ones.
    for (unsigned int i = 0; i < size; i++) {
        individuals[i] = offspring[i];
        offspring[i] = individuals[i]->clone();
    }
}

Genome *Population::select() {
    float totalFitness = 0.0;
    float previousProb = 0.0;

    // Generate a random number.
    float random = ((float) rand()) / (RAND_MAX + 1.0);
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
    for (unsigned int i = 0; i < size; i++) {
        individuals[i]->scale(individuals[size - 1]->getScore());
    }
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
        for (;;) {
            while (individuals[++i]->getScore() < score && i <= right);
            while (individuals[--j]->getScore() > score && j > 0);
            if (i >= j) {
                break;
            }
            g = individuals[i];
            individuals[i] = individuals[j];
            individuals[j] = g;
        }
        g = individuals[i];
        individuals[i] = individuals[right];
        individuals[right] = g;
        quickSort(left, i - 1);
        quickSort(i + 1, right);
        // printf("All good so far\n");
    }
}
