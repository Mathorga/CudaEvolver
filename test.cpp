#include <stdio.h>
#include <stdlib.h>
#include <ga/ga.h>

#define POP_SIZE 100

float objective(GAGenome &g) {
    GA1DArrayGenome<int> &genome = (GA1DArrayGenome<int> &)g;

    float score=0.0;
    for (int i = 0; i < genome.length(); i++) {
        score += genome.gene(i);
    }
    return score;
}

void randomInitializer(GAGenome& g) {
    GA1DArrayGenome<int> &genome=(GA1DArrayGenome<int> &)g;

    for (int i = 0; i < genome.size(); i++) {
        genome.gene(i, GARandomInt(0, 100));
    }
}

int main(int argc, char const *argv[]) {
    // Create a genome.
    GA1DArrayGenome<int> genome(10, objective);
    genome.initializer(randomInitializer);

    // Create a population.
    GAPopulation population(genome, POP_SIZE);

    for (int i = 0; i < population.size(); i++) {
        printf("Individual %d:", i);
        GA1DArrayGenome<int>& individual = (GA1DArrayGenome<int>&)population.individual(i);
        for (int j = 0; j < individual.size(); j++) {
            printf("%d", individual.gene(j));
        }
        printf("\n");
    }

    // Create the genetic algorithm.
    GASimpleGA ga(population);
    ga.nGenerations(1000);

    // Do the evolution.
    ga.evolve();
    std::cout << ga.statistics() << std::endl;
    return 0;
}
