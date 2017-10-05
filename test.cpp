#include <stdio.h>
#include <stdlib.h>
#include <ga/GASimpleGA.h>
#include <ga/GA1DBinStrGenome.h>

float Objective(GAGenome& g) {
    GA1DBinaryStringGenome & genome = (GA1DBinaryStringGenome &)g;
    float score=0.0;
    for (int i = 0; i < genome.length(); i++) {
        score += genome.gene(i);
    }
    return score;
}

int main(int argc, char const *argv[]) {
    // Create a genome.
    GA1DBinaryStringGenome genome(100, Objective);
    // Create the genetic algorithm.
    GASimpleGA ga(genome);
    ga.populationSize(100);
    ga.nGenerations(1000);
    // Do the evolution.
    ga.evolve();
    std::cout << ga.statistics() << std::endl;
    return 0;
}
