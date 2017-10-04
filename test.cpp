#include <stdio.h>
#include <stdlib.h>
#include <ga/GASimpleGA.h>
#include <ga/GA1DBinStrGenome.h>

float Objective(GAGenome& g) {
    return 0.5;
}

int main(int argc, char const *argv[]) {
    GA1DBinaryStringGenome genome(100, Objective);
    // create a genome
    GASimpleGA ga(genome);
    // create the genetic algorithm
    ga.evolve();
    // do the evolution
    std::cout << ga.statistics() << std::endl;
    return 0;
}
