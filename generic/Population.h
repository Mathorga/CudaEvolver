#ifndef __POPULATION__
#define __POPULATION__

#include "Genome.h"
#include "PathGenome.h"

class Population {
public:
    Genome **individuals;
    Genome **offspring;
    Genome *tmp;

    Population(unsigned int popSize, unsigned int genNum, Objective obj = MAXIMIZE);
    void evaluate();
    void sort();
    // Scales the individuals' scores to fitnesses.
    void scale();
    void step();

    Genome *best();
    Genome *worst();

    unsigned int getSize() {
        return size;
    }
    unsigned int getGenNumber() {
        return genNumber;
    }
private:
    // Performs fitness-proportionate selection to get an individual from the population.
    Genome *select();
    void quickSort();

private:
    bool initialized;
    unsigned int size;
    unsigned int genNumber;
    unsigned int currentGen;
};

#endif
