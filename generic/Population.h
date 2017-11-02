#ifndef __POPULATION__
#define __POPULATION__

#include "Genome.h"
#include "PathGenome.h"

class Population {
public:
    Genome **individuals;
    Genome **offspring;
    Genome *tmp;

    Population(unsigned int popSize, unsigned int genNum, float mutationRate, Genome *original);
    void initialize();
    void evaluate();
    void sort();
    // Scales the individuals' scores to fitnesses.
    void scale();
    void step();
    // Performs fitness-proportionate selection to get an individual from the population.
    Genome *select();

    Genome *best() {
        return individuals[0];
    }
    Genome *worst() {
        return individuals[size - 1];
    }

    unsigned int getSize() {
        return size;
    }
    unsigned int getGenNumber() {
        return genNumber;
    }
private:
    void quickSort(int left, int right);

private:
    bool initialized;
    unsigned int size;
    unsigned int genNumber;
    float mutRate;
    unsigned int currentGen;
};

#endif
