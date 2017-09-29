#ifndef __POPULATION__
#define __POPULATION__

#include <list>
#include "Individual.h"

using namespace std;

class Population {
public:
    // Constructor.
    Population(){};
    void insertIndividual(Individual* individual);
    void select();

private:
    std::list<Individual> individuals;

    int calculateFitness(Individual individual);
};

#endif
