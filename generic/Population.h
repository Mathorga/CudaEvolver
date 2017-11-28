/*
**************************************************************************
Population.h

This file implements the population class, responsible for common genetic
operations.
Copyright (C) 2017  Luka Micheletti

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>
**************************************************************************
*/

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
