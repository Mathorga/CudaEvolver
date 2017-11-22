#ifndef __EVOLVER__
#define __EVOLVER__

#include "Individual.h"

// Translates bidimensional indexes to a monodimensional one.
// |i| is the column index.
// |j| is the row index.
// |n| is the number of columns (length of the rows).
#define IDX(i, j, n) ((i) * (n) + (j))

__global__ void evolve(Individual *pop, unsigned int genNum);

#endif
