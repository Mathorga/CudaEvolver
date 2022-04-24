/*
**************************************************************************
Test.cu

This file implements an example of resolution to the TSP
problem using specific CUDA genetic algorithm.
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

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include "../hpc.h"

#define BLOCK_SIZE 32
#define CHECKS_NUM 15
#define MUTATION_PROB 0.001
#define CROSS_PROB 0.5

#define cudaCheckError() {                                                                                  \
            cudaError_t e = cudaGetLastError();                                                             \
            if (e != cudaSuccess) {                                                                         \
                printf("Cuda failure %s(%d): %d(%s)\n", __FILE__, __LINE__ - 1, e, cudaGetErrorString(e));  \
                exit(0);                                                                                    \
            }                                                                                               \
        }

// Translates bidimensional indexes to a monodimensional one.
// |i| is the column index.
// |j| is the row index.
// |n| is the number of columns (length of the rows).
#define IDX(i, j, n) ((i) * (n) + (j))

typedef unsigned char cell_t;

enum CellContent {
    EMPTY = 0,
    CHECK = 1,
    PATH = 2
};

struct Point2D {
    short x = -1;
    short y = -1;
    // int id = -1;
};

__host__ __device__ bool operator == (const Point2D point1, const Point2D point2) {
    return (point1.x == point2.x && point1.y == point2.y);
}

struct Individual {
    Point2D path[CHECKS_NUM];
    float score = 0.0;
};

// Code taken from Rosettacode:
// https://rosettacode.org/wiki/Bitmap/Bresenham%27s_line_algorithm#C.2B.2B
// Implementing Bresenham’s line drawing algorithm.
void drawLine(cell_t *field, int n, int x0, int y0, int x1, int y1) {
    int x = x0;
    int y = y0;

    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = (dx > dy ? dx : -dy) / 2;
    int e2;

    for (;;) {
        if (!(x == x0 && y == y0) && !(x == x1 && y == y1)) {
            field[IDX(x, y, n)] = PATH;
        }
        if (x == x1 && y == y1) {
            break;
        }
        e2 = err;
        if (e2 > -dx) {
            err -= dy;
            x += sx;
        }
        if (e2 < dy) {
            err += dx;
            y += sy;
        }
    }
}

void dump(const cell_t *field, const Point2D *path, unsigned int n, unsigned int checksNum, const char *filename) {
    cell_t *fieldCopy = (cell_t *) malloc(n * n * sizeof(cell_t));
    for (unsigned int x = 0; x < n; x++) {
        for (unsigned int y = 0; y < n; y++) {
            fieldCopy[IDX(x, y, n)] = field[IDX(x, y, n)];
        }
    }

    if (path != NULL) {
        for (unsigned int i = 0; i < checksNum; i++) {
            drawLine(fieldCopy, n, path[i].x, path[i].y, path[(i + 1) % checksNum].x, path[(i + 1) % checksNum].y);
        }
    }
    FILE *out = fopen(filename, "w");
    if (out == NULL) {
        printf("Cannot create \"%s\"\n", filename);
        abort();
    }
    fprintf(out, "P6\n");
    fprintf(out, "%d %d\n", n, n);
    fprintf(out, "255\n");
    for (unsigned int x = 0; x < n; x++) {
        for (unsigned int y = 0; y < n; y++) {
            if (field[IDX((x + 1) % n, y, n)] == CHECK ||
                field[IDX(x, (y + 1) % n, n)] == CHECK ||
                field[IDX((x + 1) % n, (y + 1) % n, n)] == CHECK ||
                field[IDX((x - 1 + n) % n, y, n)] == CHECK ||
                field[IDX(x, (y - 1 + n) % n, n)] == CHECK ||
                field[IDX((x - 1 + n) % n, (y - 1 + n) % n, n)] == CHECK ||
                field[IDX((x + 1) %n, (y - 1 + n) % n, n)] == CHECK ||
                field[IDX((x - 1 + n) %n, (y + 1) % n, n)] == CHECK) {
                fprintf(out, "%c%c%c", 255, 30, 30);
            } else if (field[IDX(x, y, n)] == CHECK) {
                fprintf(out, "%c%c%c", 20, 20, 0);
            } else if (fieldCopy[IDX(x, y, n)] == EMPTY) {
                fprintf(out, "%c%c%c", 20, 20, 20);
            } else if (fieldCopy[IDX(x, y, n)] == PATH) {
                fprintf(out, "%c%c%c", 250, 175, 53);
            } else {
                printf("Unknown cell state (%d) of cell %d-%d", fieldCopy[IDX(x, y, n)], x, y);
                abort();
            }
        }
    }
    fclose(out);
}












void initialize(Individual *pop, unsigned int popSize, Point2D *checks, unsigned int checksNum) {
    Point2D *checksCopy = (Point2D *) malloc(checksNum * sizeof(Point2D));
    for (unsigned int i = 0; i < popSize; i++) {
        for (unsigned int j = 0; j < checksNum; j++) {
            checksCopy[j] = checks[j];
        }
        for (unsigned int j = 0; j < checksNum; j++) {
            int index = rand() % (checksNum - j);
            pop[i].path[j] = checksCopy[index];
            for (unsigned int k = index; k < checksNum - 1; k++) {
                checksCopy[k] = checksCopy[k + 1];
            }
        }
    }
}

// Evaluates an element by summing the distances between each check in its path.
__device__ void evaluate(Individual *family) {
    family[threadIdx.x].score = 0;
    for (unsigned int i = 0; i < CHECKS_NUM; i++) {
        family[threadIdx.x].score +=
        sqrtf(powf(fabsf(family[threadIdx.x].path[(i + 1) % CHECKS_NUM].x - family[threadIdx.x].path[i].x), 2) +
              powf(fabsf(family[threadIdx.x].path[(i + 1) % CHECKS_NUM].y - family[threadIdx.x].path[i].y), 2));
    }
}

__device__ void select(Individual *family, Individual *tmpFamily, int *fitters, curandState_t *state) {
    int random = (int) (curand_uniform(state) * blockDim.x);
    fitters[threadIdx.x] = (family[threadIdx.x].score < family[random].score) ? threadIdx.x : random;
    tmpFamily[threadIdx.x] = family[threadIdx.x];
}

__device__ void crossover(Individual *family, Individual *tmpFamily, int *fitters, curandState_t *state) {
    // Pick two of the fitters and do crossover on them.
    Individual parent1 = family[fitters[threadIdx.x]];
    Individual parent2 = family[fitters[(threadIdx.x + 1) % blockDim.x]];
    int midPoint = curand_uniform(state) * CHECKS_NUM;
    bool insert = true;

    // Pick from parent 1.
    for (int i = 0; i <= midPoint; i++) {
        tmpFamily[threadIdx.x].path[i] = parent1.path[i];
    }

    // Pick from parent 2.
    for (int i = midPoint + 1; i < CHECKS_NUM;) {
        for (int j = 0; j < CHECKS_NUM; j++) {
            insert = true;
            for (int k = 0; k <= midPoint; k++) {
                if (parent2.path[j] == tmpFamily[threadIdx.x].path[k]) {
                    insert = false;
                    break;
                }
            }
            if (insert) {
                tmpFamily[threadIdx.x].path[i] = parent2.path[j];
                i++;
            }
        }
    }
    family[threadIdx.x] = tmpFamily[threadIdx.x];
}

__device__ void lazyCrossover(Individual *family, Individual *tmpFamily, int *fitters) {
    family[threadIdx.x] = tmpFamily[fitters[threadIdx.x]];
}

// Implements mutiation by swaps of pairs of checks.
__device__ void mutate(Individual *family, Individual *tmpFamily, curandState_t *state) {
    for (int i = 0; i < CHECKS_NUM; i++) {
        if (curand_uniform(state) <= MUTATION_PROB) {
            int firstIndex = i;
            int secondIndex = curand_uniform(state) * (CHECKS_NUM - 1);
            for (int j = 0; j < CHECKS_NUM; j++) {
                tmpFamily[threadIdx.x].path[j] = family[threadIdx.x].path[j];
            }
            family[threadIdx.x].path[firstIndex] = family[threadIdx.x].path[secondIndex];
            family[threadIdx.x].path[secondIndex] = tmpFamily[threadIdx.x].path[firstIndex];
        }
    }
}

// Implements odd-even transposition sort on the elements of a family.
__device__ void sort(Individual *family, Individual *tmpFamily) {
    for (int i = 0; i < blockDim.x / 2; i++) {
        // Even phase.
        if (!(threadIdx.x & 1) && threadIdx.x < gridDim.x - 1) {
            if (family[threadIdx.x].score > family[threadIdx.x + 1].score) {
                // Swap.
                tmpFamily[threadIdx.x] = family[threadIdx.x];
                family[threadIdx.x] = family[threadIdx.x + 1];
                family[threadIdx.x + 1] = tmpFamily[threadIdx.x];
            }
        }
        __syncthreads();

        // Odd phase.
        if ((threadIdx.x & 1) && threadIdx.x < gridDim.x * blockDim.x - 1) {
            if (family[threadIdx.x].score > family[threadIdx.x + 1].score) {
                // Swap.
                tmpFamily[threadIdx.x] = family[threadIdx.x];
                family[threadIdx.x] = family[threadIdx.x + 1];
                family[threadIdx.x + 1] = tmpFamily[threadIdx.x];
            }
        }
        __syncthreads();
    }
}

__global__ void evolve(Individual *pop, unsigned int genNum) {
    curandState_t singleState;
    curandState_t coupleState;
    extern __shared__ Individual family[];
    Individual *tmpFamily = &family[blockDim.x];
    int *fitters = (int *) &tmpFamily[blockDim.x];

    // Initialize the inter-block random number generator.
    // Different threads get different randoms.
    curand_init((unsigned long) clock(), blockIdx.x, threadIdx.x, &singleState);

    // Initialize the intra-block random number generator.
    // Pairs of threads get the same randoms.
    curand_init((unsigned long) clock(), blockIdx.x, threadIdx.x % (blockDim.x / 2), &coupleState);

    // Copy the family to shared memory.
    family[threadIdx.x] = pop[IDX(blockIdx.x, threadIdx.x, blockDim.x)];

    for (unsigned int g = 0; g < genNum; g++) {
        // Evaluation.
        evaluate(family);
        __syncthreads();

        // Selection.
        select(family, tmpFamily, fitters, &singleState);
        __syncthreads();

        // Crossover.
        if (curand_uniform(&coupleState) < CROSS_PROB) {
            crossover(family, tmpFamily, fitters, &singleState);
        }
        __syncthreads();

        // Mutation.
        mutate(family, tmpFamily, &singleState);
        __syncthreads();
    }

    // Sort the family.
    sort(family, tmpFamily);
    __syncthreads();

    // Copy the family back to global memory.
    pop[IDX(blockIdx.x, threadIdx.x, blockDim.x)] = family[threadIdx.x];
    __syncthreads();
}















int main(int argc, char const *argv[]) {
    Individual *population;
    Individual *d_population;

    unsigned int fieldSize = 500;
    unsigned int popSize = 2048;
    unsigned int famNumber = 32;
    unsigned int genNumber = 1000;
    double startTime = 0.0;
    double endTime = 0.0;
    cell_t *field;
    Point2D *checks;

    if (argc > 4) {
        printf("Usage: %s [fieldSize [popSize [genNumber]]\n", argv[0]);
        return -1;
    }
    if (argc > 1) {
        fieldSize = atoi(argv[1]);
    }
    if (argc > 2) {
        popSize = atoi(argv[3]);
    }
    if (argc > 3) {
        genNumber = atoi(argv[4]);
    }

    // Create a field of checks.
    field = (cell_t *) malloc(fieldSize * fieldSize * sizeof(cell_t));
    checks = (Point2D *) malloc(CHECKS_NUM * sizeof(Point2D));

    for (unsigned int i = 0; i < fieldSize * fieldSize; i++) {
        field[i] = EMPTY;
    }

    srand(time(NULL));
    for (unsigned int i = 0; i < CHECKS_NUM; i++) {
        checks[i].x = (rand() % fieldSize);
        checks[i].y = (rand() % fieldSize);
        // checks[i].id = i;
        field[IDX(checks[i].x, checks[i].y, fieldSize)] = true;
    }

    dump(field, NULL, fieldSize, CHECKS_NUM, "field.ppm");

    printf("Field:\n");
    for (unsigned int i = 0; i < CHECKS_NUM; i++) {
        printf("x:%d\ty:%d\n", checks[i].x, checks[i].y);
    }



    const size_t size = popSize * sizeof(Individual);

    dim3 members(popSize / famNumber);
    dim3 families(famNumber);
    size_t familySize = members.x * sizeof(Individual);
    size_t intArraySize = members.x * sizeof(int);
    size_t sharedMemSize = 2 * familySize + intArraySize;
    printf("total shared mem / block:\t%zuB\n", sharedMemSize);
    printf("family size:\t\t\t%zuB\n", familySize);
    printf("int array size:\t\t\t%zuB\n", intArraySize);

    // Create the host population.
    population = (Individual *) malloc(size);

    // Create the device populations.
    cudaMalloc(&d_population, size);
    cudaCheckError();

    // Initialize the population.
    initialize(population, popSize, checks, CHECKS_NUM);

    // Copy the host population to the device.
    cudaMemcpy(d_population, population, size, cudaMemcpyHostToDevice);






    // ***Execution.***
    printf("Execution:\n");
    startTime = hpc_gettime();

    evolve<<<families, members, sharedMemSize>>>(d_population, genNumber);
    cudaDeviceSynchronize();

    endTime = hpc_gettime();
    printf("Execution time: %fs\n\n", endTime - startTime);





    // Copy the device population back to the host.
    cudaMemcpy(population, d_population, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < famNumber; i++) {
        char fileName[255];
        sprintf(fileName, "BestOfFam%d.ppm", i);
        dump(field, population[i * members.x].path, fieldSize, CHECKS_NUM, fileName);
        printf("Family %d best score %f\n", i, population[i * members.x].score);
    }




    free(population);
    cudaFree(d_population);
    return 0;
}
