#include "Individual.h"

__host__ __device__ bool operator == (const Point2D point1, const Point2D point2) {
    return (point1.x == point2.x && point1.y == point2.y);
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
