#include "PathGenome.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void PathGenome::initialize() {
    // Create a copy of the checks array.
    _Point2D *checksCopy = (_Point2D *) malloc((checksNumber + 1) * sizeof(_Point2D));
    for (unsigned int i = 0; i < checksNumber; i++) {
        checksCopy[i] = checks[i];
        // printf("x:%d\ty:%d\n", checks[i].x, checks[i].y);
    }
    // printf("Created a copy of checks\n");

    // Randomly initialize path;
    for (unsigned int i = 0; i < checksNumber; i++) {
        int index = rand() % (checksNumber - i);
        path[i] = checksCopy[index];
        for (unsigned int j = index; j < checksNumber - i; j++) {
            checksCopy[j] = checksCopy[j + 1];
        }
    }
}

void PathGenome::evaluate() {
    float *tmpDists;
    tmpDists = (float *) malloc(checksNumber * sizeof(float));
    score = 0;

    // Calculate distances between each check.
    for (unsigned int i = 0; i < checksNumber; i++) {
        float dx = (float) path[(i + 1) % checksNumber].x - (float) path[i].x;
        float dy = (float) path[(i + 1) % checksNumber].y - (float) path[i].y;
        score += sqrt(pow(dx, 2) + pow(dy, 2));
    }
}

void PathGenome::crossover(Genome *partner, Genome **offspring) {
    PathGenome *child = (PathGenome *) (*offspring);
    PathGenome *mate = (PathGenome *) partner;
    unsigned int midPoint = 0;

    midPoint = rand() % (checksNumber - 1);

    // Pick from parent 1.
    for (unsigned int i = 0; i <= midPoint; i++) {
        child->path[i] = path[i];
    }

    // Pick from parent 2.
    for (unsigned int i = midPoint + 1; i < checksNumber; ) {
        for (unsigned int j = 0; j < checksNumber; j++) {
            bool insert = true;
            for (unsigned int k = 0; k <= midPoint; k++) {
                if (mate->path[j].id == child->path[k].id) {
                    insert = false;
                    break;
                }
            }
            if (insert) {
                // printf("Inserting index %u to index %u\n", j, i);
                child->path[i] = mate->path[j];
                i++;
            }
        }
    }
}

void PathGenome::mutate(float mutRate) {
    _Point2D *tmp = (_Point2D *) malloc(checksNumber * sizeof(_Point2D));

    for (unsigned int i = 0; i < checksNumber; i++) {
        // printf("%f out of %f\n", ((float) rand()) / (RAND_MAX + 1.0), mutRate);
        if (((float) rand()) / (RAND_MAX + 1.0) <= mutRate) {
            // printf("%f\n", ((float) rand()) / (RAND_MAX + 1.0));
            int firstIndex = i;
            int secondIndex = rand() % (checksNumber - 1);

            for (unsigned int j = 0; j < checksNumber; j++) {
                tmp[j] = path[j];
            }

            path[firstIndex] = path[secondIndex];
            path[secondIndex] = tmp[firstIndex];

        }
    }
}

Genome *PathGenome::clone() {
    return new PathGenome(checks, checksNumber);
}

void PathGenome::scale(float baseScore) {
    fitness = (baseScore - score) + 1;
}

void PathGenome::print() {
    for (unsigned int i = 0; i < checksNumber; i++) {
        printf("x:%u\ty:%u\tid:%d\n", path[i].x, path[i].y, path[i].id);
    }
};

// void PathGenome::output(char *string) {
//     for (int i = 0; i < COORD_SIZE; i++) {
//         memcpy(&(string[threadIdx.x * POINT_SIZE]), &(path[threadIdx.x].x), COORD_SIZE);
//         memcpy(&(string[threadIdx.x * POINT_SIZE + COORD_SIZE]), &(path[threadIdx.x].y), COORD_SIZE);
//     }
// }


PathGenome::PathGenome(_Point2D *checkArray, unsigned int checksNum) : Genome(checksNum) {
    checksNumber = checksNum;
    checks = (_Point2D *) malloc(checksNum * sizeof(_Point2D));
    path = (_Point2D *) malloc(checksNum * sizeof(_Point2D));
    distances = (float *) malloc(checksNum * sizeof(float));
    score = 0;
    fitness = 0;
    for (unsigned int i = 0; i < checksNum; i++) {
        checks[i] = checkArray[i];
        _Point2D newCheck;
        path[i] = newCheck;
        distances[i] = 0.0;
    }
}
