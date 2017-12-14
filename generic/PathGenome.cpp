/*
**************************************************************************
PathGenome.cpp

This file contains the definition of the PathGenome methods.
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
    }

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
                child->path[i] = mate->path[j];
                i++;
            }
        }
    }
}

void PathGenome::mutate(float mutRate) {
    _Point2D *tmp = (_Point2D *) malloc(checksNumber * sizeof(_Point2D));

    for (unsigned int i = 0; i < checksNumber; i++) {
        if (((float) rand()) / (RAND_MAX + 1.0) <= mutRate) {
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
