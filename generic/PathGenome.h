/*
**************************************************************************
PathGenome.h

This file implements an implementation to the Genome class for the
TSP.
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

#ifndef __PATH_GENOME__
#define __PATH_GENOME__

#include "Genome.h"

#define COORD_SIZE sizeof(unsigned int)
#define POINT_SIZE 2 * COORD_SIZE

class PathGenome : public Genome {
public:
    typedef struct {
        unsigned int x = 0;
        unsigned int y = 0;
        int id = -1;
    } _Point2D;

    _Point2D *checks;
    _Point2D *path;
    float *distances;

    void initialize();
    void evaluate();
    void crossover(Genome *partner, Genome **offspring);
    void mutate(float mutRate);
    Genome *clone();
    void scale(float base);
    void print();
    // void output(char *string);

    PathGenome(_Point2D *checks, unsigned int checksNum);
    void setCheck(unsigned int index, _Point2D check);
    _Point2D getCheck(unsigned int index) {
        return checks[index];
    }
    _Point2D *getPathCheck(unsigned int index) {
        return &path[index];
    }
    _Point2D *getChecks() {
        return checks;
    }
    _Point2D *getPath() {
        return path;
    }
    unsigned int getChecksNum() {
        return checksNumber;
    }

protected:
    unsigned int checksNumber;
};

#endif
