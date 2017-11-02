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
