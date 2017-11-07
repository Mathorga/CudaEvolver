#ifndef __CUDA_PATH_GENOME__
#define __CUDA_PATH_GENOME__

#include "CUDAGenome.h"

#define COORD_SIZE sizeof(unsigned int)
#define POINT_SIZE 2 * COORD_SIZE

class CUDAPathGenome : public CUDAGenome {
public:
    typedef struct {
        unsigned int x = 0;
        unsigned int y = 0;
        int id = -1;
    } _Point2D;

    _Point2D *checks;
    _Point2D *path;
    float *distances;

    __device__ void initialize();
    __device__ void evaluate();
    __device__ void crossover(CUDAGenome *partner, CUDAGenome **offspring);
    __device__ void mutate();
    __device__ CUDAGenome *clone();
    __device__ void scale(float base);
    __device__ void print();
    __device__ void output(char *string);
    __device__ void dealloc();

    __device__ CUDAPathGenome(_Point2D *checks, unsigned int checksNum);
    __device__ void setCheck(unsigned int index, _Point2D check);
    __device__ _Point2D getCheck(unsigned int index) {
        return checks[index];
    }
    __device__ _Point2D *getPathCheck(unsigned int index) {
        return &path[index];
    }
    __device__ _Point2D *getChecks() {
        return checks;
    }
    __device__ _Point2D *getPath() {
        return path;
    }
    __device__ unsigned int getChecksNum() {
        return checksNumber;
    }

protected:
    unsigned int checksNumber;
};

#endif
