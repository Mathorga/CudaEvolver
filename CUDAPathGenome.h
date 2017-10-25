#ifndef __CUDA_PATH_GENOME__
#define __CUDA_PATH_GENOME__

#include "CUDAGenome.h"

class CUDAPathGenome : public CUDAGenome {
public:
    typedef struct {
        unsigned int x;
        unsigned int y;
        unsigned int id;
    } _Point2D;

    _Point2D *checks;
    _Point2D *path;
    float *distances;

    __device__ void initialize();
    __device__ void evaluate();
    __device__ void crossover(CUDAGenome *partner, CUDAGenome *offspring);
    __device__ void mutate();
    __device__ void scale(float base);
    __host__ __device__ CUDAGenome *clone();
    __host__ __device__ void print();

    __host__ __device__ CUDAPathGenome(_Point2D *checks, unsigned int checksNum);
    __host__ __device__ void setCheck(unsigned int index, _Point2D check);
    __host__ __device__ _Point2D getCheck(unsigned int index) {
        return checks[index];
    }
    __host__ __device__ _Point2D *getPathCheck(unsigned int index) {
        return &path[index];
    }
    __host__ __device__ _Point2D *getChecks() {
        return checks;
    }
    __host__ __device__ _Point2D *getPath() {
        return path;
    }
    __host__ __device__ unsigned int getChecksNum() {
        return checksNumber;
    }

protected:
    unsigned int checksNumber;
};

#endif
