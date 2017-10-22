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

    __host__ __device__ void initialize();
    __device__ void evaluate();
    __device__ void crossover(CUDAGenome *partner, CUDAGenome *offspring);
    __device__ void mutate();
    __device__ void scale(float base);
    CUDAGenome *clone();
    void allocateCopySingle(CUDAGenome **deviceIndividual, CUDAGenome **hostIndividual, cudaMemcpyKind direction);
    void allocateCopyMultiple(CUDAGenome ***deviceIndividuals, CUDAGenome ***hostIndividuals, unsigned int count, cudaMemcpyKind direction);

    __host__ __device__ CUDAPathGenome(_Point2D *checks, unsigned int checksNum);
    __host__ __device__ void setCheck(unsigned int index, _Point2D check);
    __host__ __device__ _Point2D getCheck(unsigned int index) {
        return checks[index];
    }
    __host__ __device__ _Point2D *getPathCheck(unsigned int index) {
        return &path[index];
    }


    __host__ __device__ unsigned int getChecksNum() {
        return checksNumber;
    }
    __host__ __device__ _Point2D *getChecks() {
        return checks;
    }
    __host__ __device__ _Point2D *getPath() {
        return path;
    }

protected:
    unsigned int checksNumber;
};

__global__ void createCUDAPathGenome(CUDAGenome **genome, CUDAPathGenome::_Point2D *checks, unsigned int checksNum);

#endif
