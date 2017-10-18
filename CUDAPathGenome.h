#ifndef __CUDA_PATH_GENOME__
#define __CUDA_PATH_GENOME__

#include "CUDAGenome.h"

class CUDAPathGenome : public CUDAGenome {
public:
    typedef struct {
        unsigned int x;
        unsigned int y;
        unsigned int id;
    } _2DDot;

    void initialize();
    __device__ float evaluate();
    __device__ CUDAGenome *crossover(CUDAGenome *partner);
    __device__ void mutate();
    __device__ void scale(float base);
    CUDAGenome *clone();

    CUDAPathGenome(unsigned int checksNum);

    _2DDot *getPath();

    __device__ void setPath(_2DDot *p);
    __device__ void setCheck(unsigned int index, _2DDot check);
    __device__ _2DDot getCheck(unsigned int index);

protected:
    unsigned int checksNumber;
    _2DDot *checks;
    _2DDot *d_checks;
    _2DDot *path;
    _2DDot *d_path;
    float *distances;
    float *d_distances;
};

#endif
