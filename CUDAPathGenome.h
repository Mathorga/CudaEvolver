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
    __device__ void evaluate();
    __device__ void crossover(CUDAGenome *partner, CUDAGenome *offspring);
    __device__ void mutate();
    __device__ void scale(float base);
    CUDAGenome *clone();
    void allocateCopySingle(CUDAGenome **deviceIndividual, CUDAGenome **hostIndividual, cudaMemcpyKind direction);
    void allocateCopyMultiple(CUDAGenome ***deviceIndividuals, CUDAGenome ***hostIndividuals, unsigned int count, cudaMemcpyKind direction);

    CUDAPathGenome(_2DDot *checks, unsigned int checksNum);

    _2DDot *getPath();

    __device__ void setPath(_2DDot *p) {
        path = p;
    }
    __device__ void setCheck(unsigned int index, _2DDot check);
    __device__ _2DDot getCheck(unsigned int index) {
        return checks[index];
    }
    __host__ __device__ _2DDot *getPathCheck(unsigned int index) {
        return &path[index];
    }

    _2DDot *getHostChecks() {
        return checks;
    }
    _2DDot *getHostPath() {
        return path;
    }
    _2DDot *getDeviceChecks() {
        return d_checks;
    }
    _2DDot *getDevicePath() {
        return d_path;
    }
    _2DDot **getDeviceChecksAddress() {
        return &d_checks;
    }
    _2DDot **getDevicePathAddress() {
        return &d_path;
    }
    float **getDeviceDistancesAddress() {
        return &d_distances;
    }

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
