#ifndef __CUDA_GENOME__
#define __CUDA_GENOME__

class CUDAGenome {
public:
    virtual void initialize();
    __device__ virtual evaluate();
    __device__ virtual void crossover();
    __device__ virtual void mutate();

    CUDAGenome(unsigned int xDim, unsigned int yDim = 1, unsigned int zDim = 1) {
        xSize = xDim;
        ySize = yDim;
        zSize = zDim;
        score = 0;
    }

    __host__ __device__ unsigned int xSize() {
        return xSize;
    }
    __host__ __device__ unsigned int ySize() {
        return ySize;
    }
    __host__ __device__ unsigned int zSize() {
        return zSize;
    }
    __host__ __device__ float score() {
        return score;
    }

protected:
    unsigned int xSize;
    unsigned int ySize;
    unsigned int zSize;
    float score;
};

#endif
