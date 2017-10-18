#ifndef __CUDA_GENOME__
#define __CUDA_GENOME__

class CUDAGenome {
public:
    virtual void initialize();
    __device__ virtual void evaluate();
    __device__ virtual CUDAGenome *crossover(CUDAGenome *partner, CUDAGenome *offspring);
    __device__ virtual void mutate();
    __device__ virtual void scale(float base);
    virtual CUDAGenome *clone();

    CUDAGenome(unsigned int xDim, unsigned int yDim = 1, unsigned int zDim = 1) {
        xSize = xDim;
        ySize = yDim;
        zSize = zDim;
        score = 0;
    }

    __host__ __device__ unsigned int getXSize() {
        return xSize;
    }
    __host__ __device__ unsigned int getYSize() {
        return ySize;
    }
    __host__ __device__ unsigned int getZSize() {
        return zSize;
    }
    __host__ __device__ float getScore() {
        return score;
    }
    __host__ __device__ float getFitness() {
        return fitness;
    }

protected:
    unsigned int xSize;
    unsigned int ySize;
    unsigned int zSize;
    float score;
    float fitness;
};

#endif
