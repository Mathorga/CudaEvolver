#ifndef __CUDA_GA_GENOME__
#define __CUDA_GA_GENOME__

// #define CUDA_MEMBER __host__ __device__

#include <ga/GAGenome.h>

class CudaGaGenome : public GAGenome {
public:
    GADefineIdentity("CudaGaGenome", 201);
    static void Initializer(GAGenome &);
    static int Mutator(GAGenome &);
    static float Comparator(GAGenome &);
    static float Evaluator(GAGenome &);
    // Hide the superclass' member function.
    __device__ float evaluate(GABoolean flag = gaFalse);

public:

};

#endif
