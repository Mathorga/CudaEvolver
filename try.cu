#include "try.h"
#include <curand.h>
#include <curand_kernel.h>

__global__ void hi() {
    curandState_t state;
    curand_init((unsigned long) clock(), blockIdx.x, 0, &state);
    unsigned int random = curand(&state);
    float r = curand_uniform(&state);
    printf("random int:%u\tconverted:%f\tcapped:%f\n", random, (float) random, r);
}
