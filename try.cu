#include "try.h"
#include <curand.h>
#include <curand_kernel.h>

__global__ void hi() {
    curandState_t state;
    curand_init((unsigned long) clock(), blockIdx.x, 0, &state);
    printf("random1:%u\n", curand(&state));
    curand_init((unsigned long) clock(), blockIdx.x, 0, &state);
    printf("random2:%u\n", curand(&state));
}
