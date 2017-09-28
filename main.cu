#include <stdio.h>
#include <stdlib.h>

__global__ void print() {
  printf("Hello, I am thread %d of block %d\n", threadIdx.x, blockIdx.x);
}

int main(int argc, char const *argv[]) {
  print<<<1, 10>>>();
  cudaDeviceSynchronize();
  return 0;
}
