#include <stdio.h>

class Parent {public: int my_id;};
class Child : public Parent {};

class CopyClass {
  public:
    Parent ** par;
};

const int length = 5;

__global__ void test_kernel(CopyClass *my_class){

  for (int i = 0; i < length; i++)
    printf("object: %d, id: %d\n", i, my_class->par[i]->my_id);
}

int main(){


//Instantiate object on the CPU
  CopyClass cpuClass;
  cpuClass.par = new Parent*[length];
  for(int i = 0; i < length; ++i) {
    cpuClass.par[i] = new Child;
    cpuClass.par[i]->my_id = i+1;} // so we can prove that things are working

//Allocate storage for object onto GPU and copy host object to device
  CopyClass * gpuClass;
  cudaMalloc(&gpuClass,sizeof(CopyClass));
  cudaMemcpy(gpuClass,&cpuClass,sizeof(CopyClass),cudaMemcpyHostToDevice);

//Copy dynamically allocated child objects to GPU
  Parent ** d_par;
  d_par = new Parent*[length];
  for(int i = 0; i < length; ++i) {
    cudaMalloc(&d_par[i],sizeof(Child));
    printf("\tCopying data\n");
    cudaMemcpy(d_par[i],cpuClass.par[i],sizeof(Child),cudaMemcpyHostToDevice);
  }

//Copy the d_par array itself to the device

  Parent ** td_par;
  cudaMalloc(&td_par, length * sizeof(Parent *));
  cudaMemcpy(td_par, d_par, length * sizeof(Parent *), cudaMemcpyHostToDevice);

//copy *pointer value* of td_par to appropriate location in top level object
  cudaMemcpy(&(gpuClass->par),&(td_par),sizeof(Parent **),cudaMemcpyHostToDevice);

  test_kernel<<<1,1>>>(gpuClass);
  cudaDeviceSynchronize();
  return 0;


}
