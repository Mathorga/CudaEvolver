#ifndef __MANAGED__
#define __MANAGED__

class Managed {
public:
    void *operator new (size_t len) {
        void *ptr;
        cudaMallocManaged(&ptr, len);
        return ptr;
    }

    void operator delete (void *ptr) {
        cudaFree(ptr);
    }
};

#endif
