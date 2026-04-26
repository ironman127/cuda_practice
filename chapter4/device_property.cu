#include<stdio.h>
#include<cuda.h>

void print_device(int dev_i) {
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, dev_i);

    printf("maxThreadsPerBlock:%d,\n", dev_prop.maxThreadsPerBlock);
    printf("the number of SMs:%d,\n", dev_prop.multiProcessorCount);
    printf("clock rate:%d,\n", dev_prop.clockRate);
    printf("block dimension limit, x:%d, y:%d, z:%d, \n", \
        dev_prop.maxThreadsDim[0],  dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);
    printf("grid dimension limit, x:%d, y:%d, z:%d,\n", \
        dev_prop.maxGridSize[0], dev_prop.maxGridSize[1], dev_prop.maxGridSize[2]);
    printf("register num per SM:%d,\n", dev_prop.regsPerBlock);
    printf("warp size:%d,\n", dev_prop.warpSize);
}

int main() {
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    printf("total device:%d.\n", dev_count);

    for (int i=0; i < dev_count; ++i) {
        print_device(i);
    }

    return 0;
}