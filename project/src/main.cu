#include <iostream>
#include <cuda_runtime.h>

__global__ void hello_kernel() {
    printf("Hello from CUDA kernel!\\n");
}

int main() {
    std::cout << "Hello from CPU main()" << std::endl;

    hello_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    return 0;
}
