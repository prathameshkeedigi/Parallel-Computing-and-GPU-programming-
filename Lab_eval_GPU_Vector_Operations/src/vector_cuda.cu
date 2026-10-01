#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void vectorAddKernel(const float *A, const float *B, float *C, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void vectorMulKernel(const float *A, const float *B, float *C, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] * B[idx];
    }
}

int main(int argc, char *argv[]) {
    long N = 10000000;
    if (argc > 1) {
        N = atol(argv[1]);
    }

    size_t bytes = N * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C_add = (float *)malloc(bytes);
    float *h_C_mul = (float *)malloc(bytes);

    for (long i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_B, bytes);
    cudaMalloc((void **)&d_C, bytes);

    int threadsPerBlock = 256;
    long blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // --- WARM-UP (Wakes up the GPU & initializes CUDA driver context) ---
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEvent_t totalStart, totalStop, kernelStart, kernelStop;
    cudaEventCreate(&totalStart);
    cudaEventCreate(&totalStop);
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);

    // --- PHASE 1: VECTOR ADDITION ---
    cudaEventRecord(totalStart);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(kernelStart);
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(kernelStop);
    cudaEventSynchronize(kernelStop);

    cudaMemcpy(h_C_add, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(totalStop);
    cudaEventSynchronize(totalStop);

    float kTimeAdd = 0.0f, totTimeAdd = 0.0f;
    cudaEventElapsedTime(&kTimeAdd, kernelStart, kernelStop);
    cudaEventElapsedTime(&totTimeAdd, totalStart, totalStop);

    // --- PHASE 2: VECTOR MULTIPLICATION ---
    cudaEventRecord(totalStart);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(kernelStart);
    vectorMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(kernelStop);
    cudaEventSynchronize(kernelStop);

    cudaMemcpy(h_C_mul, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(totalStop);
    cudaEventSynchronize(totalStop);

    float kTimeMul = 0.0f, totTimeMul = 0.0f;
    cudaEventElapsedTime(&kTimeMul, kernelStart, kernelStop);
    cudaEventElapsedTime(&totTimeMul, totalStart, totalStop);

    printf("Vector Size             : %ld elements\n", N);
    printf("Add Kernel Time         : %.6f seconds\n", kTimeAdd / 1000.0f);
    printf("Add Total Phase Time    : %.6f seconds\n", totTimeAdd / 1000.0f);
    printf("Mul Kernel Time         : %.6f seconds\n", kTimeMul / 1000.0f);
    printf("Mul Total Phase Time    : %.6f seconds\n", totTimeMul / 1000.0f);
    printf("Verification C_add[0]   : %.2f\n", h_C_add[0]);
    printf("Verification C_mul[0]   : %.2f\n", h_C_mul[0]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_add);
    free(h_C_mul);
    cudaEventDestroy(totalStart);
    cudaEventDestroy(totalStop);
    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelStop);
    return 0;
}
