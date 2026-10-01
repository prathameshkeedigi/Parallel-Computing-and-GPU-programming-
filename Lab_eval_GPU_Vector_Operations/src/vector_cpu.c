#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    long N = 10000000;
    if (argc > 1) {
        N = atol(argv[1]);
    }

    size_t bytes = N * sizeof(float);
    float *A = (float *)malloc(bytes);
    float *B = (float *)malloc(bytes);
    float *C_add = (float *)malloc(bytes);
    float *C_mul = (float *)malloc(bytes);

    if (!A || !B || !C_add || !C_mul) {
        printf("Host memory allocation failed\n");
        return 1;
    }

    for (long i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // Vector Addition
    clock_t start_add = clock();
    for (long i = 0; i < N; i++) {
        C_add[i] = A[i] + B[i];
    }
    clock_t end_add = clock();
    double time_add = (double)(end_add - start_add) / CLOCKS_PER_SEC;

    // Vector Multiplication
    clock_t start_mul = clock();
    for (long i = 0; i < N; i++) {
        C_mul[i] = A[i] * B[i];
    }
    clock_t end_mul = clock();
    double time_mul = (double)(end_mul - start_mul) / CLOCKS_PER_SEC;

    printf("Vector Size             : %ld elements\n", N);
    printf("Addition Time           : %f seconds\n", time_add);
    printf("Multiplication Time      : %f seconds\n", time_mul);
    printf("Total CPU Time          : %f seconds\n", time_add + time_mul);
    printf("Verification C_add[0]   : %.2f\n", C_add[0]);
    printf("Verification C_mul[0]   : %.2f\n", C_mul[0]);

    free(A);
    free(B);
    free(C_add);
    free(C_mul);
    return 0;
}
