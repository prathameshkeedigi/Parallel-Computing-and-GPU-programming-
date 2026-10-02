#include <stdio.h>
#include <omp.h>

int main()
{
    omp_set_num_threads(4);

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();

        printf("Thread %d completed Stage 1\n", thread_id);

        #pragma omp barrier

        printf("Thread %d started Stage 2\n", thread_id);
    }

    return 0;
}
