#include <stdio.h>
#include <omp.h>

#define NUM_THREADS 4
#define INCREMENTS 100000

int counter = 0;

int main()
{
    omp_set_num_threads(NUM_THREADS);

    #pragma omp parallel
    {
        for (int i = 0; i < INCREMENTS; i++)
        {
            counter++;
        }
    }

    printf("Expected counter = %d\n",
           NUM_THREADS * INCREMENTS);

    printf("Actual counter   = %d\n",
           counter);

    return 0;
}
