#!/bin/bash
mkdir -p results
CSV_FILE="results/timing_data.csv"
echo "N,CPU_Total_Time,GPU_Add_Kernel,GPU_Add_Total,GPU_Mul_Kernel,GPU_Mul_Total" > $CSV_FILE

SIZES=(100000 1000000 5000000 10000000 25000000)

for N in "${SIZES[@]}"
do
    echo "Running benchmark for N = $N..."
    CPU_OUT=$(./bin/vector_cpu $N)
    CPU_TIME=$(echo "$CPU_OUT" | grep "Total CPU Time" | awk '{print $5}')

    GPU_OUT=$(./bin/vector_cuda $N)
    GPU_ADD_K=$(echo "$GPU_OUT" | grep "Add Kernel Time" | awk '{print $5}')
    GPU_ADD_TOT=$(echo "$GPU_OUT" | grep "Add Total Phase Time" | awk '{print $6}')
    GPU_MUL_K=$(echo "$GPU_OUT" | grep "Mul Kernel Time" | awk '{print $5}')
    GPU_MUL_TOT=$(echo "$GPU_OUT" | grep "Mul Total Phase Time" | awk '{print $6}')

    echo "$N,$CPU_TIME,$GPU_ADD_K,$GPU_ADD_TOT,$GPU_MUL_K,$GPU_MUL_TOT" >> $CSV_FILE
done

echo "Benchmark Complete! Results saved to $CSV_FILE"
cat $CSV_FILE
