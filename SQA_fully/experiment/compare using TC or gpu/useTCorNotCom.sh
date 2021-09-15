#!/bin/bash
# <<'COMMENT'
# proposed algo, with TC
for m in {0..7..1}
do
    mMul=$((2**$m))
    MM=$(( 4 * $mMul ))
    for n in {0..0..1}
    do
    nMul=$((2**$n))
    NN=$(( 32768 * $nMul ))
    echo tc_${MM}_${NN}
    nvcc -D N=$NN -D M=$MM v5-1-9.cu -arch=sm_86 -lcublas -o tc_${MM}_${NN}
    nsys nvprof ./tc_${MM}_${NN} ~/dataSet/Gset/G1  > ./Results/tc_${MM}_${NN}.out
    grep -E "gemm"  ./Results/tc_${MM}_${NN}.out > ./tcUsedTime/tc_${MM}_${NN}.out
    done
done
# COMMENT


<<'COMMENT'
# proposed algo, without TC
for m in {0..7..1}
do
    mMul=$((2**$m))
    MM=$(( 4 * $mMul ))
    for n in {0..0..1}
    do
    nMul=$((2**$n))
    NN=$(( 32768 * $nMul ))
    echo no_${MM}_${NN}
    nvcc -D N=$NN -D M=$MM v5-1-9.cu -arch=sm_86 -lcublas -o no_${MM}_${NN}
    nsys nvprof ./no_${MM}_${NN} ~/dataSet/Gset/G1  > ./Results/no_${MM}_${NN}.out
    grep -E "gemm"  ./Results/no_${MM}_${NN}.out > ./tcUsedTime/no_${MM}_${NN}.out
    done
done
COMMENT

<<'COMMENT'
    nvcc -D N=2048 -D M=4 v5-1-9.cu -arch=sm_86 -lcublas -o tc_4_2048
    nsys nvprof ./tc_4_2048 ~/dataSet/Gset/G1  > ./Results/tc_4_2048.out
    grep -E "gemm" ./Results/tc_4_2048.out > ./tcUsedTime/tc_4_2048.out
COMMENT

