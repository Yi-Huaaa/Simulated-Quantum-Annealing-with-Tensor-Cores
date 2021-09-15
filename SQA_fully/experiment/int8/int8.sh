#!/bin/bash
<<'COMMENT'
# proposed algo, with TC
for m in {0..4..1}
do
    mMul=$((2**$m))
    MM=$(( 32 * $mMul ))
    for n in {0..5..1} 
    do
    nMul=$((2**$n))
    NN=$(( 1024 * $nMul ))
    echo tc_${MM}_${NN}
    nvcc -D N=$NN -D M=$MM v5-1-9.cu -arch=sm_86 -lcublas -o tc_${MM}_${NN}
    ./tc_${MM}_${NN} ~/dataSet/Gset/G1  > ./Results/tc_${MM}_${NN}.out
    done
done
COMMENT

# <<'COMMENT'
# INT8
for m in {0..4..1}
do
    mMul=$((2**$m))
    MM=$(( 32 * $mMul ))
    for n in {0..5..1} 
    do
    nMul=$((2**$n))
    NN=$(( 1024 * $nMul ))
    echo int_${MM}_${NN}
    nvcc -D N=$NN -D M=$MM v5-INT8.cu -arch=sm_86 -lcublas -o int_${MM}_${NN}
    ./int_${MM}_${NN} ~/dataSet/Gset/G1  > ./Results/int_${MM}_${NN}.out
    done
done
# COMMENT