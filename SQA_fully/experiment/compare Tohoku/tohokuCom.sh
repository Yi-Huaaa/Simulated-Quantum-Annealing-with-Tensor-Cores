#!/bin/bash

# proposed algo, with TC
for m in {0..7..1}
do
    mMul=$((2**$m))
    MM=$(( 4 * $mMul ))
    for n in {0..0..1} # 0 3 1
    do
    nMul=$((2**$n))
    NN=$(( 4096 * $nMul ))
    echo tc_${MM}_${NN}
    nvcc -D N=$NN -D M=$MM v5-1-9.cu -arch=sm_86 -lcublas -o tc_${MM}_${NN}
    ./tc_${MM}_${NN} ~/dataSet/Gset/G1  > ./Results/tc_${MM}_${NN}.out
    done
done

<<'COMMENT'
# proposed algo, without TC
for m in {0..7..1}
do
    mMul=$((2**$m))
    MM=$(( 4 * $mMul ))
    for n in {0..3..1} # 0 3 1
    do
    nMul=$((2**$n))
    NN=$(( 4096 * $nMul ))
    echo no_${MM}_${NN}
    nvcc -D N=$NN -D M=$MM v5-1-9.cu -arch=sm_86 -lcublas -o no_${MM}_${NN}
    ./no_${MM}_${NN} ~/dataSet/Gset/G1  > ./Results/no_${MM}_${NN}.out
    done
done
COMMENT