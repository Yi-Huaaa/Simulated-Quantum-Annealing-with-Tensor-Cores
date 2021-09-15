#!/bin/bash
# <<'COMMENT'
# proposed algo, with TC
for m in {0..7..1}
do
    mMul=$((2**$m))
    MM=$(( 4 * $mMul ))
    for n in {0..5..1}
    do
    nMul=$((2**$n))
    NN=$(( 1024 * $nMul ))
    echo tc_${MM}_${NN}
    # grep -E "gemm"  ./Results/tc_${MM}_${NN}.out
    grep -E "gemm"  ./Results/tc_${MM}_${NN}.out > ./tcUsedTime/tc_${MM}_${NN}.out
    done
done
# COMMENT