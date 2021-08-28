#!/bin/bash
<<'COMMENT'
for m in {0..7..1}
do		
	mMul=$((2**$m))
	MM=$(( 4 * $mMul ))
	for n in {0..5..1}
	do
        nMul=$((2**$n))
        NN=$(( 1024 * $nMul ))
        echo M = ${MM} N = ${NN}
        nvcc -D N=$NN -D M=$MM v5-1-9.cu -arch=sm_86 -lcublas -o M${MM}N${NN}
	done
done
COMMENT

# <<'COMMENT'
#G6-G13
for m in {0..7..1}
do		
	mMul=$((2**$m))
	MM=$(( 4 * $mMul ))
	# for i in $(seq $1 $2)
	for i in {6..13..1}
	do
        echo M = ${MM} N = 1024 Gset = G${i}
        ./M${MM}N1024 ~/dataSet/Gset/G${i} > ./Results/G${i}M${MM}.out
	done
done

#G18-G21
for m in {0..7..1}
do		
	mMul=$((2**$m))
	MM=$(( 4 * $mMul ))
	# for i in $(seq $1 $2)
	for i in {18..21..1}
	do
        echo M = ${MM} N = 1024 Gset = G${i}
        ./M${MM}N1024 ~/dataSet/Gset/G${i}  > ./Results/G${i}M${MM}.out
	done
done
#G27-G34
for m in {0..7..1}
do		
	mMul=$((2**$m))
	MM=$(( 4 * $mMul ))
	for i in {27..34..1}
	do
        echo M = ${MM} N = 2048 Gset = G${i}
        ./M${MM}N2048 ~/dataSet/Gset/G${i}  > ./Results/G${i}M${MM}.out
	done
done
#G39-G42
for m in {0..7..1}
do		
	mMul=$((2**$m))
	MM=$(( 4 * $mMul ))
	for i in {39..43..1}
	do
        echo M = ${MM} N = 2048 Gset = G${i}
        ./M${MM}N2048 ~/dataSet/Gset/G${i}  > ./Results/G${i}M${MM}.out
	done
done
#G48-G50
for m in {0..7..1}
do		
	mMul=$((2**$m))
	MM=$(( 4 * $mMul ))
	for i in {48..50..1}
	do
        echo M = ${MM} N = 4096 Gset = G${i}
        ./M${MM}N4096 ~/dataSet/Gset/G${i}  > ./Results/G${i}M${MM}.out
	done
done
#G56-G57
for m in {0..7..1}
do		
	mMul=$((2**$m))
	MM=$(( 4 * $mMul ))
	for i in {56..57..1}
	do
        echo M = ${MM} N = 8192 Gset = G${i}
        ./M${MM}N8192 ~/dataSet/Gset/G${i}  > ./Results/G${i}M${MM}.out
	done
done
#G59
for m in {0..7..1}
do		
	mMul=$((2**$m))
	MM=$(( 4 * $mMul ))
	for i in {59..59..1}
	do
        echo M = ${MM} N = 8192 Gset = G${i}
        ./M${MM}N8192 ~/dataSet/Gset/G${i}  > ./Results/G${i}M${MM}.out
	done
done
#G61-G65
for m in {0..7..1}
do		
	mMul=$((2**$m))
	MM=$(( 4 * $mMul ))
	for i in {61..65..1}
	do
        echo M = ${MM} N = 8192 Gset = G${i}
        ./M${MM}N8192 ~/dataSet/Gset/G${i}  > ./Results/G${i}M${MM}.out
	done
done
#G66-G67
for m in {0..7..1}
do		
	mMul=$((2**$m))
	MM=$(( 4 * $mMul ))
	for i in {66..67..1}
	do
        echo M = ${MM} N = 16384 Gset = G${i}
        ./M${MM}N16384 ~/dataSet/Gset/G${i}  > ./Results/G${i}M${MM}.out
	done
done
#G72
for m in {0..7..1}
do		
	mMul=$((2**$m))
	MM=$(( 4 * $mMul ))
	for i in {72..72..1}
	do
        echo M = ${MM} N = 16384 Gset = G${i}
        ./M${MM}N16384 ~/dataSet/Gset/G${i}  > ./Results/G${i}M${MM}.out
	done
done
#G77
for m in {0..7..1}
do		
	mMul=$((2**$m))
	MM=$(( 4 * $mMul ))
	for i in {77..77..1}
	do
        echo M = ${MM} N = 16384 Gset = G${i}
        ./M${MM}N16384 ~/dataSet/Gset/G${i}  > ./Results/G${i}M${MM}.out
	done
done
#G81
for m in {0..7..1}
do		
	mMul=$((2**$m))
	MM=$(( 4 * $mMul ))
	for i in {81..81..1}
	do
        echo M = ${MM} N = 32768 Gset = G${i}
        ./M${MM}N32768 ~/dataSet/Gset/G${i}  > ./Results/G${i}M${MM}.out
	done
done


# COMMENT

