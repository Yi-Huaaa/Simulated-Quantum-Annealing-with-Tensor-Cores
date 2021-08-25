# run.sh
#!/bin/bash
#compile
#nvcc -D N=8192 -D M=4 -D M_2=128 v5-1-9.cu -arch=sm_86 -lcublas
# 6 7 5
for mTwo in {0..6..1}
	do
		mTwoMul=$((2**$mTwo))
		MTwo=$(( 16 * $mTwoMul ))
	for m in {0..7..1}
	do		
		mMul=$((2**$m))
		MM=$(( 4 * $mMul ))
		for n in {0..5..1}
		do
	        nMul=$((2**$n))
	        NN=$(( 1024 * $nMul ))
	        echo $MM $NN $MTwo
	        #echo tc_${MM}_${NN}_${MTwo}
	        nvcc -D N=$NN -D M=$MM -D M_2=$MTwo v5-1-9.cu -arch=sm_86 -lcublas -o tc_${MM}_${NN}_${MTwo}
		done
	done
done


#---
# com.sh
#!/bin/bash
# exe
for mTwo in {0..6..1}
	do
		mTwoMul=$((2**$mTwo))
		MTwo=$(( 16 * $mTwoMul ))
	for m in {0..7..1}
	do		
		mMul=$((2**$m))
		MM=$(( 4 * $mMul ))
 
        echo $MM 1024 $MTwo
        nsys nvprof ./tc_${MM}_1024_${MTwo} ~/dataSet/Gset/G1   > ./Results/tR_${MM}_1024_${MTwo}.out
        echo $MM 2048 $MTwo
        nsys nvprof ./tc_${MM}_2048_${MTwo} ~/dataSet/Gset/G22  > ./Results/tR_${MM}_2048_${MTwo}.out
        echo $MM 4096 $MTwo
        nsys nvprof ./tc_${MM}_4096_${MTwo} ~/dataSet/Gset/G48  > ./Results/tR_${MM}_4096_${MTwo}.out
        echo $MM 8192 $MTwo
        nsys nvprof ./tc_${MM}_8192_${MTwo} ~/dataSet/Gset/G65  > ./Results/tR_${MM}_8192_${MTwo}.out
        echo $MM 16384 $MTwo
        nsys nvprof ./tc_${MM}_16384_${MTwo} ~/dataSet/Gset/G77 > ./Results/tR_${MM}_16384_${MTwo}.out
        echo $MM 32768 $MTwo
        nsys nvprof ./tc_${MM}_32768_${MTwo} ~/dataSet/Gset/G81 > ./Results/tR_${MM}_32768_${MTwo}.out
	done
done


# cut.sh
# cut
#grep -E 'Avg time|judge_flipping_com|gemm' tR_4_1024_16.out
for mTwo in {0..6..1}
	do
		mTwoMul=$((2**$mTwo))
		MTwo=$(( 16 * $mTwoMul ))
	for m in {0..7..1}
	do		
		mMul=$((2**$m))
		MM=$(( 4 * $mMul ))
 
        grep -E 'Avg time|judge_flipping_com|gemm' tR_${MM}_1024_${MTwo}.out  > tC_${MM}_1024_${MTwo}.out
        grep -E 'Avg time|judge_flipping_com|gemm' tR_${MM}_2048_${MTwo}.out  > tC_${MM}_2048_${MTwo}.out
        grep -E 'Avg time|judge_flipping_com|gemm' tR_${MM}_4096_${MTwo}.out  > tC_${MM}_4096_${MTwo}.out
        grep -E 'Avg time|judge_flipping_com|gemm' tR_${MM}_8192_${MTwo}.out  > tC_${MM}_8192_${MTwo}.out
        grep -E 'Avg time|judge_flipping_com|gemm' tR_${MM}_16384_${MTwo}.out > tC_${MM}_16384_${MTwo}.out
        grep -E 'Avg time|judge_flipping_com|gemm' tR_${MM}_32768_${MTwo}.out > tC_${MM}_32768_${MTwo}.out
	done
done

