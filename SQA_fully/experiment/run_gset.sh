#!/bin/bash

# <<'COMMENT'
# exefile
for m in {1..64..1} # 1-64
do		
	for n in {0..1..1} # 1024-2048, 0-1
	do
        for g in {0..5..1} # G0, original = 8, from 2 to 32, 0 - 5
        do
        	for inc in {0..5..1} # original = 16, from 2 to 32, 0 - 5
        	do 
		        nMul=$((2**$n))
		        NN=$(( 1024 * $nMul ))
		        g0=$((2**$g))
		        iV=$((2**$inc))
		        echo M = ${m} N = ${NN} G0 = ${g0} incVal = ${iV}
		        nvcc -D N=$NN -D M=$m -D G0=$g0 -D incVal=$iV v5-1-9.cu -arch=sm_86 -lcublas -o ./exeF/M${m}N${NN}G0${g0}IV${iV}
        	done        
        done
	done
done



# 1024
declare -a array1=(13 18 19 20 21 47 51 53 54)

# for((i=0; i<${#array1[@]}; i++))
# do
# 	echo ${array1[i]}
# done
for m in {1..64..1}  # 1-64
do	
    for g in {0..5..1} # G0, original = 8, from 2 to 32
    do
    	for inc in {0..5..1} # original = 16, from 2 to 32
    	do 
	        for((gsetNum=0; gsetNum<${#array1[@]}; gsetNum++))
	        do
		        g0=$((2**$g))
		        iV=$((2**$inc))
	   			echo M${m}N1024G0${g0}IV${iV}  G${array1[gsetNum]}M${m}G0${g0}IV${iV}.out
			    ./exeF/M${m}N1024G0${g0}IV${iV} ~/Gset/G${array1[gsetNum]} > ./Results/G${array1[gsetNum]}M${m}G0${g0}IV${iV}.out
	        done
    	done        
    done
done

# COMMENT

#2048
declare -a array2=(34 39 40 41 42)
for m in {1..64..1}  # 1-64
do	
    for g in {0..5..1} # G0, original = 8, from 2 to 32
    do
    	for inc in {0..5..1} # original = 16, from 2 to 32
    	do 
	        for((gsetNum=0; gsetNum<${#array2[@]}; gsetNum++))
	        do
		        g0=$((2**$g))
		        iV=$((2**$inc))
	   			echo M${m}N2048G0${g0}IV${iV}  G${array2[gsetNum]}M${m}G0${g0}IV${iV}.out
			    ./exeF/M${m}N2048G0${g0}IV${iV} ~/Gset/G${array2[gsetNum]} > ./Results/G${array2[gsetNum]}M${m}G0${g0}IV${iV}.out
	        done
    	done        
    done
done



