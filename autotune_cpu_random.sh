#!/bin/bash

A_dim=$1
B_dim=$2
C_dim=$3
mode=$4
infile=matrix_transposed.npy
python generate_test_matrix.py $A_dim $B_dim $C_dim $mode

C_blocks=1
Gy=1
besttime=100000
for AT in 1; do
	for B_blocks in 2; do
		for CT in 1; do

			if [ $(($AT * $CT)) -gt 12 ]; then
				continue
			fi
			python code_gen_cpu.py --A_dim $A_dim --B_dim $B_dim --C_dim $C_dim --AT $AT --CT $CT --B_blocks $B_blocks --C_blocks $C_blocks --Gy $Gy --infile $infile --outfile testing.cpp --outfile_asm test1.s --x86 --no_relu --infile_bias bias.npy --fuse
			#icc -fopenmp -shared -fPIC -O3 -march=native testing.cpp -o test1.s -S
			gcc -shared -g test1.s -o test.so
			icc -I . -mkl -O0 -g -march=native -D AT=$AT -D CT=$1 -D C_Blocks=$C_blocks -DA_dim=$A_dim -DINFILE=$infile -D B_dim=$B_dim -D C_dim=$C_dim -D C_blocks=$C_blocks -D X86=1 -D MULTI=0 driver_cpu.cpp -lcnpy -o test -std=c++17
			./test > runtime
			python test_equivalence.py cpu_output.npy ref.npy
                        runtime=$(grep "millisecond" runtime | awk '{print $3}')
                        cat runtime
                        if (( $(echo "$runtime < $besttime" | bc -l) )) ; then
			       	besttime=$runtime
				bestset=${AT}_${B_blocks}_${CT}
                        fi

		done
	done

done
echo Best Runtime $besttime
echo $bestset
