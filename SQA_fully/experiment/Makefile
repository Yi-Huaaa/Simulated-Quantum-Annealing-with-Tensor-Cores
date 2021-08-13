SHELL:=/bin/bash
CC=nvcc
#CUDAFLAGS=-arch=sm_80 -lcublas -Xcompiler -fopenmp --default-stream per-thread
CUDAFLAGS=-arch=sm_70 -lcublas
# target cuda code
TARGET=v5-1-9

SPINS=1024 2048 4096 8192 16384 32768
#SPINS=131072ULL 262144ULL
#TROTTERS=16 32 64 128 256
TROTTERS=4 8 16 32 64 128 256 512
M_2=16 32 64 128 256 512 1024

all: directories binary

binary: $(TARGET).cu
	@for n in $(SPINS); do \
                for m in $(TROTTERS); do \
                        echo "Generate binary, N=$$n, M=$$m"; \
                        sed -i "s/^#define N .*$$/#define N $$n/g" $(TARGET).cu; \
                        sed -i "s/^#define M .*$$/#define M $$m/g" $(TARGET).cu; \
                        $(CC) $(CUDAFLAGS) $(TARGET).cu -o src/$(TARGET)-$$n-$$m; \
                done \
        done

directories:
	@# build binaries to src/
	@if ! [ -d "./src" ]; then mkdir src/; fi
	
	@# make sure Gset/ is here
	@if ! [ -d "./Gset" ]; then echo "Gset/ dose not exist, check http://web.stanford.edu/~yyye/yyye/Gset/"; exit 0 ;fi

speed:
	@# make sure Gset/ is here
	@if ! [ -d "./speed_logs" ]; then mkdir speed_logs/; fi
	
	@declare -A GSETS=( ["1024"]=G1 ["2048"]=G22 ["4096"]=G48 ["8192"]=G65 ["16384"]=G77 ["32768"]=G81 ["65536ULL"]=G81 )\
	; for n in $(SPINS) ; do \
		for m in $(TROTTERS); do \
			Gset_file=$${GSETS[$${n}]}; \
			echo "Testing, N=$$n, M=$$m, Gset=$$Gset_file"; \
			./src/$(TARGET)-$$n-$$m Gset/$$Gset_file > ./speed_logs/$(TARGET)-$$n-$$m-log.txt; \
		done \
	done

m2: 	
	@for m in $(TROTTERS); do \
		for m2 in $(M_2); do \
                        echo "Generate binary, M=$$m, M_2=$$m2"; \
                        sed -i "s/^#define M .*$$/#define M $$m/g" $(TARGET).cu; \
                        sed -i "s/^#define M_2 .*$$/#define M_2 $$m2/g" $(TARGET).cu; \
                        $(CC) $(CUDAFLAGS) $(TARGET).cu -o src/$(TARGET)-$$m-$$m2; \
                done \
        done
	
m2_run: 	
	@for m in $(TROTTERS); do \
		for m2 in $(M_2); do \
			nvprof src/$(TARGET)-$$m-$$m2 Gset/G81 > ./speed_logs/$(TARGET)-$$m-$$m2-log.txt; \
                done \
	done

m2_run_stats:
	@for m in $(TROTTERS); do \
		for m2 in $(M_2); do \
			echo "Testing, M=$$m, M_2=$$m2"; \
			cat ./speed_logs/$(TARGET)-$$m-$$m2-log.txt | grep "Avg"; \
		done \
	done
	
m2_plot:
	@cat nohup.out | grep -E "judge|gemm|src" > temp
	@python plot_m2.py

tc:
	@declare -A GSETS=( ["1024"]=G1 ["2048"]=G22 ["4096"]=G48 ["8192"]=G65 ["16384"]=G77 ["32768"]=G81 ["65536ULL"]=G81 )\
	; for n in $(SPINS); do \
		for m in $(TROTTERS); do \
			Gset_file=$${GSETS[$${n}]}; \
			sed -i "s/^#define N .*$$/#define N $$n/g" $(TARGET).cu; \
			sed -i "s/^#define M .*$$/#define M $$m/g" $(TARGET).cu; \
			echo "TC, N=$$n, M=$$m, Gset=$$Gset_file"; \
			$(CC) $(CUDAFLAGS) -DTC $(TARGET).cu ; \
			nvprof ./a.out Gset/$$Gset_file ; \
			echo "no TC, N=$$n, M=$$m, Gset=$$Gset_file"; \
			$(CC) $(CUDAFLAGS) $(TARGET).cu; \
			nvprof ./a.out Gset/$$Gset_file ; \
		done \
        done
	

profile:
	@sudo /usr/local/cuda-11.2/bin/nv-nsight-cu-cli --section ComputeWorkloadAnalysis --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables -f -o output ./a.out Gset/G65

speed_stats:
	@for n in $(SPINS) ; do \
		for m in $(TROTTERS); do \
			echo "Testing, N=$$n, M=$$m"; \
			cat ./speed_logs/$(TARGET)-$$n-$$m-log.txt | grep "Avg"; \
		done \
	done

clean: 
	rm -rf src/*
	rm -rf speed_logs/*
