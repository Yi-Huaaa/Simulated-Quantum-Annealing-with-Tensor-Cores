SHELL:=/bin/bash
CC=nvcc
CUDAFLAGS=-arch=sm_80 -lcublas
# target cuda code
TARGET=v5-1-8

SPINS=1024 2048 4096 8192 16384
TROTTERS=8 16 32 64 128

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
	@declare -A GSETS=( ["1024"]=G1 ["2048"]=G22 ["4096"]=G48 ["8192"]=G65 ["16384"]=G77 ["32767"]=G81 )\
	; for n in $(SPINS) ; do \
		for m in $(TROTTERS); do \
			Gset_file=$${GSETS[$${n}]}; \
			echo "Testing, N=$$n, M=$$m, Gset=$$Gset_file"; \
			./src/$(TARGET)-$$n-$$m Gset/$$Gset_file > ./speed_logs/$(TARGET)-$$n-$$m-log.txt; \
		done \
	done

clean: 
	rm -rf src/*
	rm -rf speed_logs/*
