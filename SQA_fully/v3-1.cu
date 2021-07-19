#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>
#include "cuda_profiler_api.h"
#include <cublas_v2.h>
#include <mma.h>
using namespace nvcuda;
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

// SQA parameters
#define N 32 
#define M 16 

#define TIMES 10
#define STEP 100
#define THREADS 64

// Must be multiples of 16
#define MATRIX_M 32
#define MATRIX_K 32
#define MATRIX_N 16

// Error check macros
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert (cudaError_t code, const char *file, int line, bool abort=true){
   if (code != cudaSuccess){
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// CPU functions
void usage ();
void construct_spin(float *spin, int total_spins);
void construct_delta_H(cublasHandle_t cublasHandle, float *couplings_fp32, float *spin_fp32, float *delta_H_fp32);
void check(float *delta_H);

int main(int argc, char* argv[]) {
	if (argc != 2) 
		usage();
    
    //Initialize TC, for check
    cublasHandle_t cublasHandle;
    cudaEvent_t startcublas;
    cudaEvent_t stopcublas;

    cudaErrCheck(cudaEventCreate(&startcublas));
    cudaErrCheck(cudaEventCreate(&stopcublas));
    cublasErrCheck(cublasCreate(&cublasHandle));
	
	// Initialize couplings
    float *couplings; // cpu    
    couplings = (float*)malloc(N * N * sizeof(float));
    memset(couplings, 0, N*N*sizeof(float));
    
    float *couplings_fp32; // tc-32
    cudaErrCheck(cudaMalloc((void**)&couplings_fp32, N * N * sizeof(float)));
    
	/*// Read files
    FILE *instance = fopen(argv[1], "r");
    assert(instance != NULL);
    int a, b, w, total_spins;
    fscanf(instance, "%d%d", &total_spins, &b);
    while (!feof(instance)) {
        fscanf(instance, "%d%d%d", &a, &b, &w);
        assert(a != b);
        a--;
        b--;
        couplings[a * N + b] = w;
        couplings[b * N + a] = w;
    }
    fclose(instance);*/
    //testing
    int total_spins =  30;
    for (int i = 0; i < N; i++){
    	for (int j = 0; j < N; j++){
    		couplings[i*N+j] = 1;
    	}
    }

    // copy couplings to target device
    cudaErrCheck ( cudaMemcpy(couplings_fp32, couplings, N*N*sizeof(float), cudaMemcpyHostToDevice) );
    printf("couplings:\n");
	for (int i = 0; i < N; i++){
		for (int k = 0; k < N; k++){
			printf("%d ",(int)couplings[k + i*N]);
		}
		printf("\n");
	}
	// Initialize spin
    float *spin;
    spin = (float*)malloc(M*N*sizeof(float));
    memset(spin, 0, M*N*sizeof(float)); // must initialize, since there are some places not 0
    float *spin_fp32;
    cudaErrCheck ( cudaMalloc((void**)&spin_fp32, M*N*sizeof(float)) );

    float *delta_H;
    delta_H = (float*)malloc(M*N*sizeof(float));
    memset(delta_H, 0, M*N*sizeof(float));
    
    float *delta_H_fp32;
    cudaErrCheck(cudaMalloc((void**)&delta_H_fp32, M*N*sizeof(float)) );
    cudaErrCheck (cudaMemcpy(delta_H_fp32, delta_H, M*N*sizeof(float), cudaMemcpyHostToDevice) );

	// TC, using tensor core
    cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH)); 
    
    // Parameters init
    float results[TIMES] = {0.};
    int delta;
    float increase = (8 - 1/(float)16) / (float)STEP;
    float G0 = 8.;
    
    //testing:
    printf("construct_spin...\n");
    construct_spin(spin, total_spins);
    
    cudaErrCheck (cudaMemcpy(spin_fp32, spin, M*N*sizeof(float), cudaMemcpyHostToDevice));
    // CPU counting delta_H correctness
    construct_delta_H(cublasHandle,couplings_fp32, spin_fp32, delta_H_fp32);
    cudaErrCheck ( cudaMemcpy(delta_H, delta_H_fp32, M*N*sizeof(float), cudaMemcpyDeviceToHost) );
    check(delta_H);

    // Release Objects
    free(couplings);
    free(spin);
    //free(hamiltonion);
    cudaFree(couplings_fp32);
    cudaFree(spin_fp32);
    //cudaFree(hamiltonion_fp32);
    //cudaFree(hamiltonion_tmp);
	return 0;
}

void usage (){
    printf("Usage:\n");
    printf("       ./sqa [spin configuration]\n");
    exit(0);
}

void construct_spin(float *spin, int total_spins){
	/*float x;
	for(int m = 0; m < M; m++){
		for(int n = 0; n < total_spins; n++){
			x = ((float)rand()/(float)(RAND_MAX)) * 1.0;	
			spin[m*M+n] = ((x>=0.5) ? (float)1. : (float)-1.);
		}
	}*/
	for(int n = 0; n < total_spins; n++){
		for(int m = 0; m < M; m++){
			spin[IDX2C(n,m,N)] = n*M+m+1;

			//printf("%d \n", n*M+m+1);
		}
	}
	
	printf("\nconstruct_spin:\n");
    for (int n = 0; n < N; n++){
	    for(int m = 0; m < M; m++){
    		printf("%f ", spin[IDX2C(n,m,N)] );
    	}
    	printf("\n");
    }
}

void construct_delta_H(cublasHandle_t cublasHandle, float *couplings_fp32, float *spin_fp32, float *delta_H_fp32){
    float alpha_tc = 1.0f, beta_tc = 0.0f;
	for (int m = 0; m < M; m++){
		for(int n = 0; n < N; n++){
			/*cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
			            				MATRIX_M, MATRIX_N, MATRIX_K, 
				         				&alpha_tc,
				          				couplings_fp32, CUDA_R_32F, MATRIX_M,
			                			spin_fp32, CUDA_R_32F, MATRIX_K,
			                			&beta_tc, 
			               				delta_H_fp32, CUDA_R_32F, MATRIX_M,
			               				CUDA_R_32F, CUBLAS_GEMM_DEFAULT));*/
			cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
			            				MATRIX_M, MATRIX_N, MATRIX_K, 
				         				&alpha_tc,
				          				couplings_fp32, CUDA_R_32F, MATRIX_M,
			                			spin_fp32, CUDA_R_32F, MATRIX_K,
			                			&beta_tc, 
			               				delta_H_fp32, CUDA_R_32F, MATRIX_M,
			               				CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
		}
	}  
}

void check (float* delta_H){
	printf("\ncheck..., print delta_H\n");

    for (int n = 0; n < N; n++){
    	for (int m = 0; m < M; m++){
    		printf("%f ", delta_H[IDX2C(n,m,N)]);
    	}
    	printf("\n");
    }
}

