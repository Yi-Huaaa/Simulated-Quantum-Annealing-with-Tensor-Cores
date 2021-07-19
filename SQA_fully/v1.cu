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

// SQA paraneters
#define N 1024
#define THREADS 64
#define TIMES 10
#define M 16  // trotter layers
#define STEP 100

// Must be multiples of 16 for wmma code to work
#define MATRIX_M 1024
#define MATRIX_N 1024
#define MATRIX_K 1024

// Error checking macros
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

// GPU funcitons
__global__ void calcualte_Ham (int ham_gpu, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      ham_gpu += in[idx];
   }
}


// CPU functions
void usage ();
void construct_spin(float *spin, int total_spins);

int main(int argc, char* argv[]) {
	if (argc != 2) 
		usage();
    
    //Initialize TC, for checking
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
    
	// Read files
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
    fclose(instance);

    // copy couplings to target device
    cudaErrCheck ( cudaMemcpy(couplings_fp32, couplings, N*N*sizeof(float), cudaMemcpyHostToDevice) );
    //printf("couplings:\n");
	/*for (int i = 0; i < N; i++){
		for (int k = 0; k < N; k++){
			printf("%d ",(int)couplings[k + i*N]);
		}
		printf("\n");
	}*/
 	printf("couplings ending\n");
	// Initialize spins
    float *spin;
    spin = (float*)malloc(M*N*N*sizeof(float));
    memset(spin, 0, M*N*N*sizeof(float)); // must initialize, since there are some places not 0

    float *spin_fp32;
    cudaErrCheck ( cudaMalloc((void**)&spin_fp32, M*N*N*sizeof(float)) );

    // Hamiltonion
    float *hamiltonion;
    hamiltonion = (float*)malloc(N*N*sizeof(float));
    float *hamiltonion_fp32, *hamiltonion_tmp;
    cudaErrCheck ( cudaMalloc((void**)&hamiltonion_fp32, N*N*sizeof(float)) );
    cudaErrCheck ( cudaMalloc((void**)&hamiltonion_tmp,  N*N*sizeof(float)) );

	// TC, using tensor core
    cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH)); 
    
    // Parameters init
    float results[TIMES] = {0.};
    int delta;
    float increase = (8 - 1/(float)16) / (float)STEP;
    float G0 = 8.;
    float ham_instantaneous = 0., ham_first = 0.;

    
    for (int t = 0; t < TIMES; t++) {
        float beta = 1/(float)16; //bete = 1/Time
        
        //init spins
        construct_spin(spin, total_spins);
    	cudaErrCheck ( cudaMemcpy(spin_fp32, spin, M*N*N*sizeof(float), cudaMemcpyHostToDevice) );
    	
        //init hamiltonion
        float alpha_tc = 1.0f, beta_tc = 0.0f;
	   	// 這裡要非常小心~~~~~~~~~~~~~~~~~~~要反著放 Bt *At才會得到所求
	   	cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, 
	               					MATRIX_M, MATRIX_N, MATRIX_K, 
	               					&alpha_tc,
	              					spin_fp32, CUDA_R_32F, MATRIX_M,
	                				couplings_fp32, CUDA_R_32F, MATRIX_K,
	                				&beta_tc, 
	                				hamiltonion_fp32, CUDA_R_32F, MATRIX_M,
	                				CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
	   	//cudaErrCheck ( cudaMemcpy(hamiltonion, hamiltonion_fp32, N*N*sizeof(float), cudaMemcpyDeviceToHost) );
	   	/*printf("calcualte_Ham transpose:\n");
		for (int i = 0; i < N; i++){
			for (int k = 0; k < N; k++){
				printf("%d ",(int)hamiltonion[k + i*N]);
			}
			printf("\n");
		}*/
	   	cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N,CUBLAS_OP_T,
	                				MATRIX_M, MATRIX_N, MATRIX_K, 
	                				&alpha_tc,
	               					hamiltonion_fp32, CUDA_R_32F, MATRIX_M,
	                				spin_fp32, CUDA_R_32F, MATRIX_K,
	                				&beta_tc, 
	               					hamiltonion_tmp, CUDA_R_32F, MATRIX_M,
	               					CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));	 // 前面的矩陣要transpose 
	     		
	   	cudaErrCheck ( cudaMemcpy(hamiltonion, hamiltonion_tmp, N*N*sizeof(float), cudaMemcpyDeviceToHost) );
	   	/*printf("calcualte_Ham final:\n");
		for (int i = 0; i < N; i++){
			for (int k = 0; k < N; k++){
				printf("%d ",(int)hamiltonion[k + i*N]);
			}
			printf("\n");
		}*/
		for(int i = 0; i < N*N; i++)
			ham_first += hamiltonion[i];
		ham_first *= M;
		//printf("Firstly hamiltonion = %f\n", ham_first);

        // Currrent cost-time
        double curr = 0.;
        ham_instantaneous = ham_first;
        for (int p = 0; p < STEP; p++) {
            float Gamma = G0*(1.-(float)p/(float)STEP);
            float J_perp = -0.5*M*log(tanh((Gamma/M)*beta))/beta; //把下面要乘的M移動上來這裡乘
            clock_t begin = clock();
            int Big_N = N*N;

            for (int m = 0; m < M; m++) {
                for (int n = 0; n < Big_N; n+=N) {
                    int idx = Big_N*m + n;
                    int upper = (m == 0 ? M-1 : m-1);
                    int lower = (m == M-1 ? 0 : m+1);
                    // count delta Hamiltonion, 2 parts
                    delta = 0;
                    // parta. TC part
                    spin[idx] = -(float)spin[idx];
                    cudaErrCheck ( cudaMemcpy(spin_fp32+idx, spin+idx, 1*sizeof(float), cudaMemcpyHostToDevice) );
				   	cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, 
				               					MATRIX_M, MATRIX_N, MATRIX_K, 
				               					&alpha_tc,
				              					spin_fp32 + Big_N*m, CUDA_R_32F, MATRIX_M,
				                				couplings_fp32,  CUDA_R_32F, MATRIX_K,
				                				&beta_tc, 
				                				hamiltonion_fp32, CUDA_R_32F, MATRIX_M,
				                				CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
				   	
				   	cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N,CUBLAS_OP_N,
				                				MATRIX_M, MATRIX_N, MATRIX_K, 
				                				&alpha_tc,
				               					hamiltonion_fp32, CUDA_R_32F, MATRIX_M,
				                				spin_fp32 + Big_N*m, CUDA_R_32F, MATRIX_K,
				                				&beta_tc, 
				               					hamiltonion_tmp, CUDA_R_32F, MATRIX_M,
				               					CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));	 // 前面的矩陣要transpose 
				     		
				   	cudaErrCheck ( cudaMemcpy(hamiltonion, hamiltonion_tmp, N*N*sizeof(float), cudaMemcpyDeviceToHost) );
					int tmp_ham = 0;
					for(int i = 0; i < N*N; i++)
						tmp_ham += hamiltonion[i];
                    
                    // partb. 直接用CPU加上下兩層的東東
                    delta = 2*(tmp_ham + (J_perp * spin[idx] * (spin[Big_N*lower + n] + spin[Big_N*upper + n])));
                    
                    if ( (-log(rand() / (float) RAND_MAX) / beta) > delta ) { //近來代表不翻，而且還要翻回去
                       	spin[idx] = -spin[idx];
                    	cudaErrCheck ( cudaMemcpy(spin_fp32+idx, spin+idx, 1*sizeof(float), cudaMemcpyHostToDevice) );
                    	ham_instantaneous += delta; // 新 = delta + 舊
                    }
                }
                //printf("111\n");
            }
            beta += increase;
            clock_t end = clock();
            double duration = (double)(end-begin) / CLOCKS_PER_SEC;
            curr += duration;
            results[t] -= ham_instantaneous;
            printf("time: %d, step: %d ,curr_time: %10lf, hamiltonion: %f\n", t, p, curr, results[t]);
            
        }  
    }
    


    // Release Objects
    free(couplings);
    free(spin);
    free(Hamiltonion);
    free(sigma);
    cudaFree(couplings_fp32);
    cudaFree(spin_fp32);
    cudaFree(hamiltonion_fp32);
    cudaFree(hamiltonion_tmp);
	return 0;
}

void usage (){
    printf("Usage:\n");
    printf("       ./sqa [spin configuration]\n");
    exit(0);
}
void construct_spin(float *spin, int total_spins){
	float x;
	int layer = N*N;
	for (int i = 0; i < total_spins; i ++){ // 實際上只需要做到 total_spins的個數，後面則全部填成0
		for(int j = 0; j < M; j++){
			x = ((float)rand()/(float)(RAND_MAX)) * 1.0;
			//printf("x = %f\n", x);
			if(x >= 0.5){
				spin[layer*j + N*i + i] = (float) 1.;
			}else{
				spin[layer*j + N*i + i] = (float) -1.;
			}
		}
	}
	//printf("construct_spin:\n");
	//for(int j = 0 ; j < M; j++){
		//printf("M = %d\n", j);
		/*for (int i = 0; i < N; i++){
			for (int k = 0; k < N; k++){
				printf("%d ",(int)spin[k + i*N]);
			}
			printf("\n");
		}
		printf("\n");*/
	//}
	printf("construct_spin end\n");
}
