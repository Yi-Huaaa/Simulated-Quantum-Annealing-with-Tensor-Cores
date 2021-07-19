//GPU version vs cublas
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

#define TIMES 1//10
#define STEP 100 //100

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

// GPU funciton
__global__ void judge_flipping (float *delta_H_fp32, float *spin_fp32, float *rand_val_fp32, int J_perp, float beta, int n){
    int idx = 0, upper = 0, lower = 0;
    float delta = 0.;
    for (int m = 0; m < M; m++) {
        idx = IDX2C(n,m,N);
        delta = delta_H_fp32[idx];
        upper = (m == 0 ? M-1 : m-1);
        lower = (m == m-1 ? 0 : m+1);
        delta = 2*M*spin_fp32[idx]*(delta - M*J_perp*(spin_fp32[IDX2C(n,upper,N)] + spin_fp32[IDX2C(n,lower,N)]));
        if ( (-log(rand_val_fp32[idx]) / beta) > delta ) {
            spin_fp32[idx] = -spin_fp32[idx];
            //gpuErrchk( cudaMemcpy (spin_fp32+idx, spin+idx, 1*sizeof(float), cudaMemcpyHostToDevice));
            //printf("flipping: n = %d, m = %d\n", n, m);
        }
    }
}

// CPU functions
void usage ();
void construct_spin(float *spin, int total_spins);
void check_spin (float *spin, int total_spins);
void construct_delta_H(cublasHandle_t cublasHandle, float *couplings_fp32, float *spin_fp32, float *delta_H_fp32);
void construct_rand_val(float *rand_val, float *rand_val_fp32);
void check_delta_H(float *couplings, float *spin, float *spin_fp32,float *delta_H, float *delta_H_fp32);
int calculate_E(float *spin, float *spin_fp32, float *couplings);

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
    
    // Read files
    FILE *instance = fopen(argv[1], "r");
    assert(instance != NULL);
    int a, b, w, total_spins, total_couplings;
    fscanf(instance, "%d%d", &total_spins, &total_couplings);
    while (total_couplings --) {
        fscanf(instance, "%d%d%d", &a, &b, &w);
        //assert(a != b);
        a--;
        b--;
        couplings[IDX2C(a,b,N)] = w;
        couplings[IDX2C(b,a,N)] = w;
    }
    fclose(instance);

    // copy couplings to target device
    cudaErrCheck ( cudaMemcpy(couplings_fp32, couplings, N*N*sizeof(float), cudaMemcpyHostToDevice) );

	// Initialize spin
    float *spin;
    spin = (float*)malloc(M*N*sizeof(float));
    memset(spin, 0, M*N*sizeof(float)); // must initialize, since there are some places not 0
    
    float *spin_fp32;
    cudaErrCheck (cudaMalloc((void**)&spin_fp32, M*N*sizeof(float)));
    cudaErrCheck(cudaMemcpy(spin_fp32, spin, M*N*sizeof(float), cudaMemcpyHostToDevice));

    float *delta_H;
    delta_H = (float*)malloc(M*N*sizeof(float));
    memset(delta_H, 0, M*N*sizeof(float));
    
    float *delta_H_fp32;
    cudaErrCheck(cudaMalloc((void**)&delta_H_fp32, M*N*sizeof(float)));
    cudaErrCheck (cudaMemcpy(delta_H_fp32, delta_H, M*N*sizeof(float), cudaMemcpyHostToDevice));

    float *rand_val;
    rand_val = (float*)malloc(M*N*sizeof(float));
    memset(rand_val, 0, M*M*sizeof(float));

    float *rand_val_fp32;
    cudaErrCheck(cudaMalloc((void**)&rand_val_fp32, M*N*sizeof(float)));
    cudaErrCheck(cudaMemcpy(rand_val_fp32, rand_val, M*N*sizeof(float), cudaMemcpyHostToDevice));

	// TC, using tensor core
    cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH)); 
    
    // Parameters init
    float results[TIMES] = {0.};
    float used_time[TIMES] = {0.};
    //float delta = 0.;
    float increase = (8 - 1/(float)16) / (float)STEP;
    float G0 = 8.;
    
    
    for (int t = 0; t < TIMES; t++) {
        float beta = 1/(float)16; //bete = 1/Time
        
        //init spin
        construct_spin(spin, total_spins);
        cudaErrCheck (cudaMemcpy(spin_fp32, spin, M*N*sizeof(float), cudaMemcpyHostToDevice));        
        // Current cost-time
        double curr = 0.;
        int E1 = calculate_E(spin, spin_fp32, couplings);
        printf("Init: Energy = %d\n",E1);
        
        for (int p = 0; p < STEP; p++) {
            
            construct_rand_val(rand_val, rand_val_fp32);
            
            float Gamma = G0*(1.-(float)p/(float)STEP);
            float J_perp = -0.5*log(tanh((Gamma/M)*beta))/beta;
            clock_t begin = clock();
            
            for (int n = 0; n < N; n++) { //N, M 對調
                // count delta
                construct_delta_H(cublasHandle,couplings_fp32, spin_fp32, delta_H_fp32);
                // update
                judge_flipping <<< 1,1,0 >>> (delta_H_fp32, spin_fp32, rand_val_fp32, J_perp, beta, n);
            }
            
            beta += increase;
            clock_t end = clock();
            double duration = (double)(end-begin) / CLOCKS_PER_SEC;
            curr += duration;

            // random choose one layer to calculate
            int E = calculate_E(spin, spin_fp32, couplings);
            results[t] = E;
            used_time[t] = curr;
            //printf("STEP = %d, Energy = %d\n", p,E);

            
        } 
    }
    
    printf("Final: \n");
    for (int t = 0; t < TIMES; t++){
        printf("TIME: %d,  used time (s): %10lf,  Energy: %10lf\n", t, used_time[t], results[t]);
    }
    

    float tot_result_time = 0., tot_energy = 0.;
    for(int i = 0; i < TIMES; i++){
        tot_result_time += used_time[i];
        tot_energy += results[i];
    }
    printf("\nAvg time  : %f\n", tot_result_time/TIMES);
    printf("Avg energy: %f\n", tot_energy/TIMES);

    cublasDestroy(cublasHandle);   
    free(couplings);
    free(spin);
    free(delta_H);
    cudaFree(couplings_fp32);
    cudaFree(spin_fp32);
    cudaFree(delta_H_fp32);
    return 0;
}

void usage (){
    printf("Usage:\n");
    printf("       ./sqa [spin configuration]\n");
    exit(0);
}

void construct_spin(float *spin, int total_spins){
	float x;
    for (int n = 0; n < N; n++){
        for(int m = 0; m < M; m++){
            x = ((float)rand()/(float)(RAND_MAX)) * 1.0;    
            spin[IDX2C(n,m,N)] = ((x>=0.5) ? (float)1. : (float)-1.);
        }
    }
}

void check_spin(float *spin){
    printf("check_spin:\n");
    for (int n = 0; n < N; n++){
        for(int m = 0; m < M; m++){
            printf("%d ", (int)spin[IDX2C(n,m,N)] );
        }
        printf("\n");
    }
}
void construct_delta_H(cublasHandle_t cublasHandle, float *couplings_fp32, float *spin_fp32, float *delta_H_fp32){
    float alpha = 1.0f, beta = 0.0f;
    cublasErrCheck(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                MATRIX_M, MATRIX_N, MATRIX_K,
                                &alpha, 
                                couplings_fp32, MATRIX_M,
                                spin_fp32, MATRIX_K,
                                &beta,
                                delta_H_fp32, MATRIX_M));

}

void construct_rand_val(float *rand_val, float *rand_val_fp32){
    srand(time(0));
    for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
            rand_val[IDX2C(i,j,N)] = ((float)rand()/(float)(RAND_MAX)) * 1.0;
        }
    }
    cudaErrCheck (cudaMemcpy(rand_val_fp32, rand_val, M*N*sizeof(float), cudaMemcpyHostToDevice));
}

void check_couplings(float *couplings){
    printf("\ncheck_couplings:\n");
    for (int n = 0; n < N; n++){
        for(int k = 0; k < N; k++){
            printf("%d ", (int)couplings[IDX2C(n,k,N)] );
        }
        printf("\n");
    }

}

void check_delta_H (float *couplings, float *spin, float *spin_fp32, float *delta_H, float *delta_H_fp32){
    cudaErrCheck ( cudaMemcpy(delta_H, delta_H_fp32, M*N*sizeof(float), cudaMemcpyDeviceToHost) );
    check_couplings(couplings);
    check_spin(spin);
    printf("\ncheck..., print delta_H\n");
    for (int n = 0; n < N; n++){
    	for (int m = 0; m < M; m++){
    		printf("%d ", (int)delta_H[IDX2C(n,m,N)]);
    	}
    	printf("\n");
    }
}


int calculate_E (float *spin, float *spin_fp32, float *couplings){
    cudaErrCheck(cudaMemcpy(spin, spin_fp32, M*N*sizeof(float), cudaMemcpyDeviceToHost));
    int E = 0;
    for (int i = 0; i < N; i++){
        for (int j = i+1; j < N; j++){
            E += -spin[IDX2C(i,0,N)]*spin[IDX2C(j,0,N)]*couplings[IDX2C(i,j,N)];
        }
    }
    return E;
}
