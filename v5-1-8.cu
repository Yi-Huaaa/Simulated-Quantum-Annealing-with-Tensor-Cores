/*
和 517差別：
(1) 16x16 -> 128x16 (調參數, 可以改M_2來調參數)
(2) thread <<<16,1>>> -> <<<16, 128>>>
                             ^   ^
                             |   |
                layer在不同SM做    \
                                每個layer有128 worker做加法
(3) async copy log random number using stream 
           ______________________________ ________________________________
stream 1: |Memcpy MxN log(rand()), step=0| Memcpy MxN log(rand()), step=1 | ...
           -----------------------------------------------------------------
stream 2:   | judge flip then Sgemm, step=0 | judge flip then Sgemm, step=1 | ...   
             ---------------------------------------------------------------

G1 800 spins
time from 0.09s -> 0.0098s (100step)
          9.0ms -> 0.098ms (per step)

G22 2000 spins
time from 0.21s -> 0.038s (100step)
          2.1ms -> 0.38ms (per step)

G48 3000 spins
time from 0.44s -> 0.12s (100step)
          4.4ms -> 1.2ms (per step)

G65 8000 spins
time from 0.91s -> 0.28s (100step)
          9.1ms -> 2.8ms (per step)

G77 14000 spins
time from 1.95s  -> 0.66s (100step)
          19.5ms -> 6.6ms (per step)

G81 20000 spins
time from 4.44s  -> 1.88s  (100step)
          44.4ms -> 18.8ms (per step)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>
#include "cuda_profiler_api.h"
#include <cublas_v2.h>
#include <mma.h>
#include <stdbool.h>
using namespace nvcuda;
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

// SQA parameters
#define N 32768
#define M 128
#define M_2 16

#define TIMES 10
#define STEP 100

// Must be multiples of 16
#define MATRIX_M N
#define MATRIX_K N
#define MATRIX_N M


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
void usage ();
void check_spin(float *spin);
void check_couplings(float *couplings);
void check_delta_H (float *couplings, float *spin, float *delta_H, float *delta_H_fp32);
void check_matrix_B (float *matrix_B, float *matrix_B_fp32);

void construct_spin(float *spin, float *spin_fp32,int total_spins){
    float x;
    for (int n = 0; n < N; n++){
        for(int m = 0; m < M; m++){
            x = ((float)rand()/(float)(RAND_MAX)) * 1.0;    
            spin[IDX2C(n,m,N)] = ((x>=0.5) ? (float)1. : (float)-1.);
        }
    }
    cudaErrCheck (cudaMemcpy(spin_fp32, spin, M*N*sizeof(float), cudaMemcpyHostToDevice));
}

void construct_rand_val(float *rand_val, float *rand_val_fp32){
    for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
            rand_val[IDX2C(i,j,N)] = ((float)rand()/(float)(RAND_MAX)) * 1.0;
        }
    }
    cudaErrCheck (cudaMemcpy(rand_val_fp32, rand_val, M*N*sizeof(float), cudaMemcpyHostToDevice));
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

void update_delta_H(cublasHandle_t cublasHandle, float *couplings_fp32, float *matrix_B_fp32, float *delta_H_fp32, int which_spin){
    float alpha = 1.0f, beta = 1.0f;    
    int blk_num = which_spin / M_2;
    int coup_idx = blk_num * (N * M_2);
    cublasErrCheck(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                MATRIX_M, MATRIX_N, M_2,
                                &alpha, 
                                couplings_fp32 + coup_idx, MATRIX_M,
                                matrix_B_fp32, M_2,
                                &beta,
                                delta_H_fp32, MATRIX_M));
}

void construct_lograndval(float *log_rand_val, float *log_rand_val_fp32, cudaStream_t stream){
    for(int i = 0; i < N/2; i++){
        for(int j = 0; j < M; j++){
            log_rand_val[IDX2C(i,j,N)] = (-log(((float)rand()/(float)(RAND_MAX)) * 1.0));
        }
    }
    for(int i = 0; i  < N/2; i++){
        for(int j = 0; j < M; j++){
            log_rand_val[IDX2C(i+N/2,j,N)] = log_rand_val[IDX2C(i,j,N)];
        }
    }
    cudaErrCheck (cudaMemcpyAsync(log_rand_val_fp32, log_rand_val, M*N*sizeof(float), cudaMemcpyHostToDevice, stream));
}

int calculate_E (float *spin, float *spin_fp32, float *couplings){
    cudaErrCheck(cudaMemcpy(spin, spin_fp32, N*sizeof(float), cudaMemcpyDeviceToHost));
    int E = 0;
    for (int i = 0; i < N; i++){
        for (int j = i+1; j < N; j++){
            E += -spin[IDX2C(i,0,N)]*spin[IDX2C(j,0,N)]*couplings[IDX2C(i,j,N)];
        }
    }
    return E;
}

__global__ void judge_flipping_com (float *couplings_fp32,float *delta_H_fp32, float *spin_fp32, float *matrix_B_fp32, float *log_rand_val_fp32, int J_perp, float beta, int start_spin){
    int m = blockIdx.x;
    int idx, mb_idx, upper, lower;
    float delta;
    int first_rd_idx = m&1; //even:0, odd:1
    
    extern __shared__ float deltas[];
    deltas[threadIdx.x] = delta_H_fp32[IDX2C(start_spin+threadIdx.x, m, N)];

    // even: 0~M_2/2-1; odd: M_2/2~M_2-1
    for (int n = 0; n < M_2; n++) {
        int nn = start_spin + ((first_rd_idx*(M_2/2) + n)&(M_2-1));
        idx = IDX2C(nn,m,N);
        mb_idx = IDX2C(nn&(M_2-1),m,M_2);            

        delta = deltas[nn&(M_2-1)];

        upper = (m == 0 ? M-1 : m-1);
        lower = (m == M-1 ? 0 : m+1);
        delta = 2*M*spin_fp32[idx]*(delta - M*J_perp*(spin_fp32[IDX2C(nn,upper,N)] + spin_fp32[IDX2C(nn,lower,N)]));
        delta = delta * beta;
        matrix_B_fp32[mb_idx] = 0;
        if ( (log_rand_val_fp32[idx]) > delta ) {
            spin_fp32[idx] = -spin_fp32[idx];
            matrix_B_fp32[mb_idx] = 2*spin_fp32[idx];
            int ii = start_spin + threadIdx.x;
            deltas[threadIdx.x] += couplings_fp32[IDX2C(ii,nn,N)]*matrix_B_fp32[mb_idx]; 
        }
        __syncthreads();
    }
}

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
    cudaErrCheck(cudaMalloc((void**)&couplings_fp32, N*N*sizeof(float)));
    
    // Read files
    FILE *instance = fopen(argv[1], "r");
    assert(instance != NULL);
    int a, b, w, total_spins, total_couplings;
    fscanf(instance, "%d%d", &total_spins, &total_couplings);
    while (total_couplings --) {
        fscanf(instance, "%d%d%d", &a, &b, &w);
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
    cudaErrCheck ( cudaMalloc((void**)&spin_fp32, M*N*sizeof(float)) );
    cudaErrCheck(cudaMemcpy(spin_fp32, spin, M*N*sizeof(float), cudaMemcpyHostToDevice));

    float *delta_H;
    delta_H = (float*)malloc(M*N*sizeof(float));
    memset(delta_H, 0, M*N*sizeof(float));
    
    float *delta_H_fp32;
    cudaErrCheck(cudaMalloc((void**)&delta_H_fp32, M*N*sizeof(float)));
    cudaErrCheck(cudaMemcpy(delta_H_fp32, delta_H, M*N*sizeof(float), cudaMemcpyHostToDevice));

    float *matrix_B;
    matrix_B = (float*)malloc(M*M_2*sizeof(float));
    memset(matrix_B, 0, M*M_2*sizeof(float));

    float *matrix_B_fp32;
    cudaErrCheck(cudaMalloc((void**)&matrix_B_fp32, M*M_2*sizeof(float)));
    cudaErrCheck(cudaMemcpy(matrix_B_fp32, matrix_B, M*M_2*sizeof(float), cudaMemcpyHostToDevice));
    
    float *log_rand_val;
    cudaErrCheck(cudaMallocHost((void**)&log_rand_val, M*N*sizeof(float)));
    // log_rand_val = (float*)malloc(M*N*sizeof(float));
    
    float *log_rand_val_fp32;
    cudaErrCheck(cudaMalloc((void**)&log_rand_val_fp32, M*N*sizeof(float)));
    
    // TC, using tensor core
    cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH)); 
    
    // Parameters init
    float results[TIMES] = {0.};
    float used_time[TIMES] = {0.};
    float increase = (8 - 1/(float)16) / (float)STEP;
    float G0 = 8.;

    cudaStream_t stream1, stream2;
    cudaErrCheck(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
    cudaErrCheck(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));
    cublasErrCheck(cublasSetStream(cublasHandle, stream2));
    
    
    for (int t = 0; t < TIMES; t++) {
        float beta = 1/(float)16; //bete = 1/Time
        
        //init spin
        construct_spin(spin, spin_fp32,total_spins);
        construct_delta_H(cublasHandle,couplings_fp32, spin_fp32, delta_H_fp32);
            
        // Current cost time
        clock_t begin, end;

        begin = clock();
        for (int p = 0; p < STEP; p++) {
            
            float Gamma = G0*(1.-(float)p/(float)STEP);
            float J_perp = -0.5*log(tanh((Gamma/M)*beta))/beta;
            
            construct_lograndval(log_rand_val, log_rand_val_fp32, stream1);
            for (int n = 0; n < N; n += M_2) {
                judge_flipping_com <<< M, M_2, 16*sizeof(float), stream2 >>> (couplings_fp32, delta_H_fp32, spin_fp32, matrix_B_fp32, log_rand_val_fp32, J_perp, beta, n);
                update_delta_H(cublasHandle, couplings_fp32, matrix_B_fp32, delta_H_fp32, n);              
            }
	    cudaDeviceSynchronize();
            beta += increase;
            
            //printf("curr: %10lf, energy: %10d\n", curr, E);
        } 
	cudaDeviceSynchronize();
        end = clock();
        double duration = (double)(end-begin) / CLOCKS_PER_SEC;
            
        used_time[t] = duration;
        
	int E = calculate_E(spin, spin_fp32, couplings);
        results[t] = E;
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
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    free(couplings);
    free(spin);
    free(delta_H);
    free(matrix_B);
    cudaFreeHost(log_rand_val);
    cudaFree(couplings_fp32);
    cudaFree(spin_fp32);
    cudaFree(delta_H_fp32);
    cudaFree(matrix_B_fp32);
    cudaFree(log_rand_val_fp32);
    
    return 0;
}

void usage (){
    printf("Usage:\n");
    printf("       ./sqa [spin configuration]\n");
    exit(0);
}

void check_spin(float *spin){
    printf("\ncheck_spin:\n");
    for (int n = 0; n < N; n++){
        for(int m = 0; m < M; m++){
            printf("%d ", (int)spin[IDX2C(n,m,N)] );
        }
        printf("\n");
    }
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

void check_delta_H (float *couplings, float *spin, float *delta_H, float *delta_H_fp32){
    cudaErrCheck ( cudaMemcpy(delta_H, delta_H_fp32, M*N*sizeof(float), cudaMemcpyDeviceToHost));
    printf("check..., print delta_H\n");
    for (int n = 0; n < N; n++){
        for (int m = 0; m < M; m++){
            printf("%d ", (int)delta_H[IDX2C(n,m,N)]);
        }
        printf("\n");
    }
}

void check_matrix_B (float *matrix_B, float *matrix_B_fp32){
    cudaErrCheck(cudaMemcpy(matrix_B, matrix_B_fp32, M*N*sizeof(float), cudaMemcpyDeviceToHost));
    printf("check..., matrix_B:\n");
    for (int n = 0; n < N; n++){
        for (int m = 0; m < M; m++){
            printf("%d ", (int)matrix_B[IDX2C(n,m,N)]);
        }
        printf("\n");
    }
}
