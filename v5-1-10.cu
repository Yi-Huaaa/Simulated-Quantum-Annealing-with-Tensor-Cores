/*
diff：
(1) float --> half

(ms)/per step

1024  2048  4096  8192 16384 32768
0.3   0.65  1.47  2.89  7.30  20.6
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
#include <cooperative_groups.h>
using namespace nvcuda;
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define USE_TENSOR_CORE

// SQA parameters
#define N 8192
#define M 16
#define M_2 256

#define TIMES 10
#define STEP 100
#define WORKER 8

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

__global__ void construct_delta_H(half *couplings_fp16, float *spin_fp32, float *delta_H_fp32){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    delta_H_fp32[idx] = 0;
    for (int m = 0; m < M; m++)
        for (int i = 0; i < N; i++)
            delta_H_fp32[IDX2C(idx,m,N)] += (float)couplings_fp16[IDX2C(i,idx,N)]*spin_fp32[IDX2C(i,m,N)];
}

void update_delta_H(cublasHandle_t cublasHandle, half *couplings_fp16, half *matrix_B_fp16, float *delta_H_fp32, int which_spin){
    float alpha = 1.0f, beta = 1.0f;    
    int blk_num = which_spin / M_2;
    int coup_idx = blk_num * (N * M_2);
    cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                MATRIX_M, MATRIX_N, M_2,
                                &alpha, 
                                couplings_fp16 + coup_idx, CUDA_R_16F, MATRIX_M,
                                matrix_B_fp16, CUDA_R_16F, M_2, 
                                &beta, 
                                delta_H_fp32, CUDA_R_32F, MATRIX_M,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void construct_lograndval(float *log_rand_val, float *log_rand_val_fp32, cudaStream_t stream){
    for(int i = 0; i < N; i++){
        log_rand_val[IDX2C(i,0,N)] = (-log(((float)rand()/(float)(RAND_MAX)) * 1.0));
    }
    for (int m = M-1; m >= 1; m--)
        memcpy(&log_rand_val[m*N], &log_rand_val[(m-1)*N], N*sizeof(float));
    cudaErrCheck (cudaMemcpyAsync(log_rand_val_fp32, log_rand_val, M*N*sizeof(float), cudaMemcpyHostToDevice, stream));
}

int calculate_E (float *spin, float *spin_fp32, half *couplings){
    cudaErrCheck(cudaMemcpy(spin, spin_fp32, N*sizeof(float), cudaMemcpyDeviceToHost));
    int E = 0;
    for (int i = 0; i < N; i++){
        for (int j = i+1; j < N; j++){
            E += -spin[IDX2C(i,0,N)]*spin[IDX2C(j,0,N)]*(float)couplings[IDX2C(i,j,N)];
        }
    }
    return E;
}

__global__ void judge_flipping_com (half *couplings_fp16, float *delta_H_fp32, float *spin_fp32, half *matrix_B_fp16, 
                                    float *log_rand_val_fp32, int J_perp, float beta, int start_spin, float *deltas){
    int m = blockIdx.x/WORKER;
    int idx, mb_idx, upper, lower;
    float delta;
    int first_rd_idx = m&1; // even:0, odd:1
    int worker_id = blockIdx.x % WORKER; // partition 64 items work
    int which_delta = threadIdx.x+worker_id*(M_2/WORKER);
    int ii = start_spin + which_delta;
            
    deltas[IDX2C(which_delta, m, M_2)] = delta_H_fp32[IDX2C(start_spin+which_delta, m, N)];
    
    upper = (m-1) & (M-1);
    lower = (m+1) & (M-1);
        
    // even: 0~M_2/2-1; odd: M_2/2~M_2-1
    #pragma unroll
    for (int n = 0; n < M_2; n++) {
        int nn = start_spin + ((first_rd_idx*(M_2/2) + n)&(M_2-1));
        idx = IDX2C(nn,m,N);
        mb_idx = IDX2C(nn&(M_2-1),m,M_2);            
        
        float randnum = log_rand_val_fp32[idx];
        float s = -spin_fp32[idx];
        float up = spin_fp32[IDX2C(nn,upper,N)];
        float low = spin_fp32[IDX2C(nn,lower,N)];
        float temp = J_perp*(up + low);
        matrix_B_fp16[mb_idx] = 0;
        
        delta = deltas[IDX2C(nn&(M_2-1), m, M_2)];
        delta = beta*spin_fp32[idx]*(delta - temp);
        
        if ( randnum > delta ) {
            spin_fp32[idx] = s;
            matrix_B_fp16[mb_idx] = 2*s;
            deltas[IDX2C(which_delta, m, M_2)] += (float)couplings_fp16[IDX2C(ii,nn,N)]*(float)matrix_B_fp16[mb_idx]; 
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
    half *couplings; // cpu    
    couplings = (half*)malloc(N * N * sizeof(half));
    memset(couplings, 0, N*N*sizeof(half));
    
    half *couplings_fp16; 
    cudaErrCheck(cudaMalloc((void**)&couplings_fp16, N*N*sizeof(half)));
    
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
    cudaErrCheck ( cudaMemcpy(couplings_fp16, couplings, N*N*sizeof(half), cudaMemcpyHostToDevice) );
    
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

    half *matrix_B_fp16;
    cudaErrCheck(cudaMalloc((void**)&matrix_B_fp16, M*M_2*sizeof(float)));
    
    float *log_rand_val;
    cudaErrCheck(cudaMallocHost((void**)&log_rand_val, M*N*sizeof(float)));
    // log_rand_val = (float*)malloc(M*N*sizeof(float));
    
    float *log_rand_val_fp32;
    cudaErrCheck(cudaMalloc((void**)&log_rand_val_fp32, M*N*sizeof(float)));

    float *deltas;
    cudaErrCheck(cudaMalloc((void**)&deltas, M*M_2*sizeof(float)));
    
    // TC, using tensor core
    cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH)); 
    
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
        construct_delta_H<<<N/64, 64>>>(couplings_fp16, spin_fp32, delta_H_fp32);
            
        // Current cost time
        clock_t begin, end;

        begin = clock();
        for (int p = 0; p < STEP; p++) {
            
            float Gamma = G0*(1.-(float)p/(float)STEP);
            float J_perp = -M*0.5*log(tanh((Gamma/M)*beta))/beta;
            
            construct_lograndval(log_rand_val, log_rand_val_fp32, stream1);
            for (int n = 0; n < N; n += M_2) {
                judge_flipping_com <<< M*WORKER, M_2/WORKER, M_2*sizeof(float), stream2 >>> (couplings_fp16, delta_H_fp32, 
                    spin_fp32, matrix_B_fp16, log_rand_val_fp32, J_perp, 2*M*beta, n, deltas);
                update_delta_H(cublasHandle, couplings_fp16, matrix_B_fp16, delta_H_fp32, n);              
            }
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
    cudaFreeHost(log_rand_val);
    cudaFree(couplings_fp16);
    cudaFree(spin_fp32);
    cudaFree(delta_H_fp32);
    cudaFree(matrix_B_fp16);
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
