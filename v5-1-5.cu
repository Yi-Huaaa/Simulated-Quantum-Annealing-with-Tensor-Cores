/*
和 514差別：一次翻32*16個，開16 threads

問題：答案越大越不準
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
#define N 16384
#define M 16 
#define M_2 32

#define TIMES 1//10
#define STEP 100 //100

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
void check_rand_val(float *rand_val, float *rand_val_fp32);

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

void construct_rand_val(float *rand_val, float *rand_val_fp32){
    srand(time(0));
    for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
            rand_val[IDX2C(i,j,N)] = ((float)rand()/(float)(RAND_MAX)) * 1.0;
        }
    }
    cudaErrCheck (cudaMemcpy(rand_val_fp32, rand_val, M*N*sizeof(float), cudaMemcpyHostToDevice));
}

void construct_lograndval(float *log_rand_val, float *log_rand_val_fp32, float beta){
   //(-log(rand_val_fp32[idx]) / beta) 轉到GPU上算
    srand(time(0));
    for(int i = 0; i < M_2; i++){
        for(int j = 0; j < M; j++){
            log_rand_val[IDX2C(i,j,M_2)] = (-log(((float)rand()/(float)(RAND_MAX)) * 1.0)) / beta;
        }
    }
    cudaErrCheck (cudaMemcpy(log_rand_val_fp32, log_rand_val, M_2*M*sizeof(float), cudaMemcpyHostToDevice));
}

int calculate_E (float *spin, float *spin_fp32, float *couplings){
    //cudaErrCheck(cudaMemcpy(spin, spin_fp32, M*N*sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(spin, spin_fp32, N*sizeof(float), cudaMemcpyDeviceToHost));
    int E = 0;
    for (int i = 0; i < N; i++){
        for (int j = i+1; j < N; j++){
            E += -spin[IDX2C(i,0,N)]*spin[IDX2C(j,0,N)]*couplings[IDX2C(i,j,N)];
        }
    }
    return E;
}
__device__ void flip_com (float *couplings_fp32,float *delta_H_fp32, float *spin_fp32, float *matrix_B_fp32, float *rand_val_fp32, int J_perp, float beta, int start_spin, int m){
    int idx = 0, mb_idx = 0, upper = 0, lower = 0;
    float delta = 0.;
    int first_rd_idx = m%2; //even:0, odd:1
    int second_rd_idx = (m+1)%2; //even:1, odd:0

    // 先坐前面16個，even: 0~16; odd: 17~31
    for(int n = (start_spin + first_rd_idx * 16); n < (start_spin + 16 + first_rd_idx * 16); n++){
        idx = IDX2C(n,m,N);
        mb_idx = IDX2C(n%M_2,m,M_2);            
        delta = delta_H_fp32[idx];
        
        for(int i = start_spin + first_rd_idx*16; i < n; i++){
            delta += 2*couplings_fp32[IDX2C(i,n,N)]*matrix_B_fp32[IDX2C(i%M_2, m, M_2)];
        }

        upper = (m == 0 ? M-1 : m-1);
        lower = (m == M-1 ? 0 : m+1);
        delta = M_2*spin_fp32[idx]*(delta - M*J_perp*(spin_fp32[IDX2C(n,upper,N)] + spin_fp32[IDX2C(n,lower,N)]));
        if ( (-log(rand_val_fp32[idx]) / beta) > delta ) {
            spin_fp32[idx] = -spin_fp32[idx];
            matrix_B_fp32[mb_idx] = 2*spin_fp32[idx];
        } else{
            matrix_B_fp32[mb_idx] = 0;
        }
    }

    __syncthreads();

    for(int n = (start_spin + second_rd_idx * 16); n < (start_spin + 16 + second_rd_idx * 16); n++){
        idx = IDX2C(n,m,N);
        mb_idx = IDX2C(n%M_2,m,M_2);            
        delta = delta_H_fp32[idx];
        
        //更新delta: 上面後16個spin做的
        for(int i = (start_spin + first_rd_idx * 16); i < (start_spin + 16 + first_rd_idx * 16); i++){
            delta += 2*couplings_fp32[IDX2C(i,n,N)]*matrix_B_fp32[IDX2C(i%M_2, m, M_2)];
        }
        //更新delta: 本輪做的
        for(int i = (start_spin + second_rd_idx * 16); i < n; i++){
            delta += 2*couplings_fp32[IDX2C(i,n,N)]*matrix_B_fp32[IDX2C(i%M_2, m, M_2)];
        }

        upper = (m == 0 ? M-1 : m-1);
        lower = (m == M-1 ? 0 : m+1);
        delta = M_2*spin_fp32[idx]*(delta - M*J_perp*(spin_fp32[IDX2C(n,upper,N)] + spin_fp32[IDX2C(n,lower,N)]));
        if ( (-log(rand_val_fp32[idx]) / beta) > delta ) {
            spin_fp32[idx] = -spin_fp32[idx];
            matrix_B_fp32[mb_idx] = 2*spin_fp32[idx];
        } else {
            matrix_B_fp32[mb_idx] = 0;
        }
    }
}

__global__ void judge_flipping_com (float *couplings_fp32, float *delta_H_fp32, float *spin_fp32, float *matrix_B_fp32, float *rand_val_fp32, int J_perp, float beta, int start_spin){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    flip_com(couplings_fp32, delta_H_fp32, spin_fp32, matrix_B_fp32, rand_val_fp32, J_perp, beta, start_spin, idx);
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
    matrix_B = (float*)malloc(M_2*M*sizeof(float));
    memset(matrix_B, 0, M_2*M*sizeof(float));

    float *matrix_B_fp32;
    cudaErrCheck(cudaMalloc((void**)&matrix_B_fp32, M_2*M*sizeof(float)));
    cudaErrCheck(cudaMemcpy(matrix_B_fp32, matrix_B, M_2*M*sizeof(float), cudaMemcpyHostToDevice));
    
    float *rand_val;
    rand_val = (float*)malloc(M*N*sizeof(float));
    memset(rand_val, 0, M*N*sizeof(float));

    float *rand_val_fp32;
    cudaErrCheck(cudaMalloc((void**)&rand_val_fp32, M*N*sizeof(float)));
    cudaErrCheck(cudaMemcpy(rand_val_fp32, rand_val, M*N*sizeof(float), cudaMemcpyHostToDevice));

    // TC, using tensor core
    cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH)); 
    
    // Parameters init
    float results[TIMES] = {0.};
    float used_time[TIMES] = {0.};
    float increase = (8 - 1/(float)16) / (float)STEP;
    float G0 = 8.;
    double avg_current_time = 0.;
    
    for (int t = 0; t < TIMES; t++) {
        float beta = 1/(float)16; //bete = 1/Time
        
        //init spin
        construct_spin(spin, spin_fp32,total_spins);
        construct_delta_H(cublasHandle,couplings_fp32, spin_fp32, delta_H_fp32);
        construct_rand_val(rand_val, rand_val_fp32);
        
        // Current cost time
        double curr = 0.;
        clock_t begin, end;

        for (int p = 0; p < STEP; p++) {
            float Gamma = G0*(1.-(float)p/(float)STEP);
            float J_perp = -0.5*log(tanh((Gamma/M)*beta))/beta;
            begin = clock();

            for (int n = 0; n < N; n+=32) {
                judge_flipping_com <<< 1, 16, 0 >>> (couplings_fp32, delta_H_fp32, spin_fp32, matrix_B_fp32, rand_val_fp32, J_perp, beta, n);
                update_delta_H(cublasHandle, couplings_fp32, matrix_B_fp32, delta_H_fp32, n);              
            }
            beta += increase;
            end = clock();
            double duration = (double)(end-begin) / CLOCKS_PER_SEC;
            curr += duration;
            used_time[t] = curr;
            //printf("curr: %10lf, energy: %10d\n", curr, E);
        } 
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
    free(couplings);
    free(spin);
    free(delta_H);
    free(matrix_B);
    free(rand_val);
    cudaFree(couplings_fp32);
    cudaFree(spin_fp32);
    cudaFree(delta_H_fp32);
    cudaFree(matrix_B_fp32);
    cudaFree(rand_val_fp32);
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

void check_rand_val(float *rand_val, float *rand_val_fp32){
    cudaErrCheck ( cudaMemcpy(rand_val, rand_val_fp32, M*N*sizeof(float), cudaMemcpyDeviceToHost));
    for (int n = 0; n < N; n++){
        for (int m = 0; m < M; m++){
            printf("%f ", (float)rand_val[IDX2C(n,m,N)]);
        }
        printf("\n");
    }    

}

