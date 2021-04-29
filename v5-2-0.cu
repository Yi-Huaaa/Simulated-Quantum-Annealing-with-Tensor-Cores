/*
random移到外面
和 515差別：random 的時間算進去

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
#define N 1024
#define M 16 
#define M_2 32

#define TIMES 1//10
#define STEP 10//100

// Must be multiples of 16
#define MATRIX_M N
#define MATRIX_K N
#define MATRIX_N M

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;


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
__global__ void convertFp32ToFp16 (half *out, float *in, int n, int m) {
    for(int i = 0; i < n; i++){
       for(int j = 0; j < m; j++){
          out[IDX2C(i,j,N)] = in[IDX2C(i,j,N)];
       }
    }
}
__global__ void calculate_E (float *couplings_fp32, half *spin_fp16, int time,int E){
    for (int i = 0; i < N; i++){
        for (int j = i+1; j < N; j++){
            E += -(float)spin_fp16[IDX2C(i,0,N)]*(float)spin_fp16[IDX2C(j,0,N)]*couplings_fp32[IDX2C(i,j,N)];
        }
    }
    //printf("TIMES = %d, Energy = %d\n", time, E);
}

__global__ void wmma_example(half *a, half *b, float *c, int M_1, int N_1, int K_1, float alpha, float beta) {
   // Leading dimensions. Packed with no transpositions.
   int lda = M_1;
   int ldb = K_1;
   int ldc = M_1;

   // Tile using a 2D grid
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
 
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

   wmma::fill_fragment(acc_frag, 0.0f);

   // Loop over k
   for (int i = 0; i < K_1; i += WMMA_K) {
      int aRow = warpM * WMMA_M;
      int aCol = i;

      int bRow = i;
      int bCol = warpN * WMMA_N;

      // Bounds checking
      if (aRow < M_1 && aCol < K_1 && bRow < K_1 && bCol < N_1) {
         // Load the inputs
         wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
         wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
 
         // Perform the matrix multiplication
         wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

      }
   }

   // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;

   if (cRow < M && cCol < N) {
      wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);


      for(int i=0; i < c_frag.num_elements; i++) {
         c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
      }

      // Store the output
      wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
   }
}

void usage ();
void check_spin(float *spin, float* spin_fp32);
void check_couplings(float *couplings);
void check_delta_H (float *delta_H, float *delta_H_fp32);

void construct_spin(float *spin, float *spin_fp32, half *spin_fp16, int total_spins){
    float x;
    for (int n = 0; n < N; n++){
        for(int m = 0; m < M; m++){
            x = ((float)rand()/(float)(RAND_MAX)) * 1.0;    
            spin[IDX2C(n,m,N)] = ((x>=0.5) ? (float)1. : (float)-1.);
        }
    }
    cudaErrCheck (cudaMemcpy(spin_fp32, spin, M*N*sizeof(float), cudaMemcpyHostToDevice));
    convertFp32ToFp16 <<< 1,1,0 >>> (spin_fp16, spin_fp32, N, M);
}

void construct_delta_H(cublasHandle_t cublasHandle, half *couplings_fp16, half *spin_fp16, float *delta_H_fp32){
    float alpha = 1.0f, beta = 0.0f;
    cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                            MATRIX_M, MATRIX_N, MATRIX_K,
                            &alpha, 
                            couplings_fp16, CUDA_R_16F, MATRIX_M,
                            spin_fp16, CUDA_R_16F, MATRIX_K, 
                            &beta, 
                            delta_H_fp32, CUDA_R_32F, MATRIX_M,
                            CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
/*    cublasErrCheck(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                MATRIX_M, MATRIX_N, MATRIX_K,
                                &alpha, 
                                couplings_fp32, MATRIX_M,
                                spin_fp32, MATRIX_K,
                                &beta,
                                delta_H_fp32, MATRIX_M));*/

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
                            CUBLAS_COMPUTE_32F , CUBLAS_GEMM_DEFAULT_TENSOR_OP));

}


void construct_lograndval(float *log_rand_val, float *log_rand_val_fp32, float beta){
    srand(time(0));
    for(int i = 0; i < M_2; i++){
        for(int j = 0; j < M; j++){
            log_rand_val[IDX2C(i,j,M_2)] = (-log(((float)rand()/(float)(RAND_MAX)) * 1.0)) / beta;
        }
    }
    cudaErrCheck (cudaMemcpy(log_rand_val_fp32, log_rand_val, M_2*M*sizeof(float), cudaMemcpyHostToDevice));
}


__device__ void flip_com (half *couplings_fp16, float *delta_H_fp32, half *spin_fp16, half *matrix_B_fp16, float *log_rand_val_fp32, int J_perp, float beta, int start_spin, int m){
    int idx = 0, mb_idx = 0, upper = 0, lower = 0;
    float delta = 0.;
    int first_rd_idx = m%2; //even:0, odd:1
    int second_rd_idx = (m+1)%2; //even:1, odd:0

    // 先坐前面16個，even: 0~16; odd: 17~31
    for(int n = (start_spin + first_rd_idx * 16); n < (start_spin + 16 + first_rd_idx * 16); n++){
        idx = IDX2C(n,m,N);
        mb_idx = IDX2C(n%M_2,m,M);            
        delta = delta_H_fp32[idx];
        
        for(int i = start_spin + first_rd_idx*16; i < n; i++){
            delta += (float)couplings_fp16[IDX2C(i,n,N)]*(float)matrix_B_fp16[IDX2C(i%M_2, m, M)];
        }
        //printf("1:delta = %f\n",delta );

        upper = (m == 0 ? M-1 : m-1);
        lower = (m == M-1 ? 0 : m+1);
        delta = M_2*(float)spin_fp16[idx]*(delta - M*J_perp*((float)spin_fp16[IDX2C(n,upper,N)] + (float)spin_fp16[IDX2C(n,lower,N)]));
        if ( log_rand_val_fp32[mb_idx] > delta ) {
            spin_fp16[idx] = -spin_fp16[idx];
            matrix_B_fp16[mb_idx] = (spin_fp16[idx]+spin_fp16[idx]);
        } else{
            matrix_B_fp16[mb_idx] = 0;
        }
    }

    __syncthreads();

    for(int n = (start_spin + second_rd_idx * 16); n < (start_spin + 16 + second_rd_idx * 16); n++){
        idx = IDX2C(n,m,N);
        mb_idx = IDX2C(n%M_2,m,M);            
        delta = delta_H_fp32[idx];
        
        //更新delta: 上面後16個spin做的
        for(int i = (start_spin + first_rd_idx * 16); i < (start_spin + 16 + first_rd_idx * 16); i++){
            delta += (float)couplings_fp16[IDX2C(i,n,N)]*(float)matrix_B_fp16[IDX2C(i%M_2, m, M)];
        }
        //更新delta: 本輪做的
        for(int i = (start_spin + second_rd_idx * 16); i < n; i++){
            delta += (float)couplings_fp16[IDX2C(i,n,N)]*(float)matrix_B_fp16[IDX2C(i%M_2, m, M)];
        }

        upper = (m == 0 ? M-1 : m-1);
        lower = (m == M-1 ? 0 : m+1);
        delta = M_2*(float)spin_fp16[idx]*(delta - M*J_perp*((float)spin_fp16[IDX2C(n,upper,N)] + (float)spin_fp16[IDX2C(n,lower,N)]));
        if ( log_rand_val_fp32[mb_idx] > delta ) {
            spin_fp16[idx] = -spin_fp16[idx];
            matrix_B_fp16[mb_idx] = (spin_fp16[idx]+spin_fp16[idx]);
        } else {
            matrix_B_fp16[mb_idx] = 0;
        }
    }
}

__global__ void judge_flipping_com (half *couplings_fp16, float *delta_H_fp32, half *spin_fp16, half *matrix_B_fp16, float *log_rand_val_fp32, int J_perp, float beta, int start_spin){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    flip_com(couplings_fp16, delta_H_fp32, spin_fp16, matrix_B_fp16, log_rand_val_fp32, J_perp, beta, start_spin, idx);
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
    
    float *couplings_fp32; 
    cudaErrCheck(cudaMalloc((void**)&couplings_fp32, N*N*sizeof(float)));
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
    cudaErrCheck (cudaMemcpy(couplings_fp32, couplings, N*N*sizeof(float), cudaMemcpyHostToDevice));
    convertFp32ToFp16 <<< 1,1,0 >>> (couplings_fp16, couplings_fp32, N, N);
    //check_couplings(couplings);
    
    // Initialize spin
    float *spin;
    spin = (float*)malloc(M*N*sizeof(float));
    memset(spin, 0, M*N*sizeof(float));
    
    float *spin_fp32;
    cudaErrCheck (cudaMalloc((void**)&spin_fp32, M*N*sizeof(float)) );
    cudaErrCheck(cudaMemcpy(spin_fp32, spin, M*N*sizeof(float), cudaMemcpyHostToDevice));
    
    half *spin_fp16;
    cudaErrCheck (cudaMalloc((void**)&spin_fp16, M*N*sizeof(half)));
//
    float *delta_H;
    delta_H = (float*)malloc(M*N*sizeof(float));
    memset(delta_H, 0, M*N*sizeof(float));
    
    float *delta_H_fp32;
    cudaErrCheck(cudaMalloc((void**)&delta_H_fp32, M*N*sizeof(float)));
    cudaErrCheck(cudaMemcpy(delta_H_fp32, delta_H, M*N*sizeof(float), cudaMemcpyHostToDevice));

    float *matrix_B;
    matrix_B = (float*)malloc(M_2*M*sizeof(float));
    memset(matrix_B, 0, M_2*M*sizeof(float));

    half *matrix_B_fp16;
    cudaErrCheck(cudaMalloc((void**)&matrix_B_fp16, M_2*M*sizeof(half)));
    cudaErrCheck(cudaMemcpy(matrix_B_fp16, matrix_B, M_2*M*sizeof(half), cudaMemcpyHostToDevice));
    
    float *log_rand_val;
    log_rand_val = (float*)malloc(M*M_2*sizeof(float));
    memset(log_rand_val, 0, M*M_2*sizeof(float));

    float *log_rand_val_fp32;
    cudaErrCheck(cudaMalloc((void**)&log_rand_val_fp32, M*M_2*sizeof(float)));
    cudaErrCheck(cudaMemcpy(log_rand_val_fp32, log_rand_val, M*M_2*sizeof(float), cudaMemcpyHostToDevice));


    // TC, using tensor core
    cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH)); 

    // Parameters init
    float results[TIMES] = {0.};
    float used_time[TIMES] = {0.};
    float increase = (8 - 1/(float)16) / (float)STEP;
    float G0 = 8.;

    dim3 gridDim;
    dim3 blockDim;
    
    blockDim.x = 128;
    blockDim.y = 4;

    gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
   
    for (int t = 0; t < TIMES; t++) {
        float beta = 1/(float)16; //bete = 1/Time
        
        //init spin
        construct_spin(spin, spin_fp32, spin_fp16, total_spins);
        //check_spin(spin, spin_fp32);
        //wmma_example <<< gridDim, blockDim >>> (couplings_fp16, spin_fp16, delta_H_fp32, MATRIX_M, MATRIX_N, MATRIX_K, 1.0f, 0.0);
        construct_delta_H(cublasHandle, couplings_fp16, spin_fp16, delta_H_fp32);
        //check_delta_H(delta_H, delta_H_fp32);
        
        clock_t begin, end;

        begin = clock();
        for (int p = 0; p < STEP; p++) {
            float Gamma = G0*(1.-(float)p/(float)STEP);
            float J_perp = -0.5*log(tanh((Gamma/M)*beta))/beta;

            for (int n = 0; n < N; n+=32) {
                construct_lograndval(log_rand_val, log_rand_val_fp32, beta);
                judge_flipping_com <<< 1, 16, 0 >>> (couplings_fp16, delta_H_fp32, spin_fp16, matrix_B_fp16, log_rand_val_fp32, J_perp, beta, n);
                //wmma_example <<< gridDim, blockDim >>> (couplings_fp16, matrix_B_fp16, delta_H_fp32, MATRIX_M, MATRIX_N, M_2, 1.0f, 1.0f);
                update_delta_H(cublasHandle, couplings_fp16, matrix_B_fp16, delta_H_fp32, n);              
            }
            beta += increase;
        } 
        cudaDeviceSynchronize();
        end = clock();
        double duration = (double)(end-begin) / CLOCKS_PER_SEC;

        used_time[t] = duration;
        int E = 0;
        calculate_E <<< 1,1,0 >>> (couplings_fp32, spin_fp16, t, E);
        results[t] = E;
    }
    


    printf("Final: \n");
    for (int t = 0; t < TIMES; t++){
        printf("TIME: %d,  used time (s): %10lf,  Energy: %10lf\n", t, used_time[t], results[t]);
    }

    float tot_result_time = 0.;
    for(int i = 0; i < TIMES; i++){
        tot_result_time += used_time[i];
    }
    printf("\nAvg time  : %f\n", tot_result_time/TIMES);
    
    cublasDestroy(cublasHandle);   
    free(couplings);
    free(spin);
    free(delta_H);
    free(matrix_B);
    free(log_rand_val);
    cudaFree(couplings_fp32);
    cudaFree(couplings_fp16);
    cudaFree(spin_fp32);
    cudaFree(spin_fp16);
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

void check_spin(float *spin, float *spin_fp32){
    cudaErrCheck ( cudaMemcpy(spin, spin_fp32, M*N*sizeof(float), cudaMemcpyDeviceToHost));
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

void check_delta_H (float *delta_H, float *delta_H_fp32){
    cudaErrCheck ( cudaMemcpy(delta_H, delta_H_fp32, M*N*sizeof(float), cudaMemcpyDeviceToHost));
    printf("check..., print delta_H\n");
    for (int n = 0; n < N; n++){
        for (int m = 0; m < M; m++){
            printf("%d ", (int)delta_H[IDX2C(n,m,N)]);
        }
        printf("\n");
    }
}

