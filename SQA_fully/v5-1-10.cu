#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "cuda_profiler_api.h"
#include <cublas_v2.h>
#include <omp.h>
#include <mma.h>
#include <stdbool.h>
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

// SQA parameters
#define N 32768
#define M 128
#define M_2 128
#define NUM_GPU 8
#define MAX_CONCURRENT 2

#define TIMES 1
#define STEP 100

// Must be multiples of 16
#define MATRIX_M N
#define MATRIX_K M_2
#define MATRIX_N (M/NUM_GPU)

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

void construct_spin(float *spin, float *spin_fp32){
    float x;
    #pragma omp parallel for
    for(int m = 0; m < (M/NUM_GPU); m++){
        for (int n = 0; n < N; n++){
            x = ((float)rand()/(float)(RAND_MAX)) * 1.0;    
            spin[IDX2C(n,m,N)] = ((x>=0.5) ? (float)1. : (float)-1.);
        }
    }
    cudaErrCheck(cudaMemcpy(spin_fp32, spin, (M/NUM_GPU)*N*sizeof(float), cudaMemcpyHostToDevice));
}

void construct_delta_H(half *couplings, float *spin, float *delta_H, float *delta_H_fp32){
    memset(delta_H, 0, (M/NUM_GPU)*N*sizeof(float));
    #pragma omp parallel for
    for (int m = 0; m < (M/NUM_GPU); m++)
        for (int n = 0; n < N; n++)
            for (int i = 0; i < N; i++)
                delta_H[IDX2C(n,m,N)] += (float)couplings[IDX2C(i,n,N)]*spin[IDX2C(i,m,N)];
    cudaErrCheck(cudaMemcpy(delta_H_fp32, delta_H, (M/NUM_GPU)*N*sizeof(float), cudaMemcpyHostToDevice));
}

void update_delta_H (cublasHandle_t cublasHandle, 
                     half *couplings_fp16, 
                     half *matrix_B_fp16, 
                     float *delta_H_fp32,
                     int which_spin)
{
    float alpha = 1.0f, beta = 1.0f;    
    int blk_num = which_spin / M_2;
    int coup_idx = blk_num * (N * M_2);
    cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                MATRIX_M, MATRIX_N, MATRIX_K,
                                &alpha, 
                                couplings_fp16 + coup_idx, CUDA_R_16F, MATRIX_M,
                                matrix_B_fp16, CUDA_R_16F, MATRIX_K, 
                                &beta, 
                                delta_H_fp32, CUDA_R_32F, MATRIX_M,
                                CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void construct_lograndval(float *log_rand_val, float *log_rand_val_fp32, cudaStream_t stream){
    for(int i = 0; i < N; i++){
        log_rand_val[IDX2C(i,0,N)] = (-log(((float)rand()/(float)(RAND_MAX)) * 1.0));
    }
    for (int m = (M/NUM_GPU)-1; m >= 1; m--)
        memcpy(&log_rand_val[m*N], &log_rand_val[(m-1)*N], N*sizeof(float));
    cudaErrCheck (cudaMemcpyAsync(log_rand_val_fp32, log_rand_val, (M/NUM_GPU)*N*sizeof(float), cudaMemcpyHostToDevice, stream));
}

float calculate_E (float *spin, float *spin_fp32, float *delta_H, float *delta_H_fp32, half *couplings){
    cudaErrCheck(cudaMemcpy(spin, spin_fp32, N*sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(delta_H, delta_H_fp32, N*sizeof(float), cudaMemcpyDeviceToHost));
    float E = 0.;
    for (int i = 0; i < N; i++) {
        E += -spin[IDX2C(i,0,N)]*delta_H[IDX2C(i,0,N)];
    }
    return E/2.;
}

__global__ void judge_flipping_com (half *couplings_fp16, 
                                    float *delta_H_fp32, 
                                    float *spin_fp32, 
                                    half *matrix_B_fp16, 
                                    float *log_rand_val_fp32, 
                                    int J_perp, 
                                    float beta, 
                                    int start_spin, 
                                    float *spin_fp32_lower, 
                                    float *spin_fp32_upper,
                                    int couple_start_idx)
{
    unsigned int m = blockIdx.x;
    unsigned int upper, lower;
    int idx, mb_idx;
    float delta;
    int first_rd_idx = m&1; //even:0, odd:1
    
    extern __shared__ float deltas[];
    deltas[threadIdx.x] = delta_H_fp32[IDX2C(start_spin+threadIdx.x, m, N)];
    
    upper = (m-1) & ((M/NUM_GPU)-1);
    lower = (m+1) & ((M/NUM_GPU)-1);
        
    // even: 0~M_2/2-1; odd: M_2/2~M_2-1
    #pragma unroll
    for (int n = 0; n < M_2; n++) {
        int nn  = start_spin + ((first_rd_idx*(M_2/2) + n)&(M_2-1));
        int nnn = couple_start_idx + ((first_rd_idx*(M_2/2) + n)&(M_2-1));
        idx = IDX2C(nn,m,N);
        mb_idx = IDX2C(nn&(M_2-1),m,M_2);            
        delta = deltas[nn&(M_2-1)];
        float spin_upper = (m == 0)               ? spin_fp32_upper[nn] : spin_fp32[IDX2C(nn,upper,N)];
        float spin_lower = (m == ((M/NUM_GPU)-1)) ? spin_fp32_lower[nn] : spin_fp32[IDX2C(nn,lower,N)];
        delta = beta*spin_fp32[idx]*(delta - J_perp*(spin_upper + spin_lower));
        
        matrix_B_fp16[mb_idx] = 0;
        if ( (log_rand_val_fp32[idx]) > delta ) {
            spin_fp32[idx] = -spin_fp32[idx];
            matrix_B_fp16[mb_idx] = 2*spin_fp32[idx];
            int ii = start_spin + threadIdx.x;
            deltas[threadIdx.x] += (float)couplings_fp16[IDX2C(ii,nnn,N)]*(float)matrix_B_fp16[mb_idx]; 
        } 
        __syncthreads();
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) 
        usage();
    
    //Initialize TC, for check
    cudaEvent_t startcublas;
    cudaEvent_t stopcublas;

    cudaErrCheck(cudaEventCreate(&startcublas));
    cudaErrCheck(cudaEventCreate(&stopcublas));
    
    // Initialize couplings
    half *couplings; // cpu    
    cudaErrCheck(cudaMallocHost((void**)&couplings, N*N*sizeof(half)));
    memset(couplings, 0, N*N*sizeof(half));
    
    // Read files
    FILE *instance = fopen(argv[1], "r");
    assert(instance != NULL);
    int a, b, total_spins, total_couplings;
    float w;
    fscanf(instance, "%d%d", &total_spins, &total_couplings);
    while (total_couplings --) {
        fscanf(instance, "%d%d%f", &a, &b, &w);
        a--;
        b--;
        couplings[IDX2C(a,b,N)] = w;
        couplings[IDX2C(b,a,N)] = w;
    }
    fclose(instance);

    // Buffer for spin upper in GPU
    float *spin_fp32_upper[NUM_GPU]; 

    // Buffer for spin lower in GPU
    float *spin_fp32_lower[NUM_GPU]; 

    // coupling buffers
    half *couplings_fp16[NUM_GPU]; 
    half *couplings_fp16_buf[NUM_GPU]; 

    float results[TIMES] = {0.};
    float used_time[TIMES] = {0.};
    omp_set_nested(1);

#if NUM_GPU == 2
    unsigned device_list[NUM_GPU] = {0, 1};
#elif NUM_GPU == 4
    unsigned device_list[NUM_GPU] = {0, 1, 2, 3};
#elif NUM_GPU == 8
    unsigned device_list[NUM_GPU] = {0, 1, 2, 3, 5, 4, 7, 6};
#endif

#pragma omp parallel num_threads(NUM_GPU) shared(couplings, spin_fp32_upper, spin_fp32_lower)
{
    unsigned int tidx = omp_get_thread_num();
    unsigned int cur_dev  = device_list[tidx];
    unsigned int next_dev = device_list[(tidx+1)%NUM_GPU];
    unsigned int last_dev = device_list[(tidx-1)%NUM_GPU];

    // Switch which the GPU device this thread deals with
    cudaErrCheck( cudaSetDevice(cur_dev) );
    cudaErrCheck( cudaDeviceEnablePeerAccess(next_dev, 0) );
#if NUM_GPU >= 4
    cudaErrCheck( cudaDeviceEnablePeerAccess(last_dev, 0) );
#endif

    // Initialize spin
    float *spin;
    spin = (float*)malloc((M/NUM_GPU)*N*sizeof(float));
    
    float *spin_fp32;
    cudaErrCheck( cudaMalloc((void**)&spin_fp32, (M/NUM_GPU)*N*sizeof(float)) );

    float *delta_H;
    delta_H = (float*)malloc((M/NUM_GPU)*N*sizeof(float));
    
    float *delta_H_fp32;
    cudaErrCheck(cudaMalloc((void**)&delta_H_fp32, (M/NUM_GPU)*N*sizeof(float)));

    half *matrix_B_fp16;
    cudaErrCheck(cudaMalloc((void**)&matrix_B_fp16, (M/NUM_GPU)*M_2*sizeof(half)));
    
    float *log_rand_val;
    cudaErrCheck(cudaMallocHost((void**)&log_rand_val, (M/NUM_GPU)*N*sizeof(float)));
    
    float *log_rand_val_fp32;
    cudaErrCheck(cudaMalloc((void**)&log_rand_val_fp32, (M/NUM_GPU)*N*sizeof(float)));
    
    // Buffer for J in GPU
    cudaErrCheck(cudaMalloc(couplings_fp16 + tidx, (N/NUM_GPU)*N*sizeof(half)));
    cudaErrCheck(cudaMemcpy(couplings_fp16[tidx], couplings+(N/NUM_GPU)*N*(tidx), (N/NUM_GPU)*N*sizeof(half), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMalloc(couplings_fp16_buf + tidx, (N/NUM_GPU)*N*sizeof(half)));

    // Buffer for spin buffer in GPU
    cudaErrCheck(cudaMalloc(spin_fp32_upper + tidx, N*sizeof(float)));
    cudaErrCheck(cudaMalloc(spin_fp32_lower + tidx, N*sizeof(float)));
    
    // TC, using tensor core
    
    cudaStream_t stream1[N/NUM_GPU];
    cublasHandle_t cublasHandle[N/NUM_GPU];
    for (int i = 0; i < MAX_CONCURRENT; i++) {
        cudaErrCheck(cudaStreamCreateWithFlags(&stream1[i], cudaStreamNonBlocking));
        cublasErrCheck(cublasCreate(&cublasHandle[i]));
        cublasErrCheck(cublasSetMathMode(cublasHandle[i], CUBLAS_TENSOR_OP_MATH)); 
        //cublasErrCheck(cublasSetStream(cublasHandle[i], stream1[i]));
    }

    cudaStream_t stream2;
    cudaErrCheck(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));

    // Parameters init
    float increase = (16 - 1/(float)16) / (float)STEP;
    float G0 = 8.;

    float *best_spin;
    best_spin = (float*)malloc(N*sizeof(float));
    memset(best_spin, 0, N*sizeof(float)); 
    float best_E = 1e9;

    for (int t = 0; t < TIMES; t++) {
        float beta = 1/(float)16; //bete = 1/Time
        
        //init spin
        construct_spin(spin, spin_fp32);
        construct_delta_H(couplings, spin, delta_H, delta_H_fp32);
            
        // Current cost time
        struct timeval start, end;
        gettimeofday(&start, NULL);

        for (int p = 0; p < STEP; p++) {
            
            float Gamma = G0*(1.-(float)p/(float)STEP);
            float J_perp = -M*0.5*log(tanh((Gamma/M)*beta))/beta;
            
            construct_lograndval(log_rand_val, log_rand_val_fp32, stream2);
            for (int d = 0; d < NUM_GPU; d++){
                if (d % 2 == 0) {
                    cudaErrCheck( cudaMemcpy(couplings_fp16_buf[(tidx+1)%NUM_GPU], 
                                                  couplings_fp16[tidx], (N/NUM_GPU)*N*sizeof(half),
                                                  cudaMemcpyDeviceToDevice) ); 
                } else {
                    cudaErrCheck( cudaMemcpy(couplings_fp16[(tidx+1)%NUM_GPU], 
                                                  couplings_fp16_buf[tidx], (N/NUM_GPU)*N*sizeof(half),
                                                  cudaMemcpyDeviceToDevice) ); 
                }
                for (int n = 0, i = 0; n < N/NUM_GPU; n += M_2, i++) {
                    if (d % 2 == 0) {
                        judge_flipping_com<<< (M/NUM_GPU), M_2, M_2*sizeof(float) >>>(
                            couplings_fp16[tidx], delta_H_fp32, 
                            spin_fp32, matrix_B_fp16, log_rand_val_fp32, J_perp, 2*M*beta, n+((tidx+d)%NUM_GPU)*(N/NUM_GPU), 
                            spin_fp32_lower[tidx], spin_fp32_upper[tidx], n);
                        update_delta_H(cublasHandle[0], couplings_fp16[tidx], matrix_B_fp16, delta_H_fp32, n);              
                    } else {
                        judge_flipping_com<<< (M/NUM_GPU), M_2, M_2*sizeof(float) >>>(
                            couplings_fp16_buf[tidx], delta_H_fp32, 
                            spin_fp32, matrix_B_fp16, log_rand_val_fp32, J_perp, 2*M*beta, n+((tidx+d)%NUM_GPU)*(N/NUM_GPU), 
                            spin_fp32_lower[tidx], spin_fp32_upper[tidx], n);
                        update_delta_H(cublasHandle[0], couplings_fp16_buf[tidx], matrix_B_fp16, delta_H_fp32, n);              
                    }
                }
                cudaErrCheck( cudaMemcpy(spin_fp32_lower[(tidx-1)%NUM_GPU]+((tidx+d)%NUM_GPU)*(N/NUM_GPU), 
                                         spin_fp32+((tidx+d)%NUM_GPU)*(N/NUM_GPU), 
                                               (N/NUM_GPU)*sizeof(float), cudaMemcpyDeviceToDevice) ); 
                cudaErrCheck( cudaMemcpy(spin_fp32_upper[(tidx+1)%NUM_GPU]+((tidx+d)%NUM_GPU)*(N/NUM_GPU), 
                                         spin_fp32+((M/NUM_GPU-1)*N)+((tidx+d)%NUM_GPU)*(N/NUM_GPU), 
                                               (N/NUM_GPU)*sizeof(float), cudaMemcpyDeviceToDevice) ); 
#pragma omp barrier
            }
            beta += increase;

            /*float E = calculate_E(spin, spin_fp32, delta_H, delta_H_fp32, couplings);
            if (E < best_E) {
                best_E = E;
                memcpy(best_spin, spin, N*sizeof(float));
            }
            gettimeofday(&end, NULL);
            double delta = ((end.tv_sec  - start.tv_sec) * 1000000u + 
                             end.tv_usec - start.tv_usec) / 1.e6;
            printf("Step: %d, id: %d, curr: %10lf, energy: %10f\n", p, tidx, delta, E);*/
        } 
        cudaDeviceSynchronize();
#pragma omp barrier
        gettimeofday(&end, NULL);
        double delta = ((end.tv_sec  - start.tv_sec) * 1000000u + 
                         end.tv_usec - start.tv_usec) / 1.e6;
       
        // record
        float best_E = calculate_E(spin, spin_fp32, delta_H, delta_H_fp32, couplings);
        printf("%d, %lf, %f\n", tidx, delta, best_E);
        if (tidx == 0) {
            results[t] = best_E;
            used_time[t] = delta;
        }
    }
    
    if (tidx == 0) {
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
    }
}

    return 0;
}

void usage (){
    printf("Usage:\n");
    printf("       ./sqa [spin configuration]\n");
    exit(0);
}
