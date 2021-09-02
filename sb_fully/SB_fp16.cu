//fp16
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cuda_profiler_api.h"
#include <sys/time.h>
#define N 32768
#define THREADS 64
#define TIMES 1
#define MAX 4294967295.0

// parameter settings (p > detune - xi, bifurcation point)
#define K 1. //hua
#define detune 1. //hua
// #define xi 0.004736 // 0.7*detune / (rho * sqrt(N))
#define xi ((0.7*detune)/(0.57735*sqrt(N)))
#define deltaT 0.5
#define DeltaT 1.0 //hua
#define M 2
#define STEP 100 //hua
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

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
inline void gpuAssert (cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__device__ uint xorshift32 (uint *state)
{
    uint x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// __global__ void prepare_points (float *x,
//                                 float *y,
//                                 uint *randvals) 
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     x[idx] = ((xorshift32(&randvals[idx]) / (float) MAX) / 10.0);
//     y[idx] = ((xorshift32(&randvals[idx]) / (float) MAX) / 10.0);
// }

void construct_spin(half *x, half *x_buf, float *y, float *y_buf){
    for (int n = 0; n < N; n++){  
        x[IDX2C(n,0,N)] = ((float)rand()/(float)(RAND_MAX)) * 0.1;
        y[IDX2C(n,0,N)] = ((float)rand()/(float)(RAND_MAX)) * 0.1;
    }
    x[IDX2C(0, 0, N)] = 0;
    x[IDX2C(1, 0, N)] = 0;
    y[IDX2C(0, 0, N)] = ((float)rand()/(float)(RAND_MAX)) * 0.1;
    y[IDX2C(1, 0, N)] = ((float)rand()/(float)(RAND_MAX)) * 0.1;


    cudaErrCheck (cudaMemcpy(x_buf, x, N*16*sizeof(half), cudaMemcpyHostToDevice));
    cudaErrCheck (cudaMemcpy(y_buf, y, N*16*sizeof(float), cudaMemcpyHostToDevice));
}

__global__ void UpdateTwice (half *x,
                             float *y,
				             float amplitude)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx = IDX2C(idx, 0, N);
	float lx = x[idx];
	float ly = y[idx];

#pragma unroll
	for (int i = 0; i < M; i++) {
        lx = lx + detune*ly*deltaT;
        ly = ly-deltaT*(K*lx*lx*lx + (detune - amplitude)*lx);
	}

	x[idx] = lx;
	y[idx] = ly;
}
void UpdateTwice_CPU (float *x, float *y, float amplitude, half *x_buf, float *y_buf){
    float deltaY = 0., deltaX = 0.;
    for(int n = 0; n < N; n++){
        x[IDX2C(n, 0, N)] += detune*y[IDX2C(n, 0, N)]*deltaT;
        deltaX = x[IDX2C(n, 0, N)];
        deltaY = (-(K*deltaX*deltaX - amplitude + detune))*deltaX*deltaT;
        y[IDX2C(n, 0, N)] += deltaY;
    }
    gpuErrchk( cudaMemcpy(x_buf, x, N*16*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(y_buf, y, N*16*sizeof(float), cudaMemcpyHostToDevice));

}

int print_energy(half *x, half *x_buf, half *couplings, double curr, int t)
{
	//GPU version
    gpuErrchk( cudaMemcpy(x, x_buf, N*16*sizeof(half), cudaMemcpyDeviceToHost));
	int E = 0;
	for (int i = 0; i < N; i++) {
		for (int j = i+1; j < N; j++) {
			int a = (int)x[IDX2C(i, 0, N)] > 0.0 ? 1 : -1;
			int b = (int)x[IDX2C(j, 0, N)] > 0.0 ? 1 : -1;
			E += a*b*(float)couplings[IDX2C(i,j,N)];
		}
	}
    printf("%d %lf %d\n", t, curr, -E);
    return -E;
}

void usage () 
{
    printf("Usage:\n");
    printf("       ./Bifurcation-cuda [spin configuration]\n");
    exit(0);
}

int main (int argc, char *argv[]) 
{
    if (argc != 2) 
        usage();

    //Initialize TC, for check
    cublasHandle_t cublasHandle;
    cudaEvent_t startcublas;
    cudaEvent_t stopcublas;

    cudaErrCheck(cudaEventCreate(&startcublas));
    cudaErrCheck(cudaEventCreate(&stopcublas));
    cublasErrCheck(cublasCreate(&cublasHandle));
    
    // initialize couplings
    half *couplings, *couplings_buf;
    couplings = (half*)malloc(N*N*sizeof(half));
    memset(couplings, 0, N*N*sizeof(half));
    gpuErrchk( cudaMalloc(&couplings_buf, N*N*sizeof(half)) );

    // Read couplings file 
    FILE *instance = fopen(argv[1], "r");
    assert(instance != NULL);
    float w = 0.;
    int a, b, total_spins, total_couplings;

    fscanf(instance, "%d%d", &total_spins, &total_couplings);

    while (!feof(instance)) {
        fscanf(instance, "%d%d%f", &a, &b, &w);
        assert(a != b); // not dealing with external field
        a--;
        b--;
        couplings[IDX2C(a,b,N)] = -w;
        couplings[IDX2C(b,a,N)] = -w;
    }
    fclose(instance);

    // copy couplings to target device
    gpuErrchk( cudaMemcpy(couplings_buf, couplings, N*N*sizeof(half), cudaMemcpyHostToDevice));

    // initialize points
    half *x, *x_buf;
    x = (half*)malloc(N*16*sizeof(half));
    memset(x, 0, N*16*sizeof(half));
    gpuErrchk( cudaMalloc(&x_buf, N*16*sizeof(half)) );
    gpuErrchk( cudaMemcpy(x_buf, x, N*16*sizeof(half), cudaMemcpyHostToDevice));

    float *y, *y_buf;
    y = (float*)malloc(N*16*sizeof(float));
    memset(y, 0, N*16*sizeof(float));
    gpuErrchk( cudaMalloc(&y_buf, N*16*sizeof(float)) );
    gpuErrchk( cudaMemcpy(y_buf, y, N*16*sizeof(float), cudaMemcpyHostToDevice));

    // launching kernel
	float alpha = DeltaT * xi;
	float beta = 1.;
    dim3 grid(N/THREADS), block(THREADS);
    int results[TIMES] = {0};
    float used_time[TIMES] = {0.};
    
    cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));  //CUBLAS_PEDANTIC_MATH, CUBLAS_TENSOR_OP_MATH

    for (int t = 0; t < TIMES; t++) {
        // prepare_points<<<grid, block>>>(x_buf, y_buf, randvals);
        construct_spin(x, x_buf, y, y_buf);

        struct timeval begin, end;
        gettimeofday(&begin, NULL);

        for (int s = 0; s < STEP; s++) {
            // hua
            float amplitude = 2*(s/(float)STEP);
			UpdateTwice<<<grid, block>>>(x_buf, y_buf, amplitude);
            // UpdateTwice_CPU(x, y, amplitude, x_buf, y_buf);
	        cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                            N, 16, N,
                            &alpha,
                            couplings_buf, CUDA_R_16F, N,
                            x_buf, CUDA_R_16F, N,
                            &beta,
                            y_buf, CUDA_R_32F, N,
                            CUBLAS_COMPUTE_32F,  
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            // cublasSgemm(cublasHandle, 
            //     CUBLAS_OP_N, 
            //     CUBLAS_OP_N,
            //     N, 1, N,
            //     &alpha,
            //     couplings_buf, N, 
            //     x_buf, N,
            //     &beta, 
            //     y_buf, N);
            // cublasSgemv(cublasHandle, CUBLAS_OP_N, N, N, &alpha, couplings_buf, N, x_buf, 1, &beta, y_buf, 1);
            // gpuErrchk( cudaMemcpy(x, x_buf, N*16*sizeof(float), cudaMemcpyDeviceToHost));
            // gpuErrchk( cudaMemcpy(y, y_buf, N*16*sizeof(float), cudaMemcpyDeviceToHost));
        }
        cudaDeviceSynchronize();
        gettimeofday(&end, NULL);
        double duration = ((end.tv_sec  - begin.tv_sec) * 1000000u +
                         end.tv_usec - begin.tv_usec) / 1.e6;
            
        used_time[t] = duration;
        results[t] = print_energy(x, x_buf, couplings, duration, t);

        // // Get Result from device
        // gpuErrchk( cudaMemcpy(x, x_buf, N*16*sizeof(float), cudaMemcpyDeviceToHost) );

        // // calculate energy
        // for (int i = 0; i < N; i++) {
        //     for (int j = i+1; j < N; j++) {

        //         int a = x[IDX2C(i,0,N)] > 0.0 ? 1 : -1;
        //         int b = x[IDX2C(j,0,N)] > 0.0 ? 1 : -1;
        //         E += a*b*couplings[IDX2C(i,j,N)];
        //     }
        // }
        printf("TIME: %d,  used_time = %f, Energy: %d\n", t, used_time[t], results[t]);
    }

    // Write statistics to file
    // FILE *output;
    // output = fopen("output.txt", "w");
    // for (int i = 0; i < TIMES; i++)
    //      fprintf(output, "%d\n", results[i]);
    // fclose(output);

    // Release Objects
    free(couplings);
    free(x);
    free(y);
    // free(initRand);
    cudaFree(couplings_buf);
    cudaFree(x_buf);
    cudaFree(y_buf);
    // cudaFree(randvals);
    return 0;
}