/*
答案對
Properties:
new method
GPU
parallel: odd, even do @ z same time
random number只constrcut一次之後，就一直用同一批
-----
和513差別：合併 "judge_flipping_even" and "judge_flipping_odd" -> launch kernel的時間
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
#define N 2048
#define M 16 

#define TIMES 1//10
#define STEP 100 //100

// Must be multiples of 16
#define MATRIX_M N
#define MATRIX_K N
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
void usage ();
void check_spin(float *spin);
void check_couplings(float *couplings);
void check_delta_H (float *couplings, float *spin, float *delta_H, float *delta_H_fp32);
void check_matrix_B (float *matrix_B, float *matrix_B_fp32);
void check_rand_val(float *rand_val, float *rand_val_fp32);

void construct_spin(float *spin, int total_spins){
    float x;
    for (int n = 0; n < N; n++){
        for(int m = 0; m < M; m++){
            x = ((float)rand()/(float)(RAND_MAX)) * 1.0;    
            spin[IDX2C(n,m,N)] = ((x>=0.5) ? (float)1. : (float)-1.);
        }
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

void update_delta_H(cublasHandle_t cublasHandle, float *couplings_fp32, float *matrix_B_fp32, float *delta_H_fp32, int which_spin){
    float alpha = 1.0f, beta = 1.0f;    
    int blk_num = which_spin / M;
    int coup_idx = blk_num * (N*M);
    cublasErrCheck(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                MATRIX_M, MATRIX_N, MATRIX_N,
                                &alpha, 
                                couplings_fp32 + coup_idx, MATRIX_M,
                                matrix_B_fp32, MATRIX_N,
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

__global__ void judge_flipping_com (float *couplings_fp32,float *delta_H_fp32, float *spin_fp32, float *matrix_B_fp32, float *rand_val_fp32, int J_perp, float beta, int start_spin){
    // judge even: 
    int idx = 0, mb_idx = 0, upper = 0, lower = 0;
    float delta = 0.;
    int m = (blockIdx.x * blockDim.x + threadIdx.x)*2;
    int sixteen = start_spin+16;
    int changed[16] = {0};
    int changed_id = 0, in_loop_changed_id = 0;
    
    for(int n = start_spin; n < sixteen; n++){
        idx = IDX2C(n,m,N);
        mb_idx = IDX2C(n%M,m,M);            
        delta = delta_H_fp32[idx];
        
        //更新delta
        in_loop_changed_id = 0;
        for(int pre_n = start_spin; pre_n < n; pre_n++){ //pre_n: 這一輪中前面的所有spin
            if(changed[in_loop_changed_id] == 1){
                delta += 2*couplings_fp32[IDX2C( pre_n, n, N)]*spin_fp32[IDX2C( pre_n, m, N)];
            }
            in_loop_changed_id ++;
        }

        upper = (m == 0 ? M-1 : m-1);
        lower = (m == M-1 ? 0 : m+1);
        delta = 2*M*spin_fp32[idx]*(delta - M*J_perp*(spin_fp32[IDX2C(n,upper,N)] + spin_fp32[IDX2C(n,lower,N)]));
        if ( (-log(rand_val_fp32[idx]) / beta) > delta ) {
            spin_fp32[idx] = -spin_fp32[idx];
            matrix_B_fp32[mb_idx] = 2*spin_fp32[idx];
            changed[changed_id] = 1;
        }    
        changed_id ++;   
    }

    // judge odd:
    idx = 0; 
    mb_idx = 0; 
    upper = 0; 
    lower = 0;
    delta = 0.;
    m = (blockIdx.x * blockDim.x + threadIdx.x)*2 + 1;
    changed[16] = {0};
    changed_id = 0;
    in_loop_changed_id = 0;

    for(int n = start_spin; n < sixteen; n++){
        idx = IDX2C(n,m,N);
        mb_idx = IDX2C(n%M,m,M);            
        delta = delta_H_fp32[idx];
        
        //更新delta
        in_loop_changed_id = 0;
        for(int pre_n = start_spin; pre_n < n; pre_n++){ //pre_n: 這一輪中前面的所有spin
            if(changed[in_loop_changed_id] == 1){
                delta += 2*couplings_fp32[IDX2C( pre_n, n, N)]*spin_fp32[IDX2C( pre_n, m, N)];
            }
            in_loop_changed_id ++;
        }

        upper = (m == 0 ? M-1 : m-1);
        lower = (m == M-1 ? 0 : m+1);
        delta = 2*M*spin_fp32[idx]*(delta - M*J_perp*(spin_fp32[IDX2C(n,upper,N)] + spin_fp32[IDX2C(n,lower,N)]));
        if ( (-log(rand_val_fp32[idx]) / beta) > delta ) {
            spin_fp32[idx] = -spin_fp32[idx];
            matrix_B_fp32[mb_idx] = 2*spin_fp32[idx];
            changed[changed_id] = 1;
        }   
        changed_id ++;   
    }
}


__global__ void clear_matrix_B (float *matrix_B_fp32){
    for(int i = 0; i < M; i++){
        for(int j = 0; j < M; j++){
            matrix_B_fp32[IDX2C(i,j,M)] = 0.;
        }
    }
}


__global__ void trytry_spin32(float *spin_fp32){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	for(int m = 0; m < M; m++){
		spin_fp32[IDX2C(idx, m ,N)] = idx;
	}
	if(idx == 0){
		for (int n = 0; n < N; n++){
	        for(int m = 0; m < M; m++){
	            printf("%d ", (int)spin_fp32[IDX2C(n,m,N)]);
	        }
	        printf("\n");
	    }
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
    matrix_B = (float*)malloc(M*M*sizeof(float));
    memset(matrix_B, 0, M*M*sizeof(float));

    float *matrix_B_fp32;
    cudaErrCheck(cudaMalloc((void**)&matrix_B_fp32, M*M*sizeof(float)));
    cudaErrCheck(cudaMemcpy(matrix_B_fp32, matrix_B, M*M*sizeof(float), cudaMemcpyHostToDevice));
    
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
    float increase = (8 - 1/(float)16) / (float)STEP;
    float G0 = 8.;
    
    for (int t = 0; t < TIMES; t++) {
        float beta = 1/(float)16; //bete = 1/Time
        
        //init spin
        construct_spin(spin, total_spins);
        cudaErrCheck (cudaMemcpy(spin_fp32, spin, M*N*sizeof(float), cudaMemcpyHostToDevice));
        // Current cost time
        double curr = 0.;
        construct_delta_H(cublasHandle,couplings_fp32, spin_fp32, delta_H_fp32);
        construct_rand_val(rand_val, rand_val_fp32);
        clock_t begin, end;

        for (int p = 0; p < STEP; p++) {

            float Gamma = G0*(1.-(float)p/(float)STEP);
            float J_perp = -0.5*log(tanh((Gamma/M)*beta))/beta;
            begin = clock();

            for (int n = 0; n < N; n+=16) {
                judge_flipping_com <<<1, 8, 0>>> (couplings_fp32, delta_H_fp32, spin_fp32, matrix_B_fp32, rand_val_fp32, J_perp, beta, n);
                update_delta_H(cublasHandle, couplings_fp32, matrix_B_fp32, delta_H_fp32, n);              
                clear_matrix_B <<< 1,1,0 >>> (matrix_B_fp32);
            }
            beta += increase;
            end = clock();
            double duration = (double)(end-begin) / CLOCKS_PER_SEC;
            curr += duration;
            used_time[t] = curr;
            // int E拉到最外面的時候，時間會變得很長
            int E = calculate_E(spin, spin_fp32, couplings);
            results[t] = E;
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



//try
/*    double cost_time = 0.;    
    //1,16,0: 最快
    clock_t begin = clock();
	trytry_spin32 <<< 1,1,0 >>> (spin_fp32);
	clock_t end = clock();
	cost_time = begin - end;
	printf("1,1,0, cost time = %10lf\n",cost_time);


    clock_t begin1 = clock();
	trytry_spin32 <<< 1,16,0 >>> (spin_fp32);
	clock_t end1 = clock();
	cost_time = begin1 - end1;
	printf("1,16,0, cost time = %10lf\n",cost_time);
*/

//try