#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda.h>
 
#define M 16
#define N 1048576
#define EDGE 1024
#define THREADS 32
#define TIMES 10
#define SWEEP 1000
#define MAX 4294967295.0

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
      printf("Error at %s:%d\n",__FILE__,__LINE__);     \
      return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
      printf("Error at %s:%d\n",__FILE__,__LINE__);            \
      return EXIT_FAILURE;}} while(0)

__device__ uint xorshift32(uint *state)
{
    uint x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

__global__ void prepare_spins(int *spins, 
                              uint *randvals) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    for (int i = 0; i < M; i++)
        spins[i*N+idx] = ((xorshift32(&randvals[idx]) & 1) << 1) - 1;
}

__global__ void ising(int* spins, 
                      const int* couplings, 
                      int signal, 
                      float beta,
                      uint *randvals,
                      int m,
                      float Jtrans)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    // printf("%d %d %f %f\n", idx, m, beta, Jtrans);

    // annealing (calculate 8 neighbor energy difference)
    int n = idx-EDGE-1;
    int e0 = ((idx%EDGE) == 0 || idx < EDGE) ?
                0 : couplings[idx*9+0]*spins[m*N+n];  

    n = idx-EDGE;
    int e1 = (idx < EDGE) ?
                0 : couplings[idx*9+1]*spins[m*N+n];  

    n = idx-EDGE+1;
    int e2 = (((idx+1)%EDGE) == 0 || idx < EDGE) ?
                0 : couplings[idx*9+2]*spins[m*N+n];  

    n = idx-1;
    int e3 = ((idx%EDGE) == 0) ?
                0 : couplings[idx*9+3]*spins[m*N+n];  

    n = idx+1;
    int e4 = (((idx+1)%EDGE) == 0) ?
                0 : couplings[idx*9+4]*spins[m*N+n];  

    n = idx+EDGE-1;
    int e5 = ((idx%EDGE) == 0 || idx >= EDGE*(EDGE-1)) ?
                0 : couplings[idx*9+5]*spins[m*N+n];  

    n = idx+EDGE;
    int e6 = (idx >= EDGE*(EDGE-1)) ?
                0 : couplings[idx*9+6]*spins[m*N+n];  

    n = idx+EDGE+1;
    int e7 = (((idx+1)%EDGE) == 0 || idx >= EDGE*(EDGE-1)) ?
                0 : couplings[idx*9+7]*spins[m*N+n];  

    float difference = -(e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7 + couplings[idx*9+8]);
    int up = (m!=0) ? m-1: M-1;
    int down = (m!=M-1) ? m+1: 0;
    difference -= Jtrans*M*(spins[up*N+idx] + spins[down*N+idx]);
    difference *= 2*spins[m*N+idx]*beta;
    int row = (idx/EDGE)%2;
    int col = (idx%EDGE)%2;
    int group = 2*row + col;
    if (signal == group && difference > M*log(xorshift32(&randvals[idx]) / MAX)) {
        spins[m*N+idx] = -spins[m*N+idx];
    }
}

void usage() {
    printf("Usage:\n");
    printf("       ./Ising-opencl [spin configuration]\n");
    exit(0);
}

int relation (int a, int b) {
    switch (b-a) {
        case 0:
            return 8;
        case -EDGE-1:
            return 0;
        case -EDGE:
            return 1;
        case -EDGE+1:
            return 2;
        case -1:
            return 3;
        case 1:
            return 4;
        case EDGE-1:
            return 5;
        case EDGE:
            return 6;
        case EDGE+1:
            return 7;
        default:
            return -1;
    }
}

int main (int argc, char *argv[]) {
    if (argc != 2) 
        usage();

    // initialize parameters
    int *couplings, *couplings_buf, *spins_buf, *spins;
    couplings = (int*)malloc(N*9*sizeof(int));
    spins = (int*)malloc(M*N*sizeof(int));
    memset(couplings, '\0', N*9*sizeof(int));
    CUDA_CALL( cudaMalloc(&couplings_buf, N*9*sizeof(int)) );
    CUDA_CALL( cudaMalloc(&spins_buf, M*N*sizeof(int)) );

    // Read couplings file 
    FILE *instance = fopen(argv[1], "r");
    assert(instance != NULL);
    int a, b, w;
    fscanf(instance, "%d", &a);
    while (!feof(instance)) {
        fscanf(instance, "%d%d%d", &a, &b, &w);
        int r = relation(a, b);
        if (r == -1) {
            assert(false);
        } else {
            couplings[9*a+r] = w;
            r = relation(b, a);
            couplings[9*b+r] = w;
        }
    }
    fclose(instance);

    // copy couplings coefficients (when N is large, take lots of time)
    CUDA_CALL( cudaMemcpy(couplings_buf, couplings, N*9*sizeof(int), cudaMemcpyHostToDevice) );
    printf("Finish copying coefficients\n");

    // random number generation
    uint *randvals, *initRand;
    CUDA_CALL( cudaMalloc(&randvals, N * sizeof(uint)) );
    initRand = (uint*)malloc(N*sizeof(uint));
    for (int i = 0; i < N; i++)
        initRand[i] = i;
    CUDA_CALL( cudaMemcpy(randvals, initRand, N*sizeof(uint), cudaMemcpyHostToDevice) );

    // work group division
    dim3 grid(EDGE/THREADS, EDGE/THREADS), block(THREADS, THREADS);

    float increase = (8 - 1/(float)16) / (float)SWEEP;
    float G0 = 8.;
    int results[TIMES] = {0};
    printf("Start Annealing\n");
    for (int x = 0; x < TIMES; x++) {
        float beta = 1/(float)16;
        prepare_spins<<<grid, block>>>(spins_buf, randvals);
        for (int s = 0; s < SWEEP; s++) {
            float Gamma = G0*(1-(float)s/SWEEP);
            float Jtrans = -0.5*log(tanh((Gamma/M)*beta))/beta;
            for (int m = 0; m < M; m++) {
                for (int signal = 0; signal < 4; signal++) {
                    ising<<<grid, block>>>(spins_buf, couplings_buf, 
                                            signal, beta, randvals, m, Jtrans);
                }
            }
            beta += increase;
        }

        // Get result from device
        CUDA_CALL( cudaMemcpy(spins, spins_buf, M*N*sizeof(int), cudaMemcpyDeviceToHost) );

        int E = 0;
        for (int idx = 0; idx < N; idx++) {
            int n = idx-EDGE-1;
            int e0 = ((idx%EDGE) == 0 || idx < EDGE) ?
                        0 : couplings[idx*9+0]*spins[n];  
            n = idx-EDGE;
            int e1 = (idx < EDGE) ?
                        0 : couplings[idx*9+1]*spins[n];  
            n = idx-EDGE+1;
            int e2 = (((idx+1)%EDGE) == 0 || idx < EDGE) ?
                        0 : couplings[idx*9+2]*spins[n];  
            n = idx-1;
            int e3 = ((idx%EDGE) == 0) ?
                        0 : couplings[idx*9+3]*spins[n];  
            n = idx+1;
            int e4 = (((idx+1)%EDGE) == 0) ?
                        0 : couplings[idx*9+4]*spins[n];  
            n = idx+EDGE-1;
            int e5 = ((idx%EDGE) == 0 || idx >= EDGE*(EDGE-1)) ?
                        0 : couplings[idx*9+5]*spins[n];  
            n = idx+EDGE;
            int e6 = (idx >= EDGE*(EDGE-1)) ?
                        0 : couplings[idx*9+6]*spins[n];  
            n = idx+EDGE+1;
            int e7 = (((idx+1)%EDGE) == 0 || idx >= EDGE*(EDGE-1)) ?
                0 : couplings[idx*9+7]*spins[n];
            E += (e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7)*spins[idx];
        }
        E >>= 1;
        for (int idx = 0; idx < N; idx++)
            E += couplings[idx*9+8]*spins[idx];
        results[x] = -E;
    }

    printf("Finish Annealing\n");

    // Write results to file
    FILE *output;
    output = fopen("output.txt", "w");
    for (int i = 0; i < TIMES; i++)
         fprintf(output, "%d\n", results[i]);
    fclose(output);

    // Release Objects
    free(couplings);
    cudaFree(spins_buf);
    cudaFree(couplings_buf);
    return 0;
}