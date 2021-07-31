//看起來好像是對的 嗎ＱＷＱ
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
#define EDGE 32
#define N (EDGE*EDGE) // N = EDGE * EDGE
#define M 16 // 先從16開始
#define TIMES 1//10
#define STEP 1 //100

#define NQuarter N/4
#define MHalf M/2
#define totalBlkNum N/64
#define totalNumFlipOneTime (NQuarter*MHalf)

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
int n64Idx (int n){
    int row = n % EDGE; // n 在原graph上的 row idx
    int col = n / EDGE; // n 在原graph上的 col idx
    int n64Idx = row % 8 + (col % 8) * 8; // (col%8) 在64*64大小的Col，(row % 8) 在64*64大小的row
    return n64Idx;
}

int countBlkNum (int a){
    int aRow = a % EDGE;
    int aCol = a / EDGE;
    int aBlkRow = aRow / 8;
    int aBlkCol = aCol / 8;
    int aBlkNum = aBlkRow + aBlkCol * (EDGE / 8); 
    return aBlkNum;   
}

int judgeColor (int a){
    int aRow = a % EDGE; // n 在原graph上的 row idx
    int aCol = a / EDGE; // n 在原graph上的 col idx
    // if (aRow % 2 == 0 && aCol % 2 == 0) { //綠色
    //     return 0;
    // } else if (aRow % 2 != 0 && aCol % 2 == 0) { // 紅色
    //     return 1;
    // } else if (aRow % 2 == 0 && aCol % 2 != 0) { // 藍色
    //     return 2;
    // } else { // 黑色
    //     return 3;
    // }   
    // combine
    return ((aRow%2)+2*(aCol%2));
}
int couplingIdx (int a) {
    int a64Idx = n64Idx(a);
    int aRow = a % EDGE, aCol = a / EDGE; 
    int colorMinus = ((aRow%2)+(aCol%2)*8);// Green: 0, REd: 1, Blue: 8, Black: 9
    int new_a = ((a64Idx - colorMinus) % 8) / 2 + 4 * ((a64Idx - colorMinus) / 16); // new_a: 在64&64的相同顏色中他是第幾個，總共16個 for one color
    int aBlkNum = countBlkNum(a);
    new_a += aBlkNum*16;//block累積起來的相同顏色
    new_a += judgeColor(a)*(N/4);//因為顏色累積起來的相同顏色
    return new_a;
}
int couplingMatrixIdx (int a, int b){
    // 讀檔案的時候就是column major
    // int a64Idx = n64Idx(a);
    // int b64Idx = n64Idx(b);
    // 假設a是col，以a為主，先換成 64-idx
    int aOnCouplings = couplingIdx(a);
    int bOnCouplings = couplingIdx(b);

    int newCouplingPosition = bOnCouplings + aOnCouplings * N; //在64*64裡面的位置
    return newCouplingPosition;
}

void construct_couplings (int a, int b, int w, float *couplings){
    int aBlkNum = countBlkNum(a);
    int bBlkNum = countBlkNum(b);
    int newPosition = 0;
    newPosition = couplingMatrixIdx(a, b);
    couplings[newPosition] = w;
}

int spinMatrixIdx (int n, int m){ //對了
    // 讀檔案的時候就是column major
    int a64Idx = n64Idx(n);
    int new_a = 0, spinIdx = 0;
    /*if (aRow % 2 == 0 && aCol % 2 == 0) { //綠色
        printf("Green\n");
        new_a = (a64Idx % 8) / 2 + 4 * (a64Idx / 16);
        if(m % 2 == 0){ // 偶數層
            spinIdx = new_a + 16 * blkNum;//blk累積，color累積
        } else {
            spinIdx = new_a + 16 * blkNum + (16*1)*totalBlkNum*(M/2) * 4;
        }
    } else if (aRow % 2 != 0 && aCol % 2 == 0) { // 紅色
        printf("Red\n");
        new_a = ((a64Idx - 1) % 8) / 2 + 4 * ((a64Idx - 1) / 16);
        if(m % 2 == 0){ // 偶數層
            spinIdx = new_a + 16 * blkNum + (16*1)*totalBlkNum*(M/2) * 1;//blk累積，color累積
        } else {
            spinIdx = new_a + 16 * blkNum + (16*1)*totalBlkNum*(M/2) * 5;
        }
    } else if (aRow % 2 == 0 && aCol % 2 != 0) { // 藍色
        printf("Blue\n");
        new_a = ((a64Idx - 8) % 8) / 2 + 4 * ((a64Idx - 8) / 16);
        if(m % 2 == 0){ // 偶數層
            spinIdx = new_a + 16 * blkNum + (16*1)*totalBlkNum*(M/2) * 2;//blk累積，color累積
        } else {
            spinIdx = new_a + 16 * blkNum + (16*1)*totalBlkNum*(M/2) * 6;
        }
    } else { // 黑色
        printf("Black\n");
        new_a = ((a64Idx - 9) % 8) / 2 + 4 * ((a64Idx - 9) / 16);
        if(m % 2 == 0){ // 偶數層
            spinIdx = new_a + 16 * blkNum + (16*1)*totalBlkNum*(M/2) * 3;//blk累積，color累積
        } else {
            spinIdx = new_a + 16 * blkNum + (16*1)*totalBlkNum*(M/2) * 7;
        }
    }*/
    //合併上述
    int aRow = n % EDGE, aCol = n / EDGE; 
    int colorMinus = ((aRow%2)+(aCol%2)*8);
    new_a = ((a64Idx - colorMinus) % 8) / 2 + 4 * ((a64Idx - colorMinus) / 16);//new_a
    // printf("n = %d, m = %d, new_a = %d\n", n, m, new_a);
    
    int blkNum = countBlkNum(n);
    int color = judgeColor(n);
    int newCol = ((M > 16) ? (MHalf*totalBlkNum) : (16*totalBlkNum))*(color+(m%2)*4);//累積大的
    newCol += 16*blkNum;//在找到block，一個block有16條
    newCol += m/2;//在找到trotter
    int newRow = new_a;
    spinIdx = IDX2C( newRow, newCol, 16);

    return spinIdx;

}

void check_spinMatrixIdx(){
    int newIdx = 0;
    printf("Continuous forth on the trotter 0:\n");
    for(int i = 0; i < 4; i++){
        printf("original, n = %d, m = %d\n", i+(i/2)*30, 0);
        newIdx = spinMatrixIdx(i+(i/2)*30, 0);
        printf("newIdx = %d\n---\n", newIdx);
    }
    printf("@@@@\nContinuous forth on the trotter 1:\n");
    for(int i = 0; i < 2*M; i++){
        printf("original, n = %d, m = %d\n", i+(i/2)*30, 1);
        newIdx = spinMatrixIdx(i+(i/2)*30, 1);
        printf("newIdx = %d\n---\n", newIdx);
    }
}
void check_couplings(float *couplings, float *couplings_fp32){
    // cudaErrCheck (cudaMemcpy(couplings, couplings_fp32, N*N*sizeof(float), cudaMemcpyDeviceToHost));
    printf("\ncheck_couplings:\n");
    // for (int n = 0; n < N; n++){
    //     for(int k = 0; k < N; k++){
    //         printf("%d ", (int)couplings[IDX2C(n,k,N)]);
    //         // if(couplings[IDX2C(n,k,N)] != 0){
    //         //     printf("row = %d, col = %d\n", n, k);
    //         // }
    //     }
    //     printf("\n");
    // }
    for(int blkNum = 0; blkNum < totalBlkNum; blkNum++){
        printf("\nblock  = %d\n", blkNum);
        for (int row = 0; row <256; row++){
            for(int col = 0; col < 256; col++){
                int colAdd = blkNum / 4;
                int rowAdd = blkNum % 4;
                printf("%d ", (int)couplings[IDX2C(row+rowAdd*256,col+colAdd*256,N)]);
            }
            printf("\n");
        }
    }
} 

void check_matrixA (float *matrixA){
    for(int outBlkNum = 0; outBlkNum < totalBlkNum; outBlkNum++){
        printf("out block = %d\n", outBlkNum);
        for(int innerBlkNum = 0; innerBlkNum < totalBlkNum; innerBlkNum++){
            printf("inner block = %d\n", innerBlkNum);
            for(int i = 0; i < 16; i++){
                for(int j = 0; j < 16; j++){
                    printf("%d ", (int)matrixA[IDX2C(i, j, 16) + outBlkNum*256*16 + innerBlkNum*256]);
                }
                printf("\n");
            }      
        }
    }
  
}

void construct_matrixA (float *couplings, float *matrixA){
    int count = 0;
    for(int blkNum = 0; blkNum < totalBlkNum; blkNum++){
        for(int innerBlkNum = 0; innerBlkNum < totalBlkNum; innerBlkNum++){
            for (int row = 0; row <16; row++){
                for(int col = 0; col < 16; col++){
                    int colAdd = blkNum / 4;
                    int rowAdd = blkNum % 4;
                    matrixA[count] = couplings[IDX2C(row+rowAdd*256+innerBlkNum*16, col+colAdd*256+innerBlkNum*16, N)];
                    count ++;
                }
            }  
        }
    }
    int matrixAInitIdx = 0;
    for(int ;;){
        for(int ;;){
            matrixA[IDX2C(,,)] = couplings[IDX2C(,,)];
        }
    }
}

void construct_spin(float *spin, float *spin_fp32, int total_spins){
    float x;
    if(M > 16){
        for (int n = 0; n < N; n++){
            for(int m = 0; m < M; m++){
                x = ((float)rand()/(float)(RAND_MAX)) * 1.0;    
                spin[IDX2C(n,m,N)] = ((x>=0.5) ? (float)1. : (float)-1.);
            }
        }        
    } else {
        for(int i = 0; i < 16; i++){
            for(int j = 0; j < (32*N)/16; j++){
                x = ((float)rand()/(float)(RAND_MAX)) * 1.0;    
                if((j%16) < (MHalf)){
                    // printf("IDX2C(%d, %d, 16) = %d\n", i,j,IDX2C(i, j, 16));
                    spin[IDX2C(i, j, 16)] = ((x>=0.5) ? (float)1. : (float)-1.);
                } else {
                    spin[IDX2C(i, j, 16)] = 0;
                }
            }
        }

    }
}

void check_spin (float *spin, float *spin_fp32){
    cudaErrCheck (cudaMemcpy(spin, spin_fp32, M*N*sizeof(float), cudaMemcpyDeviceToHost));
    printf("\ncheck_spin:\n");
    for(int i = 0; i < 16; i++){
        for(int j = 0; j < (32*N)/16; j++){
            printf("%d ", (int)spin[IDX2C(i,j,16)]);
        }
        printf("\n");
    }
}


void check_delta_H (float *delta_H, float *delta_H_fp32){
    cudaErrCheck ( cudaMemcpy(delta_H, delta_H_fp32, M*N*sizeof(float), cudaMemcpyDeviceToHost));
    printf("\ncheck print delta_H\n");
    for (int n = 0; n < N; n++){
        for (int m = 0; m < M; m++){
            printf("%d ", (int)delta_H[IDX2C(n,m,N)]);
        }
        printf("\n");
    }
}

void construct_delta_H (cublasHandle_t cublasHandle, float *matrixA, float *matrixA_fp32, float *spin, float *spin_fp32, float *delta_H, float *delta_H_fp32) {
    float alpha = 1.0f, beta = 1.0f;    
    int matrixAIdx = 0;
    int spinIdx = 0;
    int delta_HIdx = 0;
    int color = 0;
    // even trotters = 0, odd trotters = 1
    for(int evenOdd = 0; evenOdd < 2; evenOdd ++){
        matrixAIdx = 0; 
        for(int outBlkNum = 0; outBlkNum < totalBlkNum; outBlkNum++){
            color = outBlkNum/4;
            delta_HIdx = spinMatrixIdx((EDGE*(color/2)+color%2), evenOdd); // 要換顏色，不用換顏色
            spinIdx = delta_HIdx; 
            for(int innerBlkNum = 0; innerBlkNum < totalBlkNum; innerBlkNum++){
                cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                        16, 16, 16,
                                        &alpha, 
                                        matrixA_fp32 + matrixAIdx, CUDA_R_32F, 16,
                                        spin_fp32 + spinIdx, CUDA_R_32F,  16, 
                                        &beta, 
                                        delta_H_fp32 + delta_HIdx, CUDA_R_32F, 16,
                                        CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                matrixAIdx += 256;
                spinIdx += 256;
                delta_HIdx += 256;
            }
        }
    }
    cudaErrCheck(cudaMemcpy(delta_H, delta_H_fp32, M*N*sizeof(float), cudaMemcpyDeviceToHost));
}

void flip (int color, float *couplings, float *couplings_fp32, float *spin, float *spin_fp32, float *delta_H, float *delta_H_fp32, float J_perp, float beta) {
    // even
    float delta = 0., new_spin = 0.;
    // 起始flip spin：(1) 偶數：第零層、(2) 奇數：第一層；
    // 不同顏色起始spin n：(1) 綠色：0、(2) 紅色：1、(3) 藍色：32、(4)黑色：33。
    int startN = (EDGE*(color/2)+color%2);
    int fIdx = spinMatrixIdx(startN, 0);
    // printf("init fIdx = %d\n", fIdx);
    int blankBlk = ((M > 16) ? (0) : (16*(16 - MHalf)));
    int bigBlk = ((M > 16) ? (totalNumFlipOneTime) : (16*16*totalBlkNum));
    // printf("blankBlk = %d, bigBlk = %d\n", blankBlk, bigBlk);

    for(int blkNum = 0; blkNum < totalBlkNum; blkNum ++){
        for(int m = 0; m < M; m+=2){
            for(int i = 0; i < 16; i++){
                // printf("startN = %d, fIdx = %d, m = %d, i = %d, spin[%d] = %d\n", startN, fIdx, m, i, fIdx, (int)spin[fIdx]);

                assert(spin[fIdx] != 0);
                gpuErrchk(cudaMemcpy(&delta, delta_H_fp32+fIdx, 1*sizeof(float), cudaMemcpyDeviceToHost));

                int upperIdx = ((m == 0) ? (fIdx + 4*bigBlk + 16*(MHalf-1)) : (fIdx + 4*bigBlk - 16));
                int lowerIdx = fIdx + 4*bigBlk;
                // printf("fIdx = %d, upperIdx = %d,  = %d\n", fIdx, upperIdx, lowerIdx);應該是對的
                assert(spin[upperIdx] != 0);
                assert(spin[lowerIdx] != 0);

                delta = 2*M*spin[fIdx]*(delta - M*J_perp*(spin[upperIdx] + spin[lowerIdx]));

                if ( (-log(rand() / (float) RAND_MAX) / beta) > delta ) {
                    spin[fIdx] = -spin[fIdx];
                    new_spin = spin[fIdx]; 
                    gpuErrchk(cudaMemcpy(spin_fp32 + fIdx, &new_spin, 1*sizeof(float), cudaMemcpyHostToDevice));                
                }
                fIdx ++;
            }
        }
        fIdx += blankBlk;//update
    }

    // odd
    color = (color + 1) % 4;
    startN = (EDGE*(color/2)+color%2);    
    fIdx = spinMatrixIdx(startN, 1);
    for(int blkNum = 0; blkNum < totalBlkNum; blkNum ++){
        for(int m = 1; m < M; m+=2){
            for(int i = 0; i < 16; i++){

                assert(spin[fIdx] != 0);
                // if(spin[fIdx] == 0){
                //     printf("startN = %d, fIdx = %d, m = %d, i = %d, spin[%d] = %d\n", startN, fIdx, m, i, fIdx, (int)spin[fIdx]);
                // }
                gpuErrchk(cudaMemcpy(&delta, delta_H_fp32+fIdx, 1*sizeof(float), cudaMemcpyDeviceToHost));

                int upperIdx = fIdx - 4*bigBlk;
                int lowerIdx = ((m == (M-1)) ? (fIdx - 4*bigBlk - 16*(MHalf-1)) : (fIdx - 4*bigBlk + 16));
                // printf("fIdx = %d, upperIdx = %d,  = %d\n", fIdx, upperIdx, lowerIdx);應該是對的

                delta = 2*M*spin[fIdx]*(delta - M*J_perp*(spin[upperIdx] + spin[lowerIdx]));
                assert(spin[upperIdx] != 0);
                assert(spin[lowerIdx] != 0);
                // if(spin[lowerIdx] == 0){
                //     printf("startN = %d, lowerIdx = %d, m = %d, i = %d, spin[%d] = %d\n", startN, lowerIdx, m, i, lowerIdx, (int)spin[lowerIdx]);
                // }
                if ( (-log(rand() / (float) RAND_MAX) / beta) > delta ) {
                    spin[fIdx] = -spin[fIdx];
                    new_spin = spin[fIdx]; 
                    gpuErrchk(cudaMemcpy(spin_fp32 + fIdx, &new_spin, 1*sizeof(float), cudaMemcpyHostToDevice));                
                }
                fIdx ++;
            }
        }
        fIdx += blankBlk;//update
    }
}

float calculate_E (float *couplings, float *couplings_fp32, float *spin, float *spin_fp32){
    // cudaErrCheck(cudaMemcpy(spin, spin_fp32, M*N*sizeof(float), cudaMemcpyDeviceToHost));
    int E = 0;
    for (int i = 0; i < N; i++){
        for (int j = i+1; j < N; j++){
            E += -spin[spinMatrixIdx(i, 0)]*spin[spinMatrixIdx(j, 0)]*couplings[couplingMatrixIdx(i, j)];
            // if(spin[spinMatrixIdx(j, 0)] == 0)
            //     printf("spin[spinMatrixIdx(%d, 0)] = %d, spin[spinMatrixIdx(%d, 0)] = %d\n", i, (int)spin[spinMatrixIdx(i, 0)], j, (int)spin[spinMatrixIdx(j, 0)]);
        }
    }
    // printf("test\n:");
    // int E = 0;
    // for(int l = 0; l < M; l++){
    //     for (int i = 0; i < N; i++){
    //         for (int j = i+1; j < N; j++){
    //             E += -spin[spinMatrixIdx(i, l)]*spin[spinMatrixIdx(j, l)]*couplings[couplingMatrixIdx(i, j)];
    //             if(spin[spinMatrixIdx(i, l)] == 0)
    //                 printf("spin[spinMatrixIdx(%d, %d)] = %d, spin[spinMatrixIdx(%d, %d)] = %d\n", i, l, (int)spin[spinMatrixIdx(i, 0)], j, l, (int)spin[spinMatrixIdx(j, 0)]);
    //         }
    //     }
    // }
    return E;
}

int main (int argc, char *argv[]) {
    cublasHandle_t cublasHandle;
    cudaEvent_t startcublas;
    cudaEvent_t stopcublas;

    cudaErrCheck(cudaEventCreate(&startcublas));
    cudaErrCheck(cudaEventCreate(&stopcublas));
    cublasErrCheck(cublasCreate(&cublasHandle));
    
    // Initialize couplings
    float *couplings;  
    couplings = (float*)malloc(N*N*sizeof(float));
    memset(couplings, 0, N*N*sizeof(float));

    float *couplings_fp32;
    cudaErrCheck(cudaMalloc((void**)&couplings_fp32, N*N*sizeof(float)));

    // int couplingResNum = 2*((2*(EDGE-1)*(2*(EDGE-1)+1)) - (2*105)*(N/64));
    // printf("couplingResNum = %d\n", couplingResNum);

    // Read files
    FILE *instance = fopen(argv[1], "r");
    assert(instance != NULL);
    int a, b, w, total_spins, total_couplings;
    fscanf(instance, "%d%d", &total_spins, &total_couplings);
    while (total_couplings --) {
        fscanf(instance, "%d%d%d", &a, &b, &w);
        construct_couplings(a, b, w, couplings);    
        construct_couplings(b, a, w, couplings);    
    }
    fclose(instance);


    // copy couplings to target device
    //這行感覺之後可以槓掉
    cudaErrCheck ( cudaMemcpy(couplings_fp32, couplings, N*N*sizeof(float), cudaMemcpyHostToDevice));
    // check couplings, OKOK!
    // check_couplings(couplings, couplings_fp32);
    
    float *matrixA;  
    matrixA = (float*)malloc(64*N*sizeof(float));
    memset(matrixA, 0, 64*N*sizeof(float));

    float *matrixA_fp32;
    cudaErrCheck(cudaMalloc((void**)&matrixA_fp32, 64*N*sizeof(float)));
    
    construct_matrixA(couplings, matrixA);
    //check matirxA, OKOK
    check_matrixA (matrixA);
    cudaErrCheck(cudaMemcpy(matrixA_fp32, matrixA, 64*N*sizeof(float), cudaMemcpyHostToDevice));

    int trottersMatrixB = ((M > 16) ? (M): (32));

    // Initialize spin
    float *spin;
    spin = (float*)malloc(trottersMatrixB*N*sizeof(float));
    memset(spin, 0, trottersMatrixB*N*sizeof(float)); // must initialize, since there are some places not 0
    
    float *spin_fp32;
    cudaErrCheck(cudaMalloc((void**)&spin_fp32, trottersMatrixB*N*sizeof(float)) );
    cudaErrCheck(cudaMemcpy(spin_fp32, spin, trottersMatrixB*N*sizeof(float), cudaMemcpyHostToDevice));

    float *delta_H;
    delta_H = (float*)malloc(trottersMatrixB*N*sizeof(float));
    memset(delta_H, 0, trottersMatrixB*N*sizeof(float));
    
    float *delta_H_fp32;
    cudaErrCheck(cudaMalloc((void**)&delta_H_fp32, trottersMatrixB*N*sizeof(float)));
    cudaErrCheck(cudaMemcpy(delta_H_fp32, delta_H, trottersMatrixB*N*sizeof(float), cudaMemcpyHostToDevice));

    // TC, using tensor core
    cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH)); 


    // Parameters init
    float results[TIMES] = {0.};
    float used_time[TIMES] = {0.};
    float increase = (8 - 1/(float)16) / (float)STEP;
    float G0 = 8.;

    for (int t = 0; t < TIMES; t++) {
        float beta = 1/(float)16; //bete = 1/Time
        
        //init spin
        construct_spin(spin, spin_fp32, total_spins);
        // check spin, OKOK!
        // check_spin(spin, spin_fp32); 
        cudaErrCheck (cudaMemcpy(spin_fp32, spin, M*N*sizeof(float), cudaMemcpyHostToDevice));

        // Construct the initial energy
        construct_delta_H(cublasHandle, matrixA, matrixA_fp32, spin, spin_fp32, delta_H, delta_H_fp32);
        // check delta_H
        // check_delta_H(delta_H, delta_H_fp32); 

        float initE = calculate_E(couplings, couplings_fp32, spin, spin_fp32);
        printf("time = %d, initE = %f\n", t, initE);

        // Current cost time
        clock_t begin, end;
        begin = clock();

        for (int p = 0; p < STEP; p++) {
            float Gamma = G0*(1.-(float)p/(float)STEP);
            float J_perp = -0.5*log(tanh((Gamma/M)*beta))/beta;
            for(int f = 0; f < 4; f++){ // f: flip
                flip(f, couplings, couplings_fp32, spin, spin_fp32, delta_H, delta_H_fp32, J_perp, beta); 
                construct_delta_H(cublasHandle, matrixA, matrixA_fp32, spin, spin_fp32, delta_H, delta_H_fp32);
            }
            float tmpE = calculate_E(couplings, couplings_fp32, spin, spin_fp32);
            printf("step: %d, Energy: %10lf\n", p, tmpE);
         beta += increase;
        } 
        cudaDeviceSynchronize();
        
        
        end = clock();
        double duration = (double)(end-begin) / CLOCKS_PER_SEC;

        used_time[t] = duration;

        float E = calculate_E(couplings, couplings_fp32, spin, spin_fp32);
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
    free(matrixA);
    cudaFree(couplings_fp32);
    cudaFree(spin_fp32);
    cudaFree(delta_H_fp32);
    cudaFree(matrixA_fp32);
    
    return 0;
}