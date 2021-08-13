// prepare: awk '{print $3}' time.txt > timeAndEnergy.txt
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define row 7
#define col 8

int main (int argc, char* argv[]){
    float *tmp; // cpu    
    tmp = (float*)malloc(col*row*2*sizeof(float));
    memset(tmp, 0, col*row*2*sizeof(float));


    FILE *instance = fopen(argv[1], "r");
    int totalNum = col*row;
    float time = 0., energy = 0.;
    int tmp_idx = 0;
    while (totalNum --) {
        fscanf(instance, "%f%f", &time, &energy);
        //printf("time = %f, energy = %f\n", time, energy);
        tmp[tmp_idx] = time;
        tmp_idx++;
        tmp[tmp_idx] = energy;
        tmp_idx++;
       // printf("tmp[%d] = %f, tmp[%d] = %f\n", tmp_idx-2,  tmp[tmp_idx-2], tmp_idx-1, tmp[tmp_idx-1]);
    }
    fclose(instance);

	printf("\\hline \n");    
    printf("\\textbf{Spin} & \\textbf{Data} & 4 & 8 & 16 & 32 & 64 & 128 & 256 & 512\\\\ \n");
	printf("\\hline \n");		
    int N = 1024;
    for(int r = 0; r < row; r++){
    	printf("\\multirow{2}{*}{%d} &  Time   ", N);
    	for(int c = 0; c < col*2; c+=2){
    		printf(" & %f", tmp[(r*2*col)+c]);
    	}
    	printf("\\\\ \n");
    	// printf("\\hline \n");
       	printf("&  Energy ");
    	for(int c = 0; c < col*2; c+=2){
    		printf(" & %f", tmp[(r*2*col)+(c+1)]);
    	}
    	printf("\\\\ \n");
       	printf("\\hline \n");

    	N*=2;
    }


	return 0;
}