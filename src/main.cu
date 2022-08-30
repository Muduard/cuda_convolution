#include <iostream>
#include <cstdlib>
#include <cuda_runtime_api.h>
//Kernel Image Processing
#define LOW 1.0f
#define HI 10.0f
#define MALLOCF(size) static_cast<float*>(malloc(size))

__device__
static void CheckCudaErrorAux (const char* file, unsigned line, const char *statement, cudaError_t err){
    if (err == cudaSuccess) return;
    printf("%s\n",cudaGetErrorString(err));
    
}

__global__ 
void printA(float* A, int N){
    int i=threadIdx.x;
    printf("i = %d: %f\n", i, A[i]);
}

__global__
void printM(float* A, int w, int h){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    printf("(%d,%d): %f\n", col,row,A[row*w + col]);

}

void printAH(float* A, int N){
    for(int i=0;i<N;i++)
        printf("i = %d: %f\n", i, A[i]);
}



__global__
void sumArr(float* A, float *B, float* C, int N){
    int i = threadIdx.x;
    if(i < N){
        C[i] = A[i] + B[i];
    }
}


void randInitA(float* A, int N ){
    for(int i=0;i<N;i++)
        A[i] = LOW + static_cast <float> (rand()) / ( static_cast <float> (RAND_MAX/(HI-LOW))) ;
}

__global__
void convolution(float* M, float* kernel, float* R, int w, int h){

}


int main(){
    
    float *A, *B,*C;
    float *AH, *BH;
    int w = 4;
    int h = 4;
    int kernelS = 3;
    int imageS = w*h;

    size_t imageSize = imageS * sizeof(float);
    size_t kernelSize = kernelS * kernelS * sizeof(float);

    int padding = 1;
    int stride = 0;
    //Only square images
    int resultS = (w + kernelS - 1) ;
    size_t resultSize = resultS * resultS * sizeof(float);

    AH = MALLOCF(imageSize);
    BH = MALLOCF(kernelSize);
    
    //Alloc object in global memory
    cudaMalloc((void**)&A, imageSize);
    cudaMalloc((void**)&B, kernelSize);
    cudaMalloc((void**)&C, resultSize);

    randInitA(AH,imageS);
    printAH(AH,imageS);

    randInitA(BH,kernelS);
    printAH(BH,kernelS);

    cudaMemcpy(A,AH,imageSize,cudaMemcpyHostToDevice);
    cudaMemcpy(B,BH,kernelSize,cudaMemcpyHostToDevice);
    dim3 dimMM(w,h,1);
    
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaDeviceReset();
    return 0;
}