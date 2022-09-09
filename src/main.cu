#include <iostream>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
using namespace cv;
//Kernel Image Processing
#define LOW 0.0f
#define HI 1.0f
#define MALLOCF(size) static_cast<float*>(malloc(size))
#define KWIDTH 3
__constant__ float kernel[KWIDTH*KWIDTH];



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

void printAH(float* A, int N, std::string text){
    std::cout << "Array: " << text << std::endl;
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
void convolution1(float* M, float* R, int w, int h){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float val = 0;
    if(col < w && row < h){
        
        int startCol = col - (KWIDTH/2);
        int startRow = row - (KWIDTH/2);
        for(int i=0; i < KWIDTH; i++){
            for(int j=0; j < KWIDTH;j++){
                int curRow = startRow + i;
                int curCol = startCol + j;

                if(curRow > -1 && curRow < h && curCol > -1 && curCol < w){
                    val += M[curRow * w + curCol] * kernel[i*KWIDTH + j];
                    
                }
            }
        }
        R[row * w + col] =  val; 
        
        
    }
    
}

void setKernelRidgeDetection(float* K){
    K[0] = -1;
    K[1] = -1;
    K[2] = -1;
    K[3] = -1;
    K[4] = 4;
    K[5] = -1;
    K[6] = -1;
    K[7] = -1;
    K[8] = -1;
}
void setKernelIdentity(float* K){
    K[0] = 0;
    K[1] = 0;
    K[2] = 0;
    K[3] = 0;
    K[4] = 1;
    K[5] = 0;
    K[6] = 0;
    K[7] = 0;
    K[8] = 0;
}




int main(){
    
    float *A, *B,*C;
    float *AH, *BH, *CH,*DH;
    int w = 128;
    int h = 128;
    int kernelS = KWIDTH*KWIDTH;
    int imageS = w*h;

    size_t imageSize = imageS * sizeof(float);
    size_t kernelSize = kernelS * sizeof(float);

    //Only square images
    int resultS = imageS;
    
    size_t resultSize = resultS * sizeof(float);

    AH = MALLOCF(imageSize);
    BH = MALLOCF(kernelSize);
    CH = MALLOCF(resultSize);
    DH = MALLOCF(kernelSize);

    //Init Kernel
    setKernelRidgeDetection(BH);

    //Alloc object in global memory
    cudaMalloc((void**)&A, imageSize);
    cudaMalloc((void**)&B, kernelSize);
    cudaMemcpyToSymbol(kernel, BH, kernelSize);
    //cudaMalloc((void**)&B, kernelSize);
    cudaMalloc((void**)&C, resultSize);

    randInitA(AH,imageS);
    //printAH(AH,imageS, "image");
    cudaMemcpyFromSymbol(DH,kernel,kernelSize);
    //cudaMemcpy(DH,kernel,kernelSize, cudaMemcpyDeviceToHost);
    //printAH(DH, kernelS, "kernel");

    cudaMemcpy(A,AH,imageSize,cudaMemcpyHostToDevice);
    //cudaMemcpy(B,BH,KWIDTH * sizeof(float),cudaMemcpyHostToDevice);
    dim3 dimMM(16,16,1);
    dim3 dimGrid(ceil(w/16.0), ceil(h/16.0), 1);
    convolution1<<<dimGrid,dimMM>>>(A,C,w,h);
    
    cudaMemcpy(CH,C,resultSize, cudaMemcpyDeviceToHost);
    
    printAH(CH, resultS,"result");
    
    Mat AImage(w,h,CV_32FC1, AH);
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", AImage);
    waitKey(0);

    Mat CImage(w,h,CV_32FC1, CH);
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", CImage);
    waitKey(0);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaDeviceReset();
    return 0;
}