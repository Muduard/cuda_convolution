#include <iostream>
#include <cstdlib>
#include <cuda_runtime_api.h>
//Kernel Image Processing
#define LOW 1.0f
#define HI 10.0f
#define MALLOCF(size) static_cast<float*>(malloc(size))
#define KWIDTH 3
__constant__ float kernel[KWIDTH*KWIDTH];

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


int main(){
    
    float *A, *B,*C;
    float *AH, *BH, *CH,*DH;
    int w = 4;
    int h = 4;
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
    BH[0] = -1;
    BH[1] = -1;
    BH[2] = -1;
    BH[3] = -1;
    BH[4] = 4;
    BH[5] = -1;
    BH[6] = -1;
    BH[7] = -1;
    BH[8] = -1;

    //Alloc object in global memory
    cudaMalloc((void**)&A, imageSize);
    cudaMalloc((void**)&B, kernelSize);
    cudaMemcpyToSymbol(kernel, BH, kernelSize);
    //cudaMalloc((void**)&B, kernelSize);
    cudaMalloc((void**)&C, resultSize);

    randInitA(AH,imageS);
    printAH(AH,imageS, "image");
    cudaMemcpyFromSymbol(DH,kernel,kernelSize);
    //cudaMemcpy(DH,kernel,kernelSize, cudaMemcpyDeviceToHost);
    printAH(DH, kernelS, "kernel");

    cudaMemcpy(A,AH,imageSize,cudaMemcpyHostToDevice);
    //cudaMemcpy(B,BH,KWIDTH * sizeof(float),cudaMemcpyHostToDevice);
    dim3 dimMM(w,h,1);
    convolution1<<<1,dimMM>>>(A,C,w,h);

    cudaMemcpy(CH,C,resultSize, cudaMemcpyDeviceToHost);
    
    printAH(CH, resultS,"result");
    
    
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaDeviceReset();
    return 0;
}