#include <iostream>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "plotutils.hpp"

using namespace cv;
using namespace std::chrono;
//Kernel Image Processing
#define LOW 0
#define HI 255
#define MALLOCF(size) static_cast<uint8_t*>(malloc(size))
#define KWIDTH 3
#define GAUSSIAN_COEFF 0.0625f
__constant__ uint8_t kernel[KWIDTH*KWIDTH];

 #define gpuErr(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
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

void printAH(uint8_t* A, int N, std::string text){
    std::cout << "Array: " << text << std::endl;
    for(int i=0;i<N;i++)
        printf("i = %d: %d\n", i, A[i]);
}
void printAHF(double* A, int N, std::string text){
    std::cout << "Array: " << text << std::endl;
    for(int i=0;i<N;i++)
        printf("i = %d: %lf\n", i, A[i]);
}


void randInitA(uint8_t* A, int N ){
    for(int i=0;i<N;i++)
        A[i] = LOW + static_cast <uint8_t> (rand()) / ( static_cast <float> (RAND_MAX/(HI-LOW))) ;
}





__global__
void convolutionCuda(uint8_t* M, uint8_t* R, int w, int h, float kernelCoeff){
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
        R[row * w + col] = ceil(val * kernelCoeff);
        
        
    }
    
}

void convolutionCpu(uint8_t* M, uint8_t* R, uint8_t* kernel, int w, int h, float kernelCoeff){
    float val = 0;
    for(int col = 0; col < w;col++){
        for(int row=0;row < h;row++){
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
        R[row * w + col] = ceil( (val * kernelCoeff)); 
        }
    }
    
}


void setKernelGaussian(uint8_t* K){
    K[0] = 1;
    K[1] = 2;
    K[2] = 1;
    K[3] = 2;
    K[4] = 4;
    K[5] = 2;
    K[6] = 1;
    K[7] = 2;
    K[8] = 1;
}
void setKernelIdentity(uint8_t* K){
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


void imageConvolution2(){
    //Read image
    Mat img = imread("../images/sudoku.png", IMREAD_GRAYSCALE);
    imwrite("initial.png", img);
    std::vector<uint8_t> imageVector;
    imageVector.assign(img.data, img.data + img.total()*img.channels());
    
    uint8_t imageFloat[imageVector.size()];
    std::copy(imageVector.begin(), imageVector.end(), imageFloat);
    
    int w = img.cols;
    int h = img.rows;



    int kernelS = KWIDTH*KWIDTH;
    int imageS = w*h;
    
    size_t imageSize = imageS * sizeof(uint8_t);
    size_t kernelSize = kernelS * sizeof(uint8_t);
    size_t resultSize = imageS * sizeof(uint8_t);

    uint8_t *hostKernel;
    hostKernel = MALLOCF(kernelSize);
    
    uint8_t *convImage, *cudaConvImage, *cudaImage;
    //Init Kernel
    setKernelGaussian(hostKernel);

    //Alloc matrix for showing result
    convImage = MALLOCF(imageSize);

    //Alloc object in global memory
    cudaMalloc((void**)&cudaImage, imageSize);
    cudaMalloc((void**)&cudaConvImage, resultSize);
    cudaMemcpyToSymbol(kernel, hostKernel, kernelSize);

    cudaMemcpy(cudaImage, imageFloat,imageSize,cudaMemcpyHostToDevice);
    
    
    dim3 dimMM(16,16,1);
    dim3 dimGrid(ceil(w/16.0), ceil(h/16.0), 1);
    auto start = high_resolution_clock::now();
    convolutionCuda<<<dimGrid,dimMM>>>(cudaImage,cudaConvImage,w,h, GAUSSIAN_COEFF);
    auto end = high_resolution_clock::now();

    cudaMemcpy(convImage,cudaConvImage,resultSize, cudaMemcpyDeviceToHost);
    
    Mat img2(h,w,CV_8U,convImage);
    imwrite("convoluted.png", img2);

    cudaFree(cudaImage);
    cudaFree(cudaConvImage);
    cudaDeviceReset();
    
}

int matrixConvolutionCuda(int size){
    int kernelS = KWIDTH*KWIDTH;
    int imageS = size * size;

    size_t imageSize = imageS * sizeof(uint8_t);
    size_t kernelSize = kernelS * sizeof(uint8_t);

    uint8_t *hostKernel;
    hostKernel = MALLOCF(kernelSize);
    
    uint8_t *convImage, *cudaConvImage, *cudaImage, *image;
    //Init Kernel
    setKernelGaussian(hostKernel);

    image = MALLOCF(imageSize);
    convImage = MALLOCF(imageSize);
    
    randInitA(image,imageS);

    //Alloc object in global memory
    cudaMalloc((void**)&cudaImage, imageSize);
    cudaMalloc((void**)&cudaConvImage, imageSize);
    cudaMemcpyToSymbol(kernel, hostKernel, kernelSize);

    cudaMemcpy(cudaImage, image,imageSize,cudaMemcpyHostToDevice);
    
    dim3 dimMM(16,16,1);
    dim3 dimGrid(ceil(size/16.0), ceil(size/16.0), 1);
    auto start = high_resolution_clock::now();
    convolutionCuda<<<dimGrid,dimMM>>>(cudaImage,cudaConvImage,size,size, GAUSSIAN_COEFF);
    auto end = high_resolution_clock::now();

    cudaMemcpy(convImage,cudaConvImage,imageSize, cudaMemcpyDeviceToHost);

    cudaFree(cudaImage);
    cudaFree(cudaConvImage);
    cudaDeviceReset();
    
    return duration_cast<microseconds>(end-start).count();
}

int matrixConvolution(int size){
    int kernelS = KWIDTH*KWIDTH;
    int imageS = size * size;
    
    size_t imageSize = imageS * sizeof(uint8_t);
    size_t kernelSize = kernelS * sizeof(uint8_t);

    uint8_t *kernel, *image, *convImage;
    image = MALLOCF(imageSize);
    convImage = MALLOCF(imageSize);
    kernel = MALLOCF(kernelSize);
    setKernelGaussian(kernel);
    randInitA(image,imageS);
    auto start = high_resolution_clock::now();
    convolutionCpu(image,convImage,kernel,size,size, GAUSSIAN_COEFF);
    auto end = high_resolution_clock::now();

    return duration_cast<microseconds>(end-start).count();
}

void Test(){

    std::vector<double> timeCpu,timeCuda, sizes;

    //Gather data from various image sizes
    for(int size = 8; size <= 32768/2; size*=2){
        timeCuda.push_back(matrixConvolutionCuda(size));
        std::cout << "size: " << size << std::endl;
        timeCpu.push_back(matrixConvolution(size));
        sizes.push_back(size);
    }

    plotSpeed(&timeCpu,&timeCuda,&sizes);
    //imageConvolution2();
}


int main(){
    Test();
    return 0;
}