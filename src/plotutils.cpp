//#include "plotutils.hpp"
#include <iostream>
#include <vector>
#include <valarray>
#include <sciplot/sciplot.hpp>
using namespace sciplot;
void plotSpeed(std::vector<double> *cpu, std::vector<double> *cuda, std::vector<double>* sizes ){
    Vec sizesVec(sizes->data(),sizes->size());
    Vec cpuVec(cpu->data(),cpu->size());
    Vec cudaVec(cuda->data(),cuda->size());
    
    std::vector<double> speedup;
    for(int i=0;i<sizes->size();i++){
        speedup.push_back(cpu->at(i)/cuda->at(i));
    }
    Vec speedupVec(speedup.data(),speedup.size());
    Plot2D plot;
    
    plot.drawCurve(sizesVec,cpuVec).label("Cpu time");
    plot.legend().atOutsideTopRight();
    plot.xlabel("Size");
    plot.ylabel("Time");
    //plot.xtics().logscale(10);
    plot.xtics().automatic();

    Figure fig = {{ plot}};
    Canvas canvas = {{ fig }};
    canvas.size(600, 600);
    // Show the canvas in a pop-up window
    canvas.show();
    canvas.save("cpu.pdf");

    Plot2D plot2;
    plot2.drawCurve(sizesVec,cudaVec).label("Cuda time");
    plot2.legend().atOutsideTopRight();
    plot2.xlabel("Size");
    plot2.ylabel("Time");
    plot2.xtics().automatic();
    fig = {{ plot2}};
    canvas = {{ fig }};
    canvas.size(600, 600);
    // Show the canvas in a pop-up window
    canvas.show();
    canvas.save("cuda.png");

    Plot2D plot3;
    plot3.drawCurve(sizesVec,speedupVec).label("Speedup");
    plot3.legend().atOutsideTopRight();
    plot3.xlabel("Size");
    plot3.ylabel("Speedup");
    plot3.xtics().automatic();
    fig = {{ plot3}};
    canvas = {{ fig }};
    canvas.size(600, 600);
    // Show the canvas in a pop-up window
    canvas.show();
    canvas.save("speedup.png");

}

