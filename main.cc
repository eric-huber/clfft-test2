#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>

// No need to explicitely include the OpenCL headers 
#include <clFFT.h>

const char* _data_file_name = "fft-data.txt";
const char* _fft_file_name  = "fft-forward.txt";
const char* _bak_file_name  = "fft-backward.txt";

void populate(size_t size, std::vector<float>& buf) {
    for (int i = 0; i < size * 2; ++i) {
        float t = i * .002;
        float amp = sin(M_PI * t);
        amp += sin(2 * M_PI * t);
        amp += sin(3 * M_PI * t); 
        buf.push_back(amp); 
    }
}

void write(std::string filename, std::vector<float>& buf) {
    std::ofstream ofs;
    ofs.open(filename);
    
    for (int i = 0; i < buf.size(); ++i) {
        ofs << buf[i] << std::endl;
    }
    
    ofs.close();   
}

void write_herm(std::string filename, std::vector<float>& buf) {
    std::ofstream ofs;
    ofs.open(filename);
    
    for (int i = 1; i < buf.size() / 2; i+=2) {
        float real = buf[i];
        float imag = buf[i+1];
        float amp = sqrt(pow(real, 2) + pow(imag, 2));
        float phase = atan2(imag, real);
        ofs << amp << ", " << phase << std::endl;
    }
    
    ofs.close();   
}

int main( void )
{
    cl_int                  err;
    cl_platform_id          platform = 0;
    cl_device_id            device = 0;
    cl_context_properties   props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context              ctx = 0;
    cl_command_queue        queue = 0;
    cl_mem                  bufX;
    std::vector<float>      X;
    cl_event                event = NULL;
    int                     ret = 0;
    size_t                  N = 8192;

    // FFT library realted declarations 
    clfftPlanHandle planHandle;
    clfftDim dim = CLFFT_1D;
    size_t clLengths[1] = {N};

    // Setup OpenCL environment. 
    err = clGetPlatformIDs( 1, &platform, NULL );
    err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );

    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
    queue = clCreateCommandQueue( ctx, device, 0, &err );

    // Setup clFFT. 
    clfftSetupData fftSetup;
    err = clfftInitSetupData(&fftSetup);
    err = clfftSetup(&fftSetup);

    // populate data
    populate(N, X);
    write(_data_file_name, X);

    // Prepare OpenCL memory objects and place data inside them. 
    bufX = clCreateBuffer( ctx, CL_MEM_READ_WRITE, N * 2 * sizeof(float), NULL, &err );

    err = clEnqueueWriteBuffer( queue, bufX, CL_TRUE, 0,
    N * 2 * sizeof(float), &X[0], 0, NULL, NULL );

    // Create a default plan for a complex FFT. 
    err = clfftCreateDefaultPlan(&planHandle, ctx, dim, clLengths);

    // Set plan parameters. 
    err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
    err = clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    err = clfftSetResultLocation(planHandle, CLFFT_INPLACE);

    // Bake the plan. 
    err = clfftBakePlan(planHandle, 1, &queue, NULL, NULL);

    // Execute the plan. 
    err = clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &queue, 0, NULL, NULL, &bufX, NULL, NULL);

    // Wait for calculations to be finished. 
    err = clFinish(queue);

    // Fetch results of calculations. 
    err = clEnqueueReadBuffer( queue, bufX, CL_TRUE, 0, N * 2 * sizeof(float), &X[0], 0, NULL, NULL );
    write_herm(_fft_file_name, X);

    // Release OpenCL memory objects. 
    clReleaseMemObject( bufX );

    X.empty();

    // Release the plan. 
    err = clfftDestroyPlan( &planHandle );

    // Release clFFT library. 
    clfftTeardown( );

    // Release OpenCL working objects. 
    clReleaseCommandQueue( queue );
    clReleaseContext( ctx );

    return ret;
}