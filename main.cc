#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>

// No need to explicitely include the OpenCL headers 
#include <clFFT.h>

const char* _data_file_name = "fft-data.txt";
const char* _fft_file_name  = "fft-forward.txt";
const char* _bak_file_name  = "fft-backward.txt";

size_t              N = 8192;

cl_context          _ctx = 0;
cl_command_queue    _queue = NULL;
clfftPlanHandle     _plan;
cl_mem              _buf;

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

void cl_init() {
    cl_int                  err;
    cl_platform_id          platform = 0;
    cl_device_id            device = 0;
    cl_context_properties   props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    
    // FFT library realted declarations 
    clfftDim dim = CLFFT_1D;
    size_t clLengths[1] = {N};

    // Setup OpenCL environment. 
    err = clGetPlatformIDs( 1, &platform, NULL );
    err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );

    props[1] = (cl_context_properties)platform;
    _ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
    _queue = clCreateCommandQueue( _ctx, device, 0, &err );

    // Setup clFFT. 
    clfftSetupData fftSetup;
    err = clfftInitSetupData(&fftSetup);
    err = clfftSetup(&fftSetup);
    
    // Prepare OpenCL memory objects and place data inside them. 
    _buf = clCreateBuffer( _ctx, CL_MEM_READ_WRITE, N * 2 * sizeof(float), NULL, &err );

    // Create a default plan for a complex FFT. 
    err = clfftCreateDefaultPlan(&_plan, _ctx, dim, clLengths);

    // Set plan parameters. 
    err = clfftSetPlanPrecision(_plan, CLFFT_SINGLE);
    err = clfftSetLayout(_plan, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    err = clfftSetResultLocation(_plan, CLFFT_INPLACE);

    // Bake the plan. 
    err = clfftBakePlan(_plan, 1, &_queue, NULL, NULL);
}

void cl_release() {
    cl_int                  err;
    
    // Release OpenCL memory objects. 
    clReleaseMemObject( _buf );

    // Release the plan. 
    err = clfftDestroyPlan( &_plan );

    // Release clFFT library. 
    clfftTeardown( );

    // Release OpenCL working objects. 
    clReleaseCommandQueue( _queue );
    clReleaseContext( _ctx );
}

int main( void )
{
    cl_int                  err;
    
    std::vector<float>      X;
    cl_event                event = NULL;
    int                     ret = 0;

    cl_init();

    // populate data
    populate(N, X);
    write(_data_file_name, X);

    err = clEnqueueWriteBuffer( _queue, _buf, CL_TRUE, 0,
              N * 2 * sizeof(float), &X[0], 0, NULL, NULL );

    // Execute the plan. 
    err = clfftEnqueueTransform(_plan, CLFFT_FORWARD, 1, &_queue, 0, NULL, NULL, &_buf, NULL, NULL);

    // Wait for calculations to be finished. 
    err = clFinish(_queue);

    // Fetch results of calculations. 
    err = clEnqueueReadBuffer( _queue, _buf, CL_TRUE, 0, N * 2 * sizeof(float), &X[0], 0, NULL, NULL );
    write_herm(_fft_file_name, X);


    X.empty();
    cl_release();

    return ret;
}