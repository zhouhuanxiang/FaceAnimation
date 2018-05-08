#ifndef MATRIX_OPENCL_H_
#define MAAIRX_OPENCL_H_

#include <stdio.h>
#include <clBLAS.h>

class OpenCLHelper
{
public:
	OpenCLHelper()
	{
		cl_int err;
		cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };

		/* Setup OpenCL environment. */
		err = clGetPlatformIDs(1, &platform, NULL);
		if (err != CL_SUCCESS) {
			printf("clGetPlatformIDs() failed with %d\n", err);
		}

		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
		if (err != CL_SUCCESS) {
			printf("clGetDeviceIDs() failed with %d\n", err);
		}

		props[1] = (cl_context_properties)platform;
		ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
		if (err != CL_SUCCESS) {
			printf("clCreateContext() failed with %d\n", err);
		}

		queue = clCreateCommandQueue(ctx, device, 0, &err);
		if (err != CL_SUCCESS) {
			printf("clCreateCommandQueue() failed with %d\n", err);
			clReleaseContext(ctx);
		}

		/* Setup clblas. */
		err = clblasSetup();
		if (err != CL_SUCCESS) {
			printf("clblasSetup() failed with %d\n", err);
			clReleaseCommandQueue(queue);
			clReleaseContext(ctx);
		}
	}

public:
	cl_platform_id platform;
	cl_device_id device;
	cl_context ctx;
	cl_command_queue queue;
};

class DeMatrixCL
{
public:
	DeMatrixCL()
		:rows(0), cols(0), ptr(nullptr)
	{}

	void SetSize(cl_context ctx, int r, int c, cl_mem_flags flag = CL_MEM_READ_WRITE)
	{
		rows = r;
		cols = c;

		/* Prepare OpenCL memory objects and place matrices inside them. */
		cl_int err;
		ptr = clCreateBuffer(ctx, flag, rows * cols * sizeof(double), NULL, &err);
	}

	void SetData(cl_command_queue queue, double *p)
	{
		clEnqueueWriteBuffer(queue, ptr, CL_TRUE, 0, rows * cols * sizeof(double), p, 0, NULL, NULL);
	}

	void GetData(cl_command_queue queue, int r, int c, double *p)
	{
		cl_event event = NULL;
		clEnqueueReadBuffer(queue, ptr, CL_TRUE, 0, r * c * sizeof(double), p, 0, NULL, &event);
		clWaitForEvents(1, &event);
	}
public:
	int rows;
	int cols;
	cl_mem ptr;
};

class SpMatrixCL
{
public:

};

#endif