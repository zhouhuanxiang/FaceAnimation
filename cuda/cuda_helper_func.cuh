#ifndef CUDA_HELPER_FUNC_H_
#define CUDA_HELPER_FUNC_H_

#include <glog/logging.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>

#include <stdio.h> 
#include <stdlib.h> 
#include <assert.h> 
#include <cuda_runtime.h> 
#include <cublas_v2.h> 
#include <cusparse.h>

class CuDenseMatrix
{
public:
	CuDenseMatrix()
	{
		rows = 0;
		cols = 0;
		d_Val = NULL;
	}

	CuDenseMatrix(int rows, int cols)
		:rows(rows), cols(cols)
	{
		cudaMalloc((void **)& d_Val, rows * cols * sizeof(double));
	}

	CuDenseMatrix(int rows, int cols, double *ptr)
		:rows(rows), cols(cols)
	{
		cudaMalloc((void **)& d_Val, rows * cols * sizeof(double));
		//cublasSetMatrix(rows, cols, sizeof(double), ptr, rows, d_Val, rows);
		cudaMemcpy(d_Val, ptr, sizeof(double) * rows * cols, cudaMemcpyHostToDevice);
	}

	~CuDenseMatrix()
	{
		cudaFree(d_Val);
	}

	void SetSize(int r, int c)
	{
		rows = r;
		cols = c;
		cudaMalloc((void **)& d_Val, rows * cols * sizeof(double));
	}

	void SetData(double *ptr)
	{
		cudaMemcpy(d_Val, ptr, sizeof(double) * rows * cols, cudaMemcpyHostToDevice);
	}

	void SetMatrix(int r, int c, double *ptr)
	{
		SetSize(r, c);
		SetData(ptr);
	}

	void GetMatrix(int r, int c, double *ptr, cudaStream_t &stream)
	{
		cudaStreamSynchronize(stream);
		LOG(INFO) << "GPU ---> CPU :" << r << " x " << c;
		cudaMemcpy(ptr, d_Val, sizeof(double) * r * c, cudaMemcpyDeviceToHost);
	}

	void SetZero(cublasHandle_t &handle)
	{
		double al = 0;
		double bet = 0;
		cublasDgeam(handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			rows, cols,
			&al,
			nullptr, 0,
			&bet,
			nullptr, 0,
			d_Val, rows);
	}

public:
	int rows;
	int cols;
	double *d_Val;
};

class CuSparseMatrix
{
public:
	CuSparseMatrix()
	{
		rows = 0;
		cols = 0;
		entries = 0;
		d_csrRowPtr = NULL;
		d_csrColInd = NULL;
		d_csrVal = NULL;

		cusparseStatus_t cudaStat;
		cudaStat = cusparseCreateMatDescr(&descr);
		assert(CUSPARSE_STATUS_SUCCESS == cudaStat);
		cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
	}

	CuSparseMatrix(Eigen::SparseMatrix<double, Eigen::RowMajor> &sm)
	{
		SetMatrix(sm);
	}

	void SetData(Eigen::SparseMatrix<double, Eigen::RowMajor> &sm)
	{
		if (!sm.isCompressed())
			sm.makeCompressed();

		if (entries != sm.nonZeros() ||
			rows != sm.rows() ||
			cols != sm.cols())
			LOG(FATAL) << "wrong size!";

		cudaMemcpy(d_csrRowPtr, sm.outerIndexPtr(), sizeof(int) * (rows + 1), cudaMemcpyHostToDevice);// cols or rows ?
		cudaMemcpy(d_csrColInd, sm.innerIndexPtr(), sizeof(int) * entries, cudaMemcpyHostToDevice);
		cudaMemcpy(d_csrVal, sm.valuePtr(), sizeof(double) * entries, cudaMemcpyHostToDevice);
	}

	void SetMatrix(Eigen::SparseMatrix<double, Eigen::RowMajor> &sm)
	{
		if (!sm.isCompressed())
			sm.makeCompressed();

		rows = sm.outerSize();
		cols = sm.innerSize();
		entries = sm.nonZeros();
		cudaMalloc((void**)&d_csrRowPtr, sizeof(int) * (rows + 1));// cols or rows ?
		cudaMalloc((void**)&d_csrColInd, sizeof(int) * entries);
		cudaMalloc((void**)&d_csrVal, sizeof(double) * entries);

		//for (int i = 0; i < rows + 1; i++) {
		//	std::cout << sm.outerIndexPtr()[i] << " ";
		//}
		//std::cout << "\n";
		//for (int i = 0; i < entries; i++) {
		//	std::cout << sm.innerIndexPtr()[i] << " ";
		//}
		//std::cout << "\n";
		//for (int i = 0; i < entries; i++) {
		//	std::cout << sm.valuePtr()[i] << " ";
		//}
		//std::cout << "\n";

		cudaMemcpy(d_csrRowPtr, sm.outerIndexPtr(), sizeof(int) * (rows + 1), cudaMemcpyHostToDevice);// cols or rows ?
		cudaMemcpy(d_csrColInd, sm.innerIndexPtr(), sizeof(int) * entries, cudaMemcpyHostToDevice);
		cudaMemcpy(d_csrVal, sm.valuePtr(), sizeof(double) * entries, cudaMemcpyHostToDevice);
	}

	~CuSparseMatrix()
	{
		cudaFree(d_csrRowPtr);
		cudaFree(d_csrColInd);
		cudaFree(d_csrVal);
	}

public:
	int rows;
	int cols;
	int entries;

	int *d_csrRowPtr;
	int *d_csrColInd;
	double *d_csrVal;

	cusparseMatDescr_t descr;
};

void DMmulDM(cublasHandle_t& handle,
	CuDenseMatrix &dm1,
	CuDenseMatrix &dm2,
	CuDenseMatrix &result,
	double al = 1.0, double bet = 0.0);

void DMaddDM(cublasHandle_t& handle,
	CuDenseMatrix &dm1,
	CuDenseMatrix &dm2,
	CuDenseMatrix &result,
	double al = 1.0, double bet = 1.0);

void DMmulDV(cublasHandle_t& handle,
	CuDenseMatrix &dm1,
	CuDenseMatrix &dv2,
	CuDenseMatrix &result,
	double al = 1.0, double bet = 0.0);

void DVaddDV(cublasHandle_t& handle, 
	CuDenseMatrix &dv1, 
	CuDenseMatrix &result, 
	double al = 1.0);

void SMmulDM(cusparseHandle_t &handle, 
	CuSparseMatrix &dm1, 
	CuDenseMatrix &dm2, 
	CuDenseMatrix &result, 
	double al = 1.0, double bet = 0.0);

void SMmulDV(cusparseHandle_t &handle, 
	CuSparseMatrix &sm1, 
	CuDenseMatrix &dm2, 
	CuDenseMatrix &result, 
	double al, double bet);

void SM2DM(cusparseHandle_t &handle, 
	CuSparseMatrix &sm1, 
	CuDenseMatrix &result);

void DM2DM(cublasHandle_t &handle,
	CuDenseMatrix &dm1,
	CuDenseMatrix &dm2);

void SM2SM(cublasHandle_t &handle,
	CuSparseMatrix &sm1,
	CuSparseMatrix &sm2);

#endif