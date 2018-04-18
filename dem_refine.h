#ifndef DEM_REFINE_H_
#define DEM_REFINE_H_

#include "parameters.h"
#include "cuda/cuda_helper_func.cuh"

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <mutex>
#include <chrono>
using namespace Eigen;

class DemRefine
{
public:
	DemRefine();

	void SetPtr(cublasHandle_t handle,
		cusparseHandle_t shandle,
		cudaStream_t stream_refine_,
		CuDenseMatrix *M_cu_,
		CuDenseMatrix *P_cu_,
		CuDenseMatrix *delta_B1_cu_,
		CuDenseMatrix *delta_B2_cu_);

	void operator()();

	// call outside
	void GetY(CuDenseMatrix &dm, cudaStream_t &stream);
	// call inside
	void UpdateY(MatrixXd &result);
	// call inside
	void GetX(cudaStream_t &stream);
	// call outside
	void UpdateX(MatrixXd &x, CuSparseMatrix A_in, CuDenseMatrix C_in, cudaStream_t &stream);

public:
	double S_refine_;
	double S_total_refine_;
	double *p_X_eg_refine_;
	double *p_Y_eg_refine_;
	Map<MatrixXd> X_eg_refine_;
	Map<MatrixXd> Y_eg_refine_;
	CuDenseMatrix X_cu_refine_;
	CuDenseMatrix Y_cu_refine_;
	// in
	bool updated;
	std::mutex x_mtx_refine_;
	MatrixXd x_in_refine_;
	CuSparseMatrix A_in_refine_;
	CuDenseMatrix C_in_refine_;
	// out
	std::mutex y_mtx_refine_;
	MatrixXd y_coeff_refine_;
	// run time variable
	MatrixXd x_coeff_refine_;
	CuSparseMatrix A_refine_;
	CuDenseMatrix C_refine_;
	CuDenseMatrix A_hat_refine_;
	CuDenseMatrix C_hat_refine_;
	// tmo references
	cublasHandle_t handle;
	cusparseHandle_t shandle;
	cudaStream_t stream_refine_;
	CuDenseMatrix *M_cu_;
	CuDenseMatrix *P_cu_;
	CuDenseMatrix *delta_B1_cu_;
	CuDenseMatrix *delta_B2_cu_;
};

#endif