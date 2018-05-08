#ifndef DEM_REFINE_H_
#define DEM_REFINE_H_

#include "parameters.h"
#include "cuda/cuda_helper_func.cuh"

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <mutex>
#include <chrono>
using namespace Eigen;

class DemRefineMiddleWare
{
public:
	DemRefineMiddleWare()
		:updated(true)
	{
		// in
		x_in_refine_.resize(exp_size, 1);
		C_in_refine_.SetSize(6 * face_landmark_size, 1);
	}

	// call outside
	void GetY(cudaStream_t &stream, CuDenseMatrix &dm);
	// call inside
	void UpdateY(MatrixXd &result);
	// call inside
	void GetX(cudaStream_t &stream,
		CuSparseMatrix &A_refine_, CuDenseMatrix &C_refine_, MatrixXd &x_coeff_refine_, SparseMatrix<double> &A_refine_eg_);
	// call outside
	void UpdateX(cudaStream_t &stream,
		MatrixXd &x, CuSparseMatrix &A_in, CuDenseMatrix &C_in, SparseMatrix<double> &A_in_refine_eg_);

	// out
	std::mutex y_mtx_refine_;
	MatrixXd y_out_refine_;
	// in
	bool updated;
	std::mutex x_mtx_refine_;
	MatrixXd x_in_refine_;
	MatrixXd y_in_refine_;
	CuSparseMatrix A_in_refine_;
	SparseMatrix<double> A_in_refine_eg_;
	CuDenseMatrix C_in_refine_;
};

class DemRefine
{
public:
	DemRefine();

	void SetPtr(cublasHandle_t handle1
		, cusparseHandle_t shandle1
		, cudaStream_t streams_cu_1
		, CuDenseMatrix &M_cu_1
		, CuDenseMatrix &P_cu_1
		, CuDenseMatrix &delta_B1_cu_1
		, CuDenseMatrix &delta_B2_cu_1
		, DemRefineMiddleWare *middleware_1
		, VectorXd &y_weights_eg_1);

	void operator()();

public:
	bool isFirst;

	double S_refine_;
	double S_total_refine_;
	double *p_X_eg_refine_;
	double *p_Y_eg_refine_;
	Map<MatrixXd> X_eg_refine_;
	Map<MatrixXd> Y_eg_refine_;

	// in
	DemRefineMiddleWare *middleware_;
	//bool updated;
	//std::mutex x_mtx_refine_;
	//MatrixXd x_in_refine_;
	//CuSparseMatrix A_in_refine_;
	//CuDenseMatrix C_in_refine_;
	// out
	//std::mutex y_mtx_refine_;
	//MatrixXd y_out_refine_;

	// run time variable
	MatrixXd x_coeff_refine_;
	CuSparseMatrix A_refine_;
	SparseMatrix<double> A_refine_eg_;
	CuDenseMatrix C_refine_;

	MatrixXd y_coeff_refine_;
	// tmo references
	cublasHandle_t handle;
	cusparseHandle_t shandle;
	cudaStream_t stream_refine_;
	CuDenseMatrix M_cu_;
	CuDenseMatrix P_cu_;
	CuDenseMatrix delta_B1_cu_;
	CuDenseMatrix delta_B2_cu_;
	MatrixXd y_weights_refine_eg_;
};

#endif