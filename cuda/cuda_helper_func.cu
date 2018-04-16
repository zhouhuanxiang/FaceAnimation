#include "cuda_helper_func.cuh"

void DMmulDM(cublasHandle_t& handle, CuDenseMatrix &dm1, CuDenseMatrix &dm2, CuDenseMatrix &result, double al, double bet)
{
	if (dm1.rows != result.rows ||
		dm2.cols != result.cols ||
		dm1.cols != dm2.rows)
		LOG(FATAL) << "wrong size!";
	cublasDgemm(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		dm1.rows, dm2.cols, dm2.rows,
		&al, dm1.d_Val, dm1.rows, dm2.d_Val, dm2.rows,
		&bet, result.d_Val, result.rows);
}

void DMaddDM(cublasHandle_t& handle, CuDenseMatrix &dm1, CuDenseMatrix &dm2, CuDenseMatrix &result, double al, double bet)
{
	if(result.rows != dm1.rows ||
		result.rows != dm2.rows ||
		result.cols != dm1.cols ||
		result.cols != dm2.cols)
		LOG(FATAL) << "wrong size!";
	cublasDgeam(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		dm1.rows, dm1.cols,
		&al, dm1.d_Val, dm1.rows,
		&bet, dm2.d_Val, dm2.rows,
		result.d_Val, result.rows);
}

void DMmulDV(cublasHandle_t& handle, CuDenseMatrix &dm1, CuDenseMatrix &dv2, CuDenseMatrix &result, double al, double bet)
{
	cublasDgemv(handle,
		CUBLAS_OP_N,
		dm1.rows, dm1.cols,
		&al, dm1.d_Val, dm1.rows,
		dv2.d_Val, 1,
		&bet, result.d_Val, 1);
}

void DVaddDV(cublasHandle_t& handle, CuDenseMatrix &dv1, CuDenseMatrix &result, double al)
{
	cublasDaxpy(handle,
		dv1.rows,
		&al,
		dv1.d_Val, 1,
		result.d_Val, 1);
}

void SMmulDM(cusparseHandle_t &handle, CuSparseMatrix &sm1, CuDenseMatrix &dm2, CuDenseMatrix &result, double al, double bet)
{
	cusparseStatus_t cusparseStat = cusparseDcsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		sm1.rows, dm2.cols, dm2.rows,
		sm1.entries, 
		&al, 
		sm1.descr, sm1.d_csrVal, sm1.d_csrRowPtr, sm1.d_csrColInd,
		dm2.d_Val, dm2.rows,
		&bet, 
		result.d_Val, result.rows);
	assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);
}

void SMmulDV(cusparseHandle_t &handle, CuSparseMatrix &sm1, CuDenseMatrix &dm2, CuDenseMatrix &result, double al, double bet)
{
	cusparseStatus_t cusparseStat = cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		sm1.rows, sm1.cols, 
		sm1.entries,
		&al,
		sm1.descr, sm1.d_csrVal, sm1.d_csrRowPtr, sm1.d_csrColInd,
		dm2.d_Val,
		&bet,
		result.d_Val);
	assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);
}


void SM2DM(cusparseHandle_t &handle, CuSparseMatrix &sm1, CuDenseMatrix &result)
{
	cusparseStatus_t cusparseStat = cusparseDcsr2dense(handle, sm1.rows, sm1.cols,
		sm1.descr, sm1.d_csrVal, sm1.d_csrRowPtr, sm1.d_csrColInd,
		result.d_Val, result.rows);
	assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);
}

void DM2DM(cublasHandle_t &handle, CuDenseMatrix &dm1, CuDenseMatrix &dm2)
{
	cudaMemcpy(dm2.d_Val, dm1.d_Val, sizeof(double) * dm1.rows * dm1.cols, cudaMemcpyDeviceToDevice);
}