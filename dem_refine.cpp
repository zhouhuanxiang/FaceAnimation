#include "dem_refine.h"

DemRefine::DemRefine()
	:S_refine_(0), S_total_refine_(0),
	X_eg_refine_(NULL, 0, 0), Y_eg_refine_(NULL, 0, 0),
	updated(false)
{
	cudaMallocHost(&p_X_eg_refine_, 6 * face_landmark.size() * pca_size * sizeof(double));
	cudaMallocHost(&p_Y_eg_refine_, 6 * face_landmark.size() * sizeof(double));
	new (&X_eg_refine_) Map<MatrixXd>(p_X_eg_refine_, 6 * face_landmark.size(), pca_size);
	new (&Y_eg_refine_) Map<MatrixXd>(p_Y_eg_refine_, 6 * face_landmark.size(), 1);
	X_eg_refine_.setZero();
	Y_eg_refine_.setZero();
	X_cu_refine_.SetMatrix(6 * face_landmark.size(), pca_size, X_eg_refine_.data());
	Y_cu_refine_.SetMatrix(6 * face_landmark.size(), 1, Y_eg_refine_.data());
	// in
	x_in_refine_.resize(exp_size, 1);
	C_in_refine_.SetSize(6 * face_landmark.size(), 1);
	// out
	y_coeff_refine_.resize(pca_size, 1);
	// run time variance
	x_coeff_refine_.resize(exp_size, 1);
	C_refine_.SetSize(6 * face_landmark.size(), 1);
	A_hat_refine_.SetSize(6 * face_landmark.size(), pca_size);
	C_hat_refine_.SetSize(6 * face_landmark.size(), 1);
	// global variance

}

void DemRefine::operator()()
{
	while (true) {
		while (!updated) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			continue;
		}
		//
		cublasSetStream(handle, stream_refine_);
		GetX(stream_refine_);
		// update A_hat C_hat
		CuDenseMatrix tA(3 * vertex_size, pca_size);
		CuDenseMatrix tC(3 * vertex_size, 1);
		DM2DM(handle, *P_cu_, tA);
		DM2DM(handle, *M_cu_, tC);
		for (int i = 0; i < exp_size; i++) {
			double a = x_coeff_refine_(i, 0);
			double b = 1;
			cublasDgeam(handle,
				CUBLAS_OP_N, CUBLAS_OP_N,
				3 * vertex_size, pca_size,
				&a,
				delta_B2_cu_->d_Val + i * 3 * vertex_size * pca_size, 3 * vertex_size,
				&b,
				tA.d_Val, tA.rows,
				tA.d_Val, tA.rows);
			cublasDaxpy(handle,
				3 * vertex_size,
				&a,
				delta_B1_cu_->d_Val + i * 3 * vertex_size, 1,
				tC.d_Val, 1);
		}
		SMmulDM(shandle, A_refine_, tA, A_hat_refine_);
		DM2DM(handle, C_refine_, C_hat_refine_);
		SMmulDV(shandle, A_refine_, tC, C_hat_refine_, -1, 1);
		// update X_total Y_total
		double gamma = 0.9;
		double S_total_refine_ = gamma * S_refine_ + 1;
		double al = 1.0 / S_total_refine_;
		double bet = gamma * S_refine_ / S_total_refine_;
		cublasDgemm(handle,
			CUBLAS_OP_T, CUBLAS_OP_N,
			pca_size, pca_size, 6 * face_landmark.size(),
			&al,
			A_hat_refine_.d_Val, A_hat_refine_.rows,
			A_hat_refine_.d_Val, A_hat_refine_.rows,
			&bet,
			X_cu_refine_.d_Val, X_cu_refine_.rows);
		cublasDgemv(handle,
			CUBLAS_OP_T,
			6 * face_landmark.size(), pca_size,
			&al,
			A_hat_refine_.d_Val, A_hat_refine_.rows,
			C_hat_refine_.d_Val, 1,
			&bet,
			Y_cu_refine_.d_Val, 1);
		S_refine_ = S_total_refine_;
		// read [X Y] --> CPU
		X_cu_refine_.GetMatrix(pca_size, pca_size, X_eg_refine_.data(), stream_refine_);
		Y_cu_refine_.GetMatrix(pca_size, 1, Y_eg_refine_.data(), stream_refine_);

		// solve it
		MatrixXd result1, result2;
		result1 = result2 = y_coeff_refine_;

		MatrixXd tmp = X_eg_refine_;
		MatrixXd D = tmp.diagonal().asDiagonal().toDenseMatrix();
		MatrixXd L = tmp.triangularView<Eigen::StrictlyLower>().toDenseMatrix();
		MatrixXd U = tmp.triangularView<Eigen::StrictlyUpper>().toDenseMatrix();

		LLT<MatrixXd> llt;
		llt.compute(D + U);
		double cost = -1;
		for (int i = 0; i < 10; i++) {
			result1 = llt.solve(Y_eg_refine_ - L * result2);
			double new_cost = (tmp * result1 - Y_eg_refine_).norm();
			if ((cost - new_cost) > 0.00001 * cost || cost < -1) {
				result2 = result1;
				cost = new_cost;
			}
			else
				break;
		}
		std::cout << Map<RowVectorXd>(result2.data(), pca_size);
		UpdateY(result2);
	}
}

void DemRefine::SetPtr(cublasHandle_t handle1,
	cusparseHandle_t shandle1,
	cudaStream_t streams_cu_1,
	CuDenseMatrix *M_cu_1,
	CuDenseMatrix *P_cu_1,
	CuDenseMatrix *delta_B1_cu_1,
	CuDenseMatrix *delta_B2_cu_1)
{
	handle = handle1;
	shandle = shandle1;
	stream_refine_ = streams_cu_1;
	M_cu_ = M_cu_1;
	P_cu_ = P_cu_1;
	delta_B1_cu_ = delta_B1_cu_1;
	delta_B2_cu_ = delta_B2_cu_1;
}



// call outside
void DemRefine::GetY(CuDenseMatrix &dm, cudaStream_t &stream)
{
	y_mtx_refine_.lock();
	dm.SetData(y_coeff_refine_.data());
	cudaStreamSynchronize(stream);
	y_mtx_refine_.unlock();
}

// call inside
void DemRefine::UpdateY(MatrixXd &result)
{
	y_mtx_refine_.lock();
	y_coeff_refine_ = result;
	y_mtx_refine_.unlock();
}

// call inside
void DemRefine::GetX(cudaStream_t &stream)
{
	x_mtx_refine_.lock();
	SM2SM(handle, A_in_refine_, A_refine_);
	DM2DM(handle, C_in_refine_, C_refine_);
	updated = false;
	x_coeff_refine_ = x_in_refine_;
	cudaStreamSynchronize(stream);
	x_mtx_refine_.unlock();
}

// call outside
void DemRefine::UpdateX(MatrixXd &x, CuSparseMatrix A_in, CuDenseMatrix C_in, cudaStream_t &stream)
{
	x_mtx_refine_.lock();
	SM2SM(handle, A_in, A_in_refine_);
	DM2DM(handle, C_in, C_in_refine_);
	updated = true;
	x_in_refine_ = x;
	cudaStreamSynchronize(stream);
	x_mtx_refine_.unlock();
}
