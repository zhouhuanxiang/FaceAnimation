#include "dem_refine.h"

DemRefine::DemRefine()
	:S_refine_(0), S_total_refine_(0),
	X_eg_refine_(NULL, 0, 0), Y_eg_refine_(NULL, 0, 0),
	isFirst(true)
{
	cudaMallocHost(&p_X_eg_refine_, 6 * pca_size * pca_size * sizeof(double));
	cudaMallocHost(&p_Y_eg_refine_, 6 * pca_size * sizeof(double));
	new (&X_eg_refine_) Map<MatrixXd>(p_X_eg_refine_, pca_size, pca_size);
	new (&Y_eg_refine_) Map<MatrixXd>(p_Y_eg_refine_, pca_size, 1);
	X_eg_refine_.setZero();
	Y_eg_refine_.setZero();

	// out
	y_coeff_refine_.resize(pca_size, 1);
	// run time variance
	x_coeff_refine_.resize(exp_size, 1);
	std::cout << 6 * face_landmark_size;
	// global variance

	cudaDeviceSynchronize();
}

void DemRefine::operator()()
{
	while (true) {
		while (middleware_->updated) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			continue;
		}
		if (isFirst) {
			y_coeff_refine_ = middleware_->y_in_refine_;
			isFirst = false;
		}
		//
		LOG(WARNING) << "X";
		//
		cublasSetStream(handle, stream_refine_);
		middleware_->GetX(stream_refine_, A_refine_, C_refine_, x_coeff_refine_, A_refine_eg_);

		MatrixXd tp(3 * vertex_size * pca_size, exp_size);
		delta_B2_cu_.GetMatrix(3 * vertex_size * pca_size, exp_size, tp.data(), stream_refine_);
		MatrixXd tm(3 * vertex_size, exp_size);
		delta_B1_cu_.GetMatrix(3 * vertex_size, exp_size, tm.data(), stream_refine_);
		MatrixXd p(3 * vertex_size * pca_size, 1);
		P_cu_.GetMatrix(3 * vertex_size, pca_size, p.data(), stream_refine_);
		MatrixXd m(3 * vertex_size, 1);
		M_cu_.GetMatrix(3 * vertex_size, 1, m.data(), stream_refine_);
		MatrixXd rhs(6 * face_landmark_size, 1);
		C_refine_.GetMatrix(6 * face_landmark_size, 1, rhs.data(), stream_refine_);
		MatrixXd r1 = tp * x_coeff_refine_ + p;
		MatrixXd r2 = tm * x_coeff_refine_ + m;
		MatrixXd r3 = A_refine_eg_ * Map<MatrixXd>(r1.data(), 3 * vertex_size, pca_size);
		MatrixXd r4 = rhs - A_refine_eg_ * r2;

		// update X_total Y_total
		double gamma = 0.9;
		double S_total_refine_ = gamma * S_refine_ + 1;
		double al = 1.0 / S_total_refine_;
		double bet = gamma * S_refine_ / S_total_refine_;
		MatrixXd r3t = r3.transpose();
		X_eg_refine_ = gamma * S_refine_ / S_total_refine_ * X_eg_refine_ + 1.0 / S_total_refine_ * r3t * r3;
		Y_eg_refine_ = gamma * S_refine_ / S_total_refine_ * Y_eg_refine_ + 1.0 / S_total_refine_ * r3t * r4;
		S_refine_ = S_total_refine_;

		// solve it
		MatrixXd result1, result2;
		result1 = result2 = y_coeff_refine_;
		//std::cout << Map<RowVectorXd>(y_coeff_refine_.data(), pca_size) << "\n\n";

		MatrixXd tmp = X_eg_refine_ + y_weights_refine_eg_ * 10000;
		auto D = tmp.diagonal().asDiagonal().toDenseMatrix();
		auto L = tmp.triangularView<Eigen::StrictlyLower>().toDenseMatrix();
		auto U = tmp.triangularView<Eigen::StrictlyUpper>().toDenseMatrix();

		double cost = (tmp * result2 - Y_eg_refine_).norm();

		FullPivLU<MatrixXd> llt;
		llt.compute(D + U);
		for (int i = 0; i < 10; i++) {
			result2 = llt.solve(Y_eg_refine_ - L * result1);
			double new_cost = (tmp * result2 - Y_eg_refine_).norm();
			std::cout << new_cost << "@@@\n";
			if ((cost - new_cost) > 0.00001 * cost) {
				result1 = result2;
				cost = new_cost;
			}
			else /*if (new_cost <= cost)*/ {
				break;
			}
		}

		LOG(WARNING) << "Y: " << Map<RowVectorXd>(result1.data(), pca_size);
		y_coeff_refine_ = result1;
		middleware_->UpdateY(y_coeff_refine_);
	}
}

void DemRefine::SetPtr(cublasHandle_t handle1
	,cusparseHandle_t shandle1
	,cudaStream_t streams_cu_1
	,CuDenseMatrix &M_cu_1
	,CuDenseMatrix &P_cu_1
	,CuDenseMatrix &delta_B1_cu_1
	,CuDenseMatrix &delta_B2_cu_1
	,DemRefineMiddleWare *middleware_1
	,VectorXd &y_weights_eg_1)
{
	handle = handle1;
	shandle = shandle1;
	stream_refine_ = streams_cu_1;
	M_cu_ = M_cu_1;
	P_cu_ = P_cu_1;
	delta_B1_cu_ = delta_B1_cu_1;
	delta_B2_cu_ = delta_B2_cu_1;
	middleware_ = middleware_1;
	
	y_weights_refine_eg_ = y_weights_eg_1.asDiagonal().toDenseMatrix();
	//std::cout << y_weights_refine_eg_;
	//std::cout << "\n" << y_weights_refine_eg_.rows() << " " << y_weights_refine_eg_.cols() << "\n";
}



// call outside
void DemRefineMiddleWare::GetY(cudaStream_t &stream, CuDenseMatrix &dm)
{
	y_mtx_refine_.lock();
	dm.SetData(y_out_refine_.data());
	cudaStreamSynchronize(stream);
	y_mtx_refine_.unlock();
}

// call inside
void DemRefineMiddleWare::UpdateY(MatrixXd &result)
{
	y_mtx_refine_.lock();
	y_out_refine_ = result;
	y_mtx_refine_.unlock();
}

// call inside
void DemRefineMiddleWare::GetX(cudaStream_t &stream,
	CuSparseMatrix &A_refine_, CuDenseMatrix &C_refine_, MatrixXd &x_coeff_refine_, SparseMatrix<double> &A_refine_eg_)
{
	x_mtx_refine_.lock();
	SM2SM(A_in_refine_, A_refine_);
	DM2DM(C_in_refine_, C_refine_);

	updated = true;
	x_coeff_refine_ = x_in_refine_;
	A_refine_eg_ = A_in_refine_eg_;
	cudaStreamSynchronize(stream);
	x_mtx_refine_.unlock();
}

// call outside
void DemRefineMiddleWare::UpdateX(cudaStream_t &stream,
	MatrixXd &x, CuSparseMatrix &A_in, CuDenseMatrix &C_in, SparseMatrix<double> &A_in_eg_)
{
	x_mtx_refine_.lock();
	SM2SM(A_in, A_in_refine_);
	DM2DM(C_in, C_in_refine_);
	updated = false;
	x_in_refine_ = x;
	A_in_refine_eg_ = A_in_eg_;
	cudaStreamSynchronize(stream);
	x_mtx_refine_.unlock();
}
