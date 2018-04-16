#include "ceres_initialization.h"

CeresInitializationError::CeresInitializationError(int index, cv::Mat& frame, Matrix3Xd& frame_normals, Matrix3Xd& model_normals, MatrixXd& pca_models,
	double fx, double fy, double cx, double cy, bool print)
	:index(index), frame(frame), frame_normals(frame_normals), model_normals(model_normals), pca_models(pca_models),
	fx(fx), fy(fy), cx(cx), cy(cy), print(print)
{
}

template <class T>
bool CeresInitializationError::operator()(const T* const R, const T* const tr, const T* const pca_coeff, T* residuals) const 
{
	residuals[0] = (T)0;
	//residuals[0] = residuals[1] = residuals[2] = residuals[3] = (T)0;

	T p1[3];
	for (int i = 0; i < 3; ++i){
		p1[i] = (T)pca_models(3 * index + i, 0);
		for (int j = 1; j < pca_size; ++j)
			p1[i] += (T)pca_models(3 * index + i, j) * pca_coeff[j];
	}
	T p2[3];
	ceres::AngleAxisRotatePoint(R, p1, p2);
	for (int i = 0; i < 3; ++i) {
		p1[i] = p2[i] + tr[i];
	}

	T x = -1.0 * p1[0] / p1[2] * fx + cx;
	T y = -1.0 * p1[1] / p1[2] * fy + cy;
	int px = (int)*(double*)&x;
	int py = (int)*(double*)&y;
	T wx = x - (T)px;
	T wy = y - (T)py;
	int rx = px + 1, ry = py + 1;
	if (!(px > 0 && py > 0
		&& rx < frame.cols
		&& ry < frame.rows)){
		return true;
	}
	int xs[4], ys[4];
	xs[0] = px; ys[0] = py;
	xs[1] = rx; ys[1] = py;
	xs[2] = px; ys[2] = ry;
	xs[3] = rx; ys[3] = ry;
	T ws[4];
	ws[0] = ((T)1. - wx) * ((T)1. - wy);
	ws[1] = wx * ((T)1. - wy);
	ws[2] = ((T)1. - wx) * wy;
	ws[3] = wx * wy;
	// depth
	T dv = T(0);
	double pz = *(double*)&(p1[2]);
	double max_depth_residual = 5;
	double beta = 3;
	for (int i = 0; i < 4; i++){
		double di = frame.at<unsigned short>(ys[i], xs[i]) /** 1e-3f*/;
		double ratio = (di - pz) / max_depth_residual;
		if (ratio > 2){
			di = pz + (1 + beta) * max_depth_residual;
		}
		else if (ratio < -2){
			di = pz - (1 + beta) * max_depth_residual;
		}
		else if (ratio > 1){
			di = pz + (beta * ratio - beta + 1) * max_depth_residual;
		}
		else if (ratio < -1){
			di = pz + (beta * ratio + beta - 1) * max_depth_residual;
		}
		dv += T(di) * ws[i];
	}
	// normal vector
	T nv[3];
	Vector3d n1 = model_normals.col(index);
	Vector3d n2 = frame_normals.col(py * frame.cols + px);
	for (int i = 0; i < 3; i++){
		nv[i] = (T)n2[i];
	}
	//
	//std::cout << *(double*)&(dv) << " " << *(double*)&(p1[2]) << " " << n1.dot(n2) << "\n";
	if (std::abs(*(double*)&(dv)-*(double*)&(p1[2])) < max_depth_residual
		&& n1.dot(n2) > 0.7){
		residuals[0] = nv[2] * (dv - p1[2])
			+ nv[0] * (dv - p1[2]) * p1[0] / p1[2]
			+ nv[1] * (dv - p1[2]) * p1[1] / p1[2];
		//std::cout << residuals[0] << "@residual[0]\n";
	}
	if (std::abs(*(double*)&(dv)-*(double*)&(p1[2])) < max_depth_residual * 2) {
		T alpha(0);
		residuals[1] = alpha * (dv - p1[2]);
		residuals[2] = alpha * (dv - p1[2]) * p1[0] / p1[2];
		residuals[3] = alpha * (dv - p1[2]) * p1[1] / p1[2];
	}
	//
	if (print){
		std::cout << *(double*)&(nv[2]) << " "
			<< *(double*)&(dv) << " "
			<< *(double*)&(p1[2]) << " "
			<< *(double*)&(residuals[0]) << "\n";
	}

	//residual[0] = (T)0;
	//std::cout << *(double*)&(residuals[0]) << "\n";
	//residuals[1] = residuals[2] = residuals[3] = (T)0;
	//std::cout << *(double*)&(residuals[1]) << " " << *(double*)&(residuals[2]) << " " << *(double*)&(residuals[3]) << "\n";
	return true;
}

ceres::CostFunction* CeresInitializationError::Create(int index, cv::Mat& frame, Matrix3Xd& frame_normals, Matrix3Xd& model_normals, MatrixXd& pca_models,
	double fx, double fy, double cx, double cy, bool print) {
	// first residual dimension, followed with parameters' dimensions
	return (new ceres::AutoDiffCostFunction<CeresInitializationError, 4, 3, 3, pca_size>(
		new CeresInitializationError(index, frame, frame_normals, model_normals, pca_models, fx, fy, cx, cy, print)));
}

CeresLandmarkError::CeresLandmarkError(int index, int index1, 
	cv::Mat& frame, 
	MatrixXd& M_eg_, MatrixXd& P_eg_,
	double fx, double fy, double cx, double cy, 
	Vector3d p3_landmark)
	:index(index), index1(index1), 
	frame(frame), 
	M_eg(M_eg_), P_eg(P_eg_),
	fx(fx), fy(fy), cx(cx), cy(cy), 
	p3_landmark(p3_landmark)
{
}

template <class T>
bool CeresLandmarkError::operator()(const T* const R, const T* const tr, const T* const pca_coeff, T* residuals) const
{
	for (int i = 0; i < 3; i++)
		residuals[i] = T(0);
	if (index1 < 17)
		return true;

	T p1[3];
	for (int i = 0; i < 3; ++i) {
		p1[i] = (T)M_eg(3 * index + i, 0);
		for (int m = 0; m < pca_size; ++m)
			p1[i] += (T)P_eg(3 * index + i, m) * pca_coeff[m];
	}
	T p2[3];
	ceres::AngleAxisRotatePoint(R, p1, p2);
	for (int i = 0; i < 3; ++i) {
		p1[i] = p2[i] + tr[i];
	}
	// 3d landmark displacement
	for (int i = 0; i < 3; i++) {
		p1[i] = p1[i] - p3_landmark(i);
	}
	//std::cout << *(double*)&(p1[0]) << " " << *(double*)&(p1[1]) << " " << *(double*)&(p1[2]) << "\n";

	//double distance = 0;
	//for (int i = 0; i < 3; i++) {
	//	distance += *(double*)&(p1[i]) * *(double*)&(p1[i]);
	//}
	//distance = std::sqrt(distance);
	//std::cout << distance << "@distance\n";

	if (true) {
		// point-to-plane
		//double alpha1 = 0.2;
		//for (int i = 0; i < 3; i++) {
		//	residuals[0] += alpha1 * model_normals(i, index) * p1[i];
		//}
		// point-to-point
		double alpha2 = 1;
		for (int r = 0; r < 3; r++) {
			residuals[r] = alpha2 * p1[r];
		}
	}
	return true;
}

ceres::CostFunction* CeresLandmarkError::Create(int index, int index1, 
	cv::Mat& frame, 
	MatrixXd& M_eg_, MatrixXd& P_eg_,
	double fx, double fy, double cx, double cy, 
	Vector3d p3_landmark) {
	// first residual dimension, followed with parameters' dimensions
	return (new ceres::AutoDiffCostFunction<CeresLandmarkError, 3, 3, 3, pca_size>(
		new CeresLandmarkError(index, index1, 
			frame, 
			M_eg_, P_eg_,
			fx, fy, cx, cy, 
			p3_landmark)));
}

CeresInitializationRegulation::CeresInitializationRegulation(VectorXd& pca_weights)
	:pca_weights(pca_weights)
{
}

template <class T>
bool CeresInitializationRegulation::operator()(const T* const pca_coeff, T* residuals) const
{
	for (int i = 0; i < pca_size; i++){
		residuals[i] = ((T)pca_coeff[i]) * pca_weights(i) * 500.0;
		//std::cout << residuals[i] << "\n";
	}
	//std::cout << "\n";
	return true;
}

ceres::CostFunction* CeresInitializationRegulation::Create(VectorXd& pca_weights) 
{
	// first residual dimension, followed with parameters' dimensions
	return (new ceres::AutoDiffCostFunction<CeresInitializationRegulation, pca_size, pca_size>(
		new CeresInitializationRegulation(pca_weights)));
}

