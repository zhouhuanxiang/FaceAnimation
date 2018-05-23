#include "ceres_initialization.h"

#include <iostream>

CeresFaceDenseError::CeresFaceDenseError(int mesh_index,
	cv::Mat& frame,
	MatrixXd& M_eg_, MatrixXd& P_eg_,
	MatrixXd& normal_eg_,
	double weight,
	bool point_to_point)
	:mesh_index(mesh_index),
	frame(frame),
	M_eg(M_eg_), P_eg(P_eg_),
	normal_eg(normal_eg_),
	weight(weight),
	point_to_point(point_to_point)
{}

template <class T>
bool CeresFaceDenseError::operator()(const T* const R, const T* const tr, const T* const pca_coeff, T* residuals) const
{
	residuals[0] = T(0);
	residuals[1] = T(0);

	T p1[3];
	for (int i = 0; i < 3; ++i) {
		p1[i] = (T)M_eg(3 * mesh_index + i, 0);
		for (int m = 0; m < pca_size; ++m)
			p1[i] += (T)P_eg(3 * mesh_index + i, m) * pca_coeff[m];
	}
	T p2[3];
	//
	ceres::AngleAxisRotatePoint(R, p1, p2);
	for (int i = 0; i < 3; ++i) {
		p1[i] = p2[i] + tr[i];
	}
	//
	T p3[3];
	p3[0] = -1.0 * p1[0] / p1[2] * depth_camera.fx + depth_camera.cx;
	p3[1] = -1.0 * p1[1] / p1[2] * depth_camera.fy + depth_camera.cy;
	p3[2] = p1[2];

	int px = (int)*(double*)&p3[0];
	int py = (int)*(double*)&p3[1];
	T wx = p3[0] - (T)px;
	T wy = p3[1] - (T)py;
	int rx = px + 1, ry = py + 1;
	if (!(px > 0 && py > 0
		&& rx < frame.cols
		&& ry < frame.rows)) {
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
	T d = T(0);
	for (int i = 0; i < 4; i++) {
		d += (double)frame.at<unsigned short>(ys[i], xs[i]) * ws[i];
	}

	//if (point_to_point) {
		residuals[0] = d - p1[2];
	//}
	//else {
		T p4[3];
		p4[0] = d / p1[2] * p1[0];
		p4[1] = d / p1[2] * p1[1];
		p4[2] = d;
		Vector3d n = normal_eg.col(mesh_index);
		residuals[1] = n(0) * (p4[0] - p1[0]) + n(1) * (p4[1] - p1[1]) + n(2) * (p4[2] - p1[2]);
	//}

		residuals[0] = weight * residuals[0];
		residuals[1] = 3 * weight * residuals[1];

	return true;
}

ceres::CostFunction* CeresFaceDenseError::Create(int mesh_index,
	cv::Mat& frame,
	MatrixXd& M_eg_, MatrixXd& P_eg_,
	MatrixXd& normal_eg_,
	double weight,
	bool point_to_point)
{
	// first residual dimension, followed with parameters' dimensions
	return (new ceres::AutoDiffCostFunction<CeresFaceDenseError, 2, 3, 3, pca_size>(
		new CeresFaceDenseError(mesh_index,
			frame,
			M_eg_, P_eg_,
			normal_eg_,
			weight,
			point_to_point)));
}

Matrix<double, 3, 1> CeresLandmarkError::camera_extrinsic_translation = Matrix<double, 3, 1>();

CeresLandmarkError::CeresLandmarkError(int mesh_index,
	cv::Mat& frame, 
	MatrixXd& M_eg_, MatrixXd& P_eg_,
	Vector2d p2_landmark)
	:mesh_index(mesh_index),
	frame(frame), 
	M_eg(M_eg_), P_eg(P_eg_),
	p2_landmark(p2_landmark)
{
}

template <class T>
bool CeresLandmarkError::operator()(const T* const R, const T* const tr, const T* const pca_coeff, T* residuals) const
{
	for (int i = 0; i < 3; i++)
		residuals[i] = T(0);

	T p1[3];
	for (int i = 0; i < 3; ++i) {
		p1[i] = (T)M_eg(3 * mesh_index + i, 0);
		for (int m = 0; m < pca_size; ++m)
			p1[i] += (T)P_eg(3 * mesh_index + i, m) * pca_coeff[m];
	}
	T p2[3];
	ceres::AngleAxisRotatePoint(R, p1, p2);
	for (int i = 0; i < 3; ++i) {
		p1[i] = p2[i] + tr[i];
	}
	T p3[3];
	double alpha1 = 0.2;
	double alpha2 = 1;

	p3[0] = -1.0 * p1[0] / p1[2] * depth_camera.fx + depth_camera.cx;
	p3[1] = -1.0 * p1[1] / p1[2] * depth_camera.fy + depth_camera.cy;
	p3[2] = p1[2];
	int px = (int)*(double*)&p3[0];
	int py = (int)*(double*)&p3[1];
	T wx = p3[0] - (T)px;
	T wy = p3[1] - (T)py;
	int rx = px + 1, ry = py + 1;
	//if (!(px > 0 && py > 0
	//	&& rx < frame.cols
	//	&& ry < frame.rows)) {
	//	return true;
	//}
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
	T d = T(0);
	for (int i = 0; i < 4; i++) {
		d += (double)frame.at<unsigned short>(ys[i], xs[i]) * ws[i];
		////std::cout << frame.at<unsigned short>(ys[i], xs[i]) << " ";
	}
	////std::cout << "\n";
	residuals[0] = alpha1 * (p3[2] - d);


	for (int i = 0; i < 3; ++i) {
		p1[i] = p1[i] + camera_extrinsic_translation(i);
	}
	p3[0] = -1.0 * p1[0] / p1[2] * rgb_camera.fx + rgb_camera.cx;
	p3[1] = -1.0 * p1[1] / p1[2] * rgb_camera.fy + rgb_camera.cy;
	p3[2] = p1[2];
	residuals[1] = alpha2 * (p3[0] - p2_landmark(0));
	residuals[2] = alpha2 * (p3[1] - p2_landmark(1));

	return true;
}

ceres::CostFunction* CeresLandmarkError::Create(int mesh_index,
	cv::Mat& frame, 
	MatrixXd& M_eg_, MatrixXd& P_eg_,
	Vector2d p2_landmark) {
	// first residual dimension, followed with parameters' dimensions
	return (new ceres::AutoDiffCostFunction<CeresLandmarkError, 2 + 1, 3, 3, pca_size>(
		new CeresLandmarkError(mesh_index,
			frame, 
			M_eg_, P_eg_,
			p2_landmark)));
}

CeresInitializationRegulation::CeresInitializationRegulation(VectorXd& pca_weights)
	:pca_weights(pca_weights)
{
}

template <class T>
bool CeresInitializationRegulation::operator()(const T* const pca_coeff, T* residuals) const
{
	for (int i = 0; i < pca_size; i++){
		residuals[i] = ((T)pca_coeff[i]) * pca_weights(i) * 75.0;
		////std::cout << residuals[i] << "\n";
	}
	////std::cout << "\n";
	return true;
}

ceres::CostFunction* CeresInitializationRegulation::Create(VectorXd& pca_weights) 
{
	// first residual dimension, followed with parameters' dimensions
	return (new ceres::AutoDiffCostFunction<CeresInitializationRegulation, pca_size, pca_size>(
		new CeresInitializationRegulation(pca_weights)));
}