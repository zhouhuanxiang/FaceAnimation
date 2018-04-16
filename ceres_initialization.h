#ifndef CERES_INITIALIZATION_
#define CERES_INITIALIZATION_

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <Eigen/Eigen>
#include <Eigen/Sparse>
using namespace std;
using namespace Eigen;

#include "parameters.h"

class CeresInitializationError {
public:
	CeresInitializationError(int index, cv::Mat& frame, Matrix3Xd& frame_normals, Matrix3Xd& model_normals, MatrixXd& pca_models,
		double fx, double fy, double cx, double cy, bool print = false);

	template <class T>
	bool operator()(const T* const R, const T* const tr, const T* const pca_coeff, T* residuals) const;

	static ceres::CostFunction* Create(int index, cv::Mat& frame, Matrix3Xd& frame_normals, Matrix3Xd& model_normals, MatrixXd& pca_models,
		double fx, double fy, double cx, double cy, bool print = false);

private:
	int index;
	cv::Mat& frame;
	Matrix3Xd& frame_normals;
	Matrix3Xd& model_normals;
	MatrixXd& pca_models;
	double fx, fy, cx, cy;
	bool print;
};

class CeresLandmarkError {
public:
	CeresLandmarkError(int index, int index1, 
		cv::Mat& frame, 
		MatrixXd& M_eg_, MatrixXd& P_eg_,
		double fx, double fy, double cx, double cy, 
		Vector3d p3_landmark);

	template <class T>
	bool operator()(const T* const R, const T* const tr, const T* const pca_coeff, T* residuals) const;

	static ceres::CostFunction* Create(int index, int index1, 
		cv::Mat& frame, 
		MatrixXd& M_eg_, MatrixXd& P_eg_,
		double fx, double fy, double cx, double cy, 
		Vector3d p3_landmark);

private:
	int index;
	int index1;
	cv::Mat& frame;
	MatrixXd& M_eg;
	MatrixXd& P_eg;
	double fx, fy, cx, cy;
	Vector3d p3_landmark;

};

class CeresInitializationRegulation
{
public:
	CeresInitializationRegulation(VectorXd& pca_weights);

	template <class T>
	bool operator()(const T* const pca_coeff, T* residuals) const;

	static ceres::CostFunction* Create(VectorXd& pca_weights);
private:
	VectorXd& pca_weights;
};

#endif