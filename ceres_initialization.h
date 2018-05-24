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

class CeresFaceDenseError
{
public:
	CeresFaceDenseError(int mesh_index,
		cv::Mat& frame,
		MatrixXd& M_eg_, MatrixXd& P_eg_,
		MatrixXd& normal_eg_,
		double weight,
		bool point_to_point);

	template <class T>
	bool operator()(const T* const R, const T* const tr, const T* const pca_coeff, T* residuals) const;

	static ceres::CostFunction* Create(int mesh_index,
		cv::Mat& frame,
		MatrixXd& M_eg_, MatrixXd& P_eg_,
		MatrixXd& normal_eg_,
		double weight,
		bool point_to_point);

public:
	int mesh_index;
	cv::Mat& frame;
	MatrixXd& M_eg;
	MatrixXd& P_eg;
	MatrixXd& normal_eg;

	DepthCameraIntrinsic depth_camera;

	double weight;
	bool point_to_point;
};

class CeresLandmarkError {
public:
	CeresLandmarkError(int mesh_index,
		cv::Mat& frame, 
		MatrixXd& M_eg_, MatrixXd& P_eg_,
		Vector2d p2_landmark);

	template <class T>
	bool operator()(const T* const R, const T* const tr, const T* const pca_coeff, T* residuals) const;

	static ceres::CostFunction* Create(int mesh_index,
		cv::Mat& frame, 
		MatrixXd& M_eg_, MatrixXd& P_eg_,
		Vector2d p2_landmark);

public:
	int mesh_index;
	cv::Mat& frame;
	MatrixXd& M_eg;
	MatrixXd& P_eg;
	Vector2d p2_landmark;

	DepthCameraIntrinsic depth_camera;
	RgbCameraIntrinsic rgb_camera;
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