#ifndef CERES_TRACK_H_
#define CERES_TRACK_H_

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


class CeresTrackDenseError {
public:
	CeresTrackDenseError(cv::Mat& frame,
		double weight,
		MatrixXd &neutral_eg,
		MatrixXd &delta_B_eg,
		int index,
		double *param);

	template <class T>
	bool operator()(const T* const x_coeff, T* residuals) const;

	static ceres::CostFunction* Create(cv::Mat& frame,
		double weight,
		MatrixXd &neutral_eg,
		MatrixXd &delta_B_eg,
		int index,
		double *param);

public:
	cv::Mat& frame;
	double weight;

	DepthCameraIntrinsic depth_camera;

	MatrixXd &neutral_eg;
	MatrixXd &delta_B_eg;
	int index;
	double *param;
};

class CeresTrackLandmarkError {
public:
	CeresTrackLandmarkError(cv::Mat& frame,
		Vector2d p2_landmark,
		MatrixXd &neutral_eg,
		MatrixXd &delta_B_eg,
		int index,
		double *param);

	template <class T>
	bool operator()(const T* const x_coeff, T* residuals) const;

	static ceres::CostFunction* Create(cv::Mat& frame,
		Vector2d p2_landmark,
		MatrixXd &neutral_eg,
		MatrixXd &delta_B_eg,
		int index,
		double *param);

public:
	cv::Mat& frame;
	Vector2d p2_landmark;

	DepthCameraIntrinsic depth_camera;
	RgbCameraIntrinsic rgb_camera;
	static Matrix<double, 3, 1> camera_extrinsic_translation;

	MatrixXd &neutral_eg;
	MatrixXd &delta_B_eg;
	int index;
	double *param;
};

class CeresTrackRegulation
{
public:
	CeresTrackRegulation(MatrixXd &xx_coeff, MatrixXd & xxx_coeff);

	template <class T>
	bool operator()(const T* const x_coeff, T* residuals) const;

	static ceres::CostFunction* Create(MatrixXd &xx_coeff, MatrixXd & xxx_coeff);
private:
	MatrixXd &xx_coeff, p;
	MatrixXd & xxx_coeff;
};

#endif // !CERES_TRACK_H_