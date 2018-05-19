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
		Vector2d p2_landmark,
		Vector3d p3_model,
		double xmin, double xmax, double ymin, double ymax);

	template <class T>
	bool operator()(const T* const y_coeff, T* residuals) const;

	static ceres::CostFunction* Create(cv::Mat& frame,
		Vector2d p2_landmark,
		Vector3d p3_model,
		double xmin, double xmax, double ymin, double ymax);

public:
	cv::Mat& frame;
	Vector2d p2_landmark;
	Vector3d p3_model;

	DepthCameraIntrinsic depth_camera;

	double xmin, xmax, ymin, ymax;
};

class CeresTrackLandmarkError {
public:
	CeresTrackLandmarkError(cv::Mat& frame,
		Vector2d p2_landmark,
		Vector3d p3_model,
		double xmin, double xmax, double ymin, double ymax);

	template <class T>
	bool operator()(const T* const y_coeff, T* residuals) const;

	static ceres::CostFunction* Create(cv::Mat& frame,
		Vector2d p2_landmark,
		Vector3d p3_model,
		double xmin, double xmax, double ymin, double ymax);

public:
	cv::Mat& frame;
	Vector2d p2_landmark;
	Vector3d p3_model;

	DepthCameraIntrinsic depth_camera;
	RgbCameraIntrinsic rgb_camera;
	static Matrix<double, 3, 1> camera_extrinsic_translation;

	double xmin, xmax, ymin, ymax;
};

#endif // !CERES_TRACK_H_