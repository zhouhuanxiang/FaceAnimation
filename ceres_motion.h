#pragma once
#ifndef CERES_MOTION_
#define CERES_MOTION_

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

//class CeresMotionDenseError {
//public:
//	CeresMotionDenseError(cv::Mat& frame,
//		Vector2d p2_landmark,
//		Vector3d p3_model,
//		double xmin, double xmax, double ymin, double ymax);
//
//	template <class T>
//	bool operator()(const T* const R, const T* const tr, T* residuals) const;
//
//	static ceres::CostFunction* Create(cv::Mat& frame,
//		Vector2d p2_landmark,
//		Vector3d p3_model,
//		double xmin, double xmax, double ymin, double ymax);
//
//public:
//	cv::Mat& frame;
//	Vector2d p2_landmark;
//	Vector3d p3_model;
//
//	DepthCameraIntrinsic depth_camera;
//
//	double xmin, xmax, ymin, ymax;
//};

class CeresMotionLandmarkError {
public:
	CeresMotionLandmarkError(cv::Mat& frame,
		Vector2d p2_landmark,
		Vector3d p3_model,
		double xmin, double xmax, double ymin, double ymax);

	template <class T>
	bool operator()(const T* const R, const T* const tr, T* residuals) const;

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

	double xmin, xmax, ymin, ymax;
};


class CeresMotionFitError {
public:
	CeresMotionFitError(double x_value, double y_value);

	template <class T>
	bool operator()(const T* const coeffs, T* residuals) const;

	static ceres::CostFunction* Create(double x_value, double y_value);

public:
	double x_value;
	double y_value;
};

class CeresMotionSmoothError {
public:
	CeresMotionSmoothError(double* p_param, double* pp_param);

	template <class T>
	bool operator()(const T* const R, const T* const tr, T* residuals) const;

	static ceres::CostFunction* Create(double* p_param, double* pp_param);

public:
	double* p_param;
	double* pp_param;
};


#endif