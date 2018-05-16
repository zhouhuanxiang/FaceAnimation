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

class CeresMotionDenseError {
public:
	CeresMotionDenseError(cv::Mat& frame,
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
	static Matrix<double, 3, 1> camera_extrinsic_translation;

	double xmin, xmax, ymin, ymax;
};

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
	static Matrix<double, 3, 1> camera_extrinsic_translation;

	double xmin, xmax, ymin, ymax;
};

#endif