#ifndef DEM_H_
#define DEM_H_

#include "ceres_param.h"
#include "ceres_initialization.h"
#include "model_reader.h"
#include "image_reader_kinect.h"
#include "dlib_face_detector.h"
#include "dlib_landmark_detector.h"

#include <Eigen/Core>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <mutex>
#include <chrono>
#include <thread>
#include <vector>
#include <iostream>
#include <algorithm>

using namespace Eigen;

struct DepthCameraIntrinsic
{
	double fx = 365.427002;
	double fy = 365.427002;
	double cx = 255.713501;
	double cy = 208.248596;
};

struct RgbCameraIntrinsic
{
	double fx = 1081.37;
	double fy = 1081.37;
	double cx = 959.5;
	double cy = 539.5;
};

extern MatrixXd M_eg_;
extern MatrixXd P_eg_;
extern MatrixXd M_eg_;
extern MatrixXd P_eg_;
extern MatrixXd delta_B1_eg_;
extern MatrixXd delta_B2_eg_;
extern MatrixXd delta_B_eg_;

extern Vector3d translation_eg_;
extern Matrix<double, 3, 3> rotation_eg_;
extern cv::Mat translation_cv_;
extern cv::Mat rotation_cv_;

extern MatrixXd x_coeff_eg_;
extern MatrixXd xx_coeff_eg_, xxx_coeff_eg_;
extern MatrixXd y_coeff_eg_;
extern VectorXd y_weights_eg_;

extern ml::MeshDatad mesh_;
extern MatrixXd neutral_eg_;
extern MatrixXd expression_eg_;
extern MatrixXd normal_eg_;

// track
extern SparseMatrix<double> A_track_eg_;
extern MatrixXd C_track_eg_;

// refine
//extern DemRefineMiddleWare middleware_;
//extern DemRefine dem_refine_;

extern int frame_count_;
extern cv::Mat dframe_;
extern cv::Mat cframe_;
extern DepthCameraIntrinsic depth_camera_;
extern RgbCameraIntrinsic rgb_camera_;
extern Matrix<double, 3, 3> depth_camera_project_;
extern Matrix<double, 3, 3> depth_camera_reproject_;
extern Matrix<double, 3, 1> camera_extrinsic_translation_;
extern Matrix<double, 3, 3> rgb_camera_project_;
//extern Matrix<double, 3, 3> rgb_camera_reproject_;

extern DlibLandmarkDetector landmark_detector_;

void DEM();

void SolvePnP();

bool UpdateFrame(bool init);

Vector3d ReprojectionDepth(Vector2d p2, int depth);

//Vector3d ReprojectionRgb(Vector2d p2, int depth);

Vector3d ProjectionDepth(Vector3d p3);

Vector3d ProjectionRgb(Vector3d p3);

void Initialize();

void Track();

void UpdateNeutralFaceCPU();

void UpdateDeltaBlendshapeCPU();

void UpdateExpressionFaceCPU();

void UpdateNormalCPU();

void WriteNeutralFace(int count, MatrixXd tmesh);

void WriteExpressionFace(int count, MatrixXd tmesh, Vector3d translation_eg, Matrix<double, 3, 3> rotation_eg);

void WritePointCloud();

#endif