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

struct Camera
{
	double fx = 365.427002;
	double fy = 365.427002;
	double cx = 255.713501;
	double cy = 208.248596;
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
extern MatrixXd X_refine_eg_;
extern MatrixXd Y_refine_eg_;

// refine
//extern DemRefineMiddleWare middleware_;
//extern DemRefine dem_refine_;

extern int frame_count_;
extern cv::Mat dframe_;
extern cv::Mat cframe_;
extern Camera camera_;
extern DlibLandmarkDetector landmark_detector_;

void DEM();

void SolvePnP();

bool UpdateFrame(bool init);

Vector3d Point2d_2_Point3d(Vector2d p2, int depth);

Vector2d Point3d_2_Point2d(Vector3d p3);

void Initialize();

void Track();

void UpdateNeutralFaceCPU();

void UpdateDeltaBlendshapeCPU();

void UpdateExpressionFaceCPU();

void UpdateNormalCPU();

void WriteNeutralFace();

void WriteExpressionFace();

#endif