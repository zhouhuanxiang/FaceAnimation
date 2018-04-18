#ifndef DEM_H_
#define DEM_H_

#include "ceres_param.h"
#include "ceres_initialization.h"
#include "model_reader.h"
#include "image_reader_kinect.h"
#include "dlib_face_detector.h"
#include "dlib_landmark_detector.h"
#include "cuda/cuda_helper_func.cuh"

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

extern cublasHandle_t handle;
extern cusparseHandle_t shandle;
extern std::vector<cudaStream_t> streams_cu_;
extern CuDenseMatrix inter_result1_;
extern CuDenseMatrix inter_result2_;
extern CuDenseMatrix inter_result3_;
extern CuDenseMatrix inter_result4_;

extern MatrixXd M_eg_;
extern MatrixXd P_eg_;
//extern MatrixXd delta_B1_eg_;
//extern MatrixXd delta_B2_eg_;
extern CuDenseMatrix M_cu_;
extern CuDenseMatrix P_cu_;
extern CuDenseMatrix delta_B_cu_;
extern CuDenseMatrix delta_B1_cu_;
extern CuDenseMatrix delta_B2_cu_;

extern Vector3d translation_eg_;
extern Matrix<double, 3, 3> rotation_eg_;
extern cv::Mat translation_cv_;
extern cv::Mat rotation_cv_;

extern MatrixXd x_coeff_eg_;
extern MatrixXd xx_coeff_eg_, xxx_coeff_eg_;
extern MatrixXd y_coeff_eg_;
extern CuDenseMatrix x_coeff_cu_;
extern CuDenseMatrix y_coeff_cu_;

extern ml::MeshDatad mesh_;
extern Map<MatrixXd> neutral_eg_;
extern Map<MatrixXd> expression_eg_;
extern CuDenseMatrix neutral_cu_;
extern CuDenseMatrix expression_cu_;

// track
extern CuSparseMatrix A_track_cu_;
extern CuDenseMatrix C_track_cu_;
extern CuDenseMatrix X_track_cu_;
extern CuDenseMatrix Y_track_cu_;

// refine
//extern DemRefine dem_refine_;c

extern int frame_count_;
extern cv::Mat dframe_;
extern cv::Mat cframe_;

extern Camera camera_;

void DEM();

void SolvePnP();

void GetFrame(bool init);

Vector3d Point2d_2_Point3d(Vector2d p2, int depth);

Vector2d Point3d_2_Point2d(Vector3d p3);

void Initialize();

void Track();

// y updated
void UpdateNeutralFaceGPU();

void UpdateNeutralFaceCPU();

// y updated
void UpdateDeltaBlendshapeGPU();

// x & y updated
void UpdateExpressionFaceGPU();

void UpdateExpressionFaceCPU();

void UpdateNormalCPU();

void WriteNeutralFace();

void WriteExpressionFace();

class DemRefine
{
public:
	DemRefine();

	void operator()();
	// call outside
	void GetY(CuDenseMatrix &dm, cudaStream_t &stream);
	// call inside
	void UpdateY(MatrixXd &result);
	// call inside
	void GetX(cudaStream_t &stream);
	// call outside
	void UpdateX(MatrixXd &x, CuSparseMatrix A_in, CuDenseMatrix C_in, cudaStream_t &stream);

public:
	double S_re_;
	double S_total_re_;
	double *p_X_eg_;
	double *p_Y_eg_;
	Map<MatrixXd> X_eg_;
	Map<MatrixXd> Y_eg_;
	CuDenseMatrix X_re_;
	CuDenseMatrix Y_re_;
	// in
	bool updated;
	std::mutex x_mtx_;
	MatrixXd x_in_;
	CuSparseMatrix A_in_;
	CuDenseMatrix C_in_;
	// out
	std::mutex y_mtx_;
	MatrixXd y_coeff_re_;
	// run time variable
	MatrixXd x_coeff_re_;
	CuSparseMatrix A_re_;
	CuDenseMatrix C_re_;
	CuDenseMatrix A_hat_cu_;
	CuDenseMatrix C_hat_cu_;
};


#endif