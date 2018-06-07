#ifndef PARAMETERS_
#define PARAMETERS_

#define USE_KINECT 0

#include <set>
#include <vector>
#include <string>
#include <Eigen/Core>
using namespace std;

struct DepthCameraIntrinsic
{
#if USE_KINECT
	double fx = 365.427002;
	double fy = 365.427002;
	double cx = 255.713501;
	double cy = 208.248596;
#else
	double fx = 213.3383;
	double fy = 213.51022;
	double cx = 110.35899;
	double cy = 84.370438;
#endif
};

struct RgbCameraIntrinsic
{
#if USE_KINECT
	double fx = 1081.37;
	double fy = 1081.37;
	double cx = 959.5;
	double cy = 539.5;
#else
	double fx = 892.20422;
	double fy = 893.18079;
	double cx = 666.89142;
	double cy = 382.2774;
#endif
};

extern Eigen::Vector3d CameraExtrinsic;

const int frame_count_begin = 25;
const int frame_count_end = 70;

const int vertex_size = 3223;
const int pca_size = 50;
const int exp_size = 33 - 6;
const int eye_exp_size = 8;
const int mouth_exp_size = 33 - 6 - 8;
const int face_landmark_size = 68;

const int eye_landmark_size = 22;
const int mouth_landmark_size = 12 + 8;
const int total_residual_size = (22 + 12 + 8) * 2;

// GPU
const int stream_size = 100;
const int default_stream = 0;
const int track_stream_begin = 1;
const int refine_stream_begin = 51;
const int refine_default_stream = 99;

extern string Desktop_Path;
extern string Test_Output_Dir;
extern string Data_Input_Dir;
extern string Kinect_Data_Dir;

extern std::vector<int> face_landmark;
extern std::vector<int> useless_expression;
extern std::vector<int> eye_expression;
extern std::vector<int> mouth_expression;

const int motion_param_size = 10;

#endif