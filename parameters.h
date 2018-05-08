#ifndef PARAMETERS_
#define PARAMETERS_

#include <set>
#include <vector>
#include <string>
using namespace std;

#define USE_KINECT 1

const int vertex_size = 3641;
const int face_size = 7028;
const int pca_size = 50;
const int exp_size = 33 - 5;
const int face_landmark_size = 68;
const int landmark_residual_pair = 68;
const int dense_residual_pair = 365;
const int total_residual_size = 68 * 6 + 365 * 3;

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
extern std::set<int> useless_expression;

#endif