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

// GPU
const int stream_size = 100;
const int default_stream = 0;
const int track_stream_begin = 1;
const int refine_stream_begin = 51;
const int track_2_refine_stream = 99;
const int refine_2_track_stream = 98;
const int refine_default_stream = 97;


extern string Desktop_Path;
extern string Test_Output_Dir;
extern string Data_Input_Dir;
extern string Kinect_Data_Dir;

extern std::vector<int> face_landmark;
extern std::set<int> useless_expression;

#endif