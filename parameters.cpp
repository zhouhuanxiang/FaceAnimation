#include "parameters.h"
using namespace std;

#define LAPTOP 0

#if LAPTOP
string Desktop_Path		= "C:/Users/zhx/Desktop/";
string Test_Output_Dir	= "C:/Users/zhx/Desktop/result/";
string Data_Input_Dir	= "C:/Work/FaceAnimation/";
string Kinect_Data_Dir	= "C:/Work/Kinect2/";
#else
string Desktop_Path		= "C:/Users/zhx/Desktop/";
string Test_Output_Dir	= "C:/Users/zhx/Desktop/result/";
string Data_Input_Dir	= "F:/FaceAnimation/pca_obj_simplified3/";
string Kinect_Data_Dir	= "F:/Kinect2/test12/";
#endif

vector<int> face_landmark =
{
	0,0,0,0,0,
	0,0,0,0,0,
	0,0,0,0,0,
	0,0,

	1804,1125,1401,
	168,167,611,612,2491,
	2500,3090,2072,101,88,
	78,259,1241,1316,2611,
	703,1794,999,1007,1041,
	1045,59,2413,2387,2373,
	546,537,2417,1497,1937,
	1485,1518,2851,3275,2858,
	921,3513,97,2172,477,
	985,945,1,493,2357,
	492,0,933
};

std::set<int> useless_expression =
{
	2, 4, 6, 8, 19
};
