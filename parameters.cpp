#include "parameters.h"
using namespace std;

#define USE_LAPTOP 1

#if USE_LAPTOP
	string Desktop_Path		= "C:/Users/zhx/Desktop/";
	string Test_Output_Dir	= "C:/Users/zhx/Desktop/result/";
	string Data_Input_Dir	= "C:/Work/data/pca_obj_simplified4/";
	string Kinect_Data_Dir	= "C:/Work/data/zhx2/";
#else
	string Desktop_Path		= "C:/Users/zhx/Desktop/";
	string Test_Output_Dir	= "C:/Users/zhx/Desktop/result/";
	string Data_Input_Dir	= "F:/FaceAnimation/pca_obj_simplified4/";
	#if USE_KINECT
		string Kinect_Data_Dir	= "F:/Kinect2/new1/";
	#else
		string Kinect_Data_Dir = "F:/SunnuMars/zhx2/";
	#endif
#endif

vector<int> face_landmark =
{
	0,0,0,0,0,
	0,0,0,0,0,
	0,0,0,0,0,
	0,0,

	//1804,1125,1401,
	//168,167,611,612,2491,
	//2500,3090,2072,101,88,
	//78,259,1241,1316,2611,
	//703,1794,999,1007,1041,
	//1045,59,2413,2387,2373,
	//546,537,2417,1497,1937,
	//1485,1518,2851,3275,2858,
	//921,3513,97,2172,477,
	//985,945,1,493,2357,
	//492,0,933

	937,1097,928,
	931,1345,2556,2155,2151,
	2314,2162,1832,65,53,
	43,223,1052,996,2273,
	609,873,827,833,859,
	862,840,448,2063,2053,
	2097,443,2086,1297,1685,
	306,1291,692,2886,2510,
	803,687,61,301,417,
	1313,811,1,432,2525,
	431,0,808
};

std::vector<int> useless_expression =
{
	2, 4, 6, 8, 19, 22
};

std::vector<int> eye_expression =
{
	0, 3, 5, 7, 25, 27, 29, 31
};

std::vector<int> mouth_expression = 
{
	1,9,10,11,12,13,14,15,16,17,18,20,21,23,24,26,28,30,32
};

#if USE_KINECT
Eigen::Vector3d CameraExtrinsic((Eigen::Vector3d() << 52.0, 0.0, 0.0).finished());
#else
Eigen::Vector3d CameraExtrinsic((Eigen::Vector3d() << 10.429403, 0.032413036, -2.6441102).finished());
#endif
