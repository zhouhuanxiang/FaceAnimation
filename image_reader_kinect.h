#ifndef IMAGE_READER_KINECT_
#define IMAGE_READER_KINECT_

#include <glog/logging.h>

#include <string>
#include <fstream>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "parameters.h"

using namespace std;

class ImageReaderKinect
{
public:
	ImageReaderKinect(string p)
	{
		path = p;
	}

	void GetFrame(int idx, cv::Mat& cframe, cv::Mat& dframe)
	{
		LOG(INFO) << "read image (I/O)";

		char str[20];
		if (USE_KINECT){
			sprintf(str, "c%d.png", idx);
			cframe = cv::imread(path + str, cv::IMREAD_UNCHANGED);
		}
		//cv::imshow("infrared", iframe);
		//cv::waitKey(0);

		sprintf(str, "d%d.png", idx);
		dframe = cv::imread(path + str, cv::IMREAD_UNCHANGED);
	}
private:
	string path;
};

#endif 