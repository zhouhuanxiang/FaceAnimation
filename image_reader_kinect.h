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

		cv::Mat tmp;

		char str[20];
		if (USE_KINECT){
			sprintf(str, "c%d.png", idx);
			cframe = cv::imread(path + str/*, cv::IMREAD_UNCHANGED*/);
			//cv::pyrUp(cframe, tmp, cframe.size() * 2);
			//cv::pyrUp(tmp, cframe, tmp.size() * 2);
		}
		//cv::imshow("infrared", iframe);
		//cv::waitKey(0);

		sprintf(str, "d%d.png", idx);
		dframe = cv::imread(path + str, cv::IMREAD_UNCHANGED);
		//cv::pyrUp(dframe, tmp, dframe.size() * 2);
		//cv::pyrUp(tmp, dframe, tmp.size() * 2);
	}
private:
	string path;
};

#endif 