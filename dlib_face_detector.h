#ifndef DLIB_FACE_DETECTOR_
#define DLIB_FACE_DETECTOR_

#include <glog/logging.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
using namespace dlib;

#include "parameters.h"

#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <mutex>
#include <chrono>
#include <thread>
using namespace std;

#include "dlib_landmark_detector.h"

class DlibFaceDetector
{
public:
	DlibFaceDetector(DlibLandmarkDetector &ld)
		:ld_(ld)
	{
		detector_ = get_frontal_face_detector();
	}

	void operator()()
	{
		while (true) {
			// update count (with mtx)
			if (ld_.fcount_ == ld_.lcount_) {
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
				continue;
			}

			// read cframe to local (with mtx)
			cv::Mat cframe_bgr_local;
			ld_.ReadFrame(cframe_bgr_local);
			int lcount = ld_.lcount_;
			// Change to dlib's image format. No memory is copied.
			cv_image<bgr_pixel> cimg(cframe_bgr_local);

			LOG(WARNING) << "face detect No." << ld_.lcount_;

			std::vector<rectangle> faces = detector_(cimg);
			if (faces.size() == 1) {
				// update face (with mtx)
				ld_.UpdateFace(faces[0]);
				ld_.fcount_ = lcount;
			}
			else {
				std::cout << "totally " << faces.size() << " faces detected!\n";
				system("pause");
			}
		}
	}

private:
	frontal_face_detector detector_;

	DlibLandmarkDetector &ld_;
};


#endif
