#ifndef DLIB_LANDMARK_DETECOR_H_
#define DLIB_LANDMARK_DETECOR_H_

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

class DlibLandmarkDetector
{
public:
	DlibLandmarkDetector()
		:lcount_(-1), fcount_(-1)
	{
		deserialize("shape_predictor_68_face_landmarks.dat") >> shape_predictor_;
		pts_.resize(shape_predictor_.num_parts());
	}

	bool Detect(cv::Mat& cframe, int frame_count, bool debug = false)
	{
		lcount_ = frame_count;

		// update cframe (with mtx)
		cframe_bgr_mtx_.lock();
		cv::cvtColor(cframe, cframe_bgr_, CV_BGRA2BGR);
		cframe_bgr_mtx_.unlock();
		// Change to dlib's image format. No memory is copied.
		cv_image<bgr_pixel> cimg(cframe_bgr_);

		while (fcount_ == -1 || lcount_ - fcount_ > 10) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}

		LOG(INFO) << "landmark detect " << lcount_ - fcount_;
		// read face to local (with mtx)
		rectangle face_local;
		ReadFace(face_local);
		
		full_object_detection shape = shape_predictor_(cimg, face_local);

		for (int i = 0; i < shape.num_parts(); i++) {
			pts_[i] = Eigen::Vector2d(shape.part(i).x(), shape.part(i).y());
		}
		xmin = face_local.left();
		xmax = face_local.right();
		ymin = face_local.top();
		ymax = face_local.bottom();

		// Custom Face Render
		if (debug) {
			for (int i = 0; i < shape.num_parts(); i++) {
				if (i < 17)	continue;
				cv::circle(cframe, cv::Point(pts_[i](0), pts_[i](1)), 2, cv::Scalar(0, 0, 255, 255));
			}
		}

		return true;
	}

	void UpdateFace(rectangle& r)
	{
		face_mtx_.lock();
		face_ = r;
		face_mtx_.unlock();
	}

	void ReadFace(rectangle& r)
	{
		face_mtx_.lock();
		r = face_;
		face_mtx_.unlock();
	}

	void UpdateFrame(cv::Mat& frame)
	{
		cframe_bgr_mtx_.lock();
		cframe_bgr_ = frame;
		cframe_bgr_mtx_.unlock();
	}

	void ReadFrame(cv::Mat& frame)
	{
		cframe_bgr_mtx_.lock();
		frame = cframe_bgr_;
		cframe_bgr_mtx_.unlock();
	}

public:
	std::vector<Eigen::Vector2d> pts_;
	int xmin, xmax, ymin, ymax;

	int lcount_;// landmark count
	int fcount_;// face count

private:
	shape_predictor shape_predictor_;

	rectangle face_;
	cv::Mat cframe_bgr_;
	std::mutex face_mtx_;
	std::mutex cframe_bgr_mtx_;
};

#endif