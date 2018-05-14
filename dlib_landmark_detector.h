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

		detector_ = get_frontal_face_detector();
	}

	bool test(cv::Mat& cframe)
	{
		cv::Mat cframe_bgr_;
		if (cframe.channels() == 4)
			cv::cvtColor(cframe, cframe_bgr_, CV_BGRA2BGR);
		else
			cframe_bgr_ = cframe;
		// Change to dlib's image format. No memory is copied.
		cv_image<bgr_pixel> cimg(cframe_bgr_);

		rectangle face_local;
		face_local = detector_(cimg)[0];

		full_object_detection shape = shape_predictor_(cimg, face_local);

		for (int i = 0; i < shape.num_parts(); i++) {
			pts_[i] = Eigen::Vector2d(shape.part(i).x(), shape.part(i).y());
		}
		xmin = face_local.left();
		xmax = face_local.right();
		ymin = face_local.top();
		ymax = face_local.bottom();

		// Custom Face Render
		for (int i = 0; i < shape.num_parts(); i++) {
			if (i < 17)	continue;
			cv::circle(cframe, cv::Point(pts_[i](0), pts_[i](1)), 3, cv::Scalar(0, 0, 255, 255), -1);
		}
		cv::imshow("debug", cframe);
		cv::waitKey(0);
	}

	bool Detect(cv::Mat cframe, int frame_count, bool debug = false)
	{
		lcount_ = frame_count;

		// update cframe (with mtx)
		cframe_bgr_mtx_.lock();
		cframe.copyTo(cframe_bgr_);
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
		//
		//face_local = detector_(cimg)[0];
		//

		cv::Mat cframe_bgr_face = cframe_bgr_(cv::Rect(face_local.left(), face_local.top(), face_local.width(), face_local.height())).clone();

		cv::Mat tmp;
		cv::pyrUp(cframe_bgr_face, tmp, cframe_bgr_face.size() * 2);
		cv::pyrUp(tmp, cframe_bgr_face, tmp.size() * 2);
		cv::pyrUp(cframe_bgr_face, tmp, cframe_bgr_face.size() * 2);
		cv::pyrUp(tmp, cframe_bgr_face, tmp.size() * 2);

		rectangle face_local_new = rectangle(0, 0, face_local.width() * 16, face_local.height() * 16);
		cv_image<bgr_pixel> cimg_new(cframe_bgr_face);
		full_object_detection shape = shape_predictor_(cimg_new, face_local_new);
		for (int i = 0; i < shape.num_parts(); i++) {
			pts_[i] = Eigen::Vector2d(shape.part(i).x() / 16.0 + face_local.left(), shape.part(i).y() / 16.0 + face_local.top()) ;
		}

		//full_object_detection shape = shape_predictor_(cimg, face_local);
		//for (int i = 0; i < shape.num_parts(); i++) {
		//	pts_[i] = Eigen::Vector2d(shape.part(i).x(), shape.part(i).y());
		//}
		xmin = face_local.left();
		xmax = face_local.right();
		ymin = face_local.top();
		ymax = face_local.bottom();

		// Custom Face Render
		if (debug) {
			for (int i = 0; i < shape.num_parts(); i++) {
				if (i < 17)	continue;
				cv::circle(cframe_bgr_, cv::Point(pts_[i](0), pts_[i](1)), 2, cv::Scalar(0, 0, 255, 255));
			}
			cv::imshow("debug", cframe_bgr_);
			cv::waitKey(0);
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

	frontal_face_detector detector_;
};

#endif