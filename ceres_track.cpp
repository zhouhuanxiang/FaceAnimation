#include "ceres_track.h"

CeresTrackDenseError::CeresTrackDenseError(cv::Mat& frame,
	Vector2d p2_landmark,
	Vector3d p3_model,
	double xmin, double xmax, double ymin, double ymax)
	:frame(frame),
	p2_landmark(p2_landmark),
	p3_model(p3_model),
	xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax)
{}

template <class T>
bool CeresTrackDenseError::operator()(const T* const y_coeff, T* residuals) const
{
	for (int i = 0; i < 1; i++)
		residuals[i] = T(0);

	T p1[3];
	for (int i = 0; i < 3; ++i) {
		p1[i] = (T)p3_model(i);
	}
	T p2[3];
	ceres::AngleAxisRotatePoint(R, p1, p2);
	for (int i = 0; i < 3; ++i) {
		p1[i] = p2[i] + tr[i];
	}
	T p3[3];
	p3[0] = -1.0 * p1[0] / p1[2] * depth_camera.fx + depth_camera.cx;
	p3[1] = -1.0 * p1[1] / p1[2] * depth_camera.fy + depth_camera.cy;
	p3[2] = p1[2];

	int px = (int)*(double*)&p3[0];
	int py = (int)*(double*)&p3[1];
	T wx = p3[0] - (T)px;
	T wy = p3[1] - (T)py;
	int rx = px + 1, ry = py + 1;
	//if (rx < xmin || ry < ymin || px > xmax || py > ymax) {
	//	if (rx < xmin - 10)
	//		residuals[0] = residuals[0] + xmin - (double)rx;
	//	if (ry < ymin - 10)
	//		residuals[0] = residuals[0] + ymin - (double)ry;
	//	if (px > xmax + 10)
	//		residuals[0] = residuals[0] + (double)px - xmax;
	//	if (py > ymax + 10)
	//		residuals[0] = residuals[0] + (double)py - ymax;
	//	return true;
	//}
	if (!(px > 0 && py > 0
		&& rx < frame.cols
		&& ry < frame.rows)) {
		return true;
	}

	int xs[4], ys[4];
	xs[0] = px; ys[0] = py;
	xs[1] = rx; ys[1] = py;
	xs[2] = px; ys[2] = ry;
	xs[3] = rx; ys[3] = ry;
	T ws[4];
	ws[0] = ((T)1. - wx) * ((T)1. - wy);
	ws[1] = wx * ((T)1. - wy);
	ws[2] = ((T)1. - wx) * wy;
	ws[3] = wx * wy;
	T d = T(0);
	for (int i = 0; i < 4; i++) {
		d += (double)frame.at<unsigned short>(ys[i], xs[i]) * ws[i];
		//std::cout << frame.at<unsigned short>(ys[i], xs[i]) << " ";
	}
	//std::cout << "\n";
	residuals[0] = (p3[2] - d);

	return true;
}

ceres::CostFunction* CeresTrackDenseError::Create(cv::Mat& frame,
	Vector2d p2_landmark,
	Vector3d p3_model,
	double xmin, double xmax, double ymin, double ymax)
{
	return (new ceres::AutoDiffCostFunction<CeresTrackDenseError, 1, 3, 3>(
		new CeresTrackDenseError(frame,
			p2_landmark,
			p3_model,
			xmin, xmax, ymin, ymax)));
}

Matrix<double, 3, 1> CeresTrackLandmarkError::camera_extrinsic_translation = Matrix<double, 3, 1>();

CeresTrackLandmarkError::CeresTrackLandmarkError(cv::Mat& frame,
	Vector2d p2_landmark,
	Vector3d p3_model,
	double xmin, double xmax, double ymin, double ymax)
	:frame(frame),
	p2_landmark(p2_landmark),
	p3_model(p3_model),
	xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax)
{}

template <class T>
bool CeresTrackLandmarkError::operator()(const T* const y_coeff, T* residuals) const
{
	for (int i = 0; i < 2; i++)
		residuals[i] = T(0);

	T p1[3];
	for (int i = 0; i < 3; ++i) {
		p1[i] = (T)p3_model(i);
	}
	T p2[3];
	ceres::AngleAxisRotatePoint(R, p1, p2);
	for (int i = 0; i < 3; ++i) {
		p1[i] = p2[i] + tr[i];
	}
	T p3[3];
	double alpha1 = 0.2;
	double alpha2 = 1;

	p3[0] = -1.0 * p1[0] / p1[2] * depth_camera.fx + depth_camera.cx;
	p3[1] = -1.0 * p1[1] / p1[2] * depth_camera.fy + depth_camera.cy;
	p3[2] = p1[2];
	int px = (int)*(double*)&p3[0];
	int py = (int)*(double*)&p3[1];
	T wx = p3[0] - (T)px;
	T wy = p3[1] - (T)py;
	int rx = px + 1, ry = py + 1;
	//if (!(px > 0 && py > 0
	//	&& rx < frame.cols
	//	&& ry < frame.rows)) {
	//	return true;
	//}
	int xs[4], ys[4];
	xs[0] = px; ys[0] = py;
	xs[1] = rx; ys[1] = py;
	xs[2] = px; ys[2] = ry;
	xs[3] = rx; ys[3] = ry;
	T ws[4];
	ws[0] = ((T)1. - wx) * ((T)1. - wy);
	ws[1] = wx * ((T)1. - wy);
	ws[2] = ((T)1. - wx) * wy;
	ws[3] = wx * wy;
	T d = T(0);
	for (int i = 0; i < 4; i++) {
		d += (double)frame.at<unsigned short>(ys[i], xs[i]) * ws[i];
		//std::cout << frame.at<unsigned short>(ys[i], xs[i]) << " ";
	}
	//std::cout << "\n";
	residuals[0] = alpha1 * (p3[2] - d);


	for (int i = 0; i < 3; ++i) {
		p1[i] = p1[i] + camera_extrinsic_translation(i);
	}
	p3[0] = -1.0 * p1[0] / p1[2] * rgb_camera.fx + rgb_camera.cx;
	p3[1] = -1.0 * p1[1] / p1[2] * rgb_camera.fy + rgb_camera.cy;
	p3[2] = p1[2];
	residuals[1] = alpha2 * (p3[0] - p2_landmark(0));
	residuals[2] = alpha2 * (p3[1] - p2_landmark(1));

	return true;
}

ceres::CostFunction* CeresTrackLandmarkError::Create(cv::Mat& frame,
	Vector2d p2_landmark,
	Vector3d p3_model,
	double xmin, double xmax, double ymin, double ymax)
{
	return (new ceres::AutoDiffCostFunction<CeresTrackLandmarkError, 3, 3, 3>(
		new CeresTrackLandmarkError(frame,
			p2_landmark,
			p3_model,
			xmin, xmax, ymin, ymax)));
}
