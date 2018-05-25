#include "dem1.h"

bool SVD()
{
	int landmark_begin = 27;
	int landmark_end = 48;

	std::vector<double> landmark_p3_depth(face_landmark_size, 0);
	std::vector<double> landmark_weight(face_landmark_size, 0);
	std::vector<double> landmark_p3_depth_approx(face_landmark_size);
	std::vector<double> landmark_distance(face_landmark_size, DBL_MAX);
	for (int j = 0; j < dframe_.rows; j++) {
		for (int i = 0; i < dframe_.cols; i++) {
			double d = dframe_.at<unsigned short>(j, i);
			Vector3d p3 = ReprojectionDepth(Vector2d(i, j), d);
			p3 -= CameraExtrinsic;
			Vector3d p2 = ProjectionRgb(p3);
			for (int k = landmark_begin; k < landmark_end; k++) {
				double px = std::abs(p2.x() - landmark_detector_.pts_[k].x());
				double py = std::abs(p2.y() - landmark_detector_.pts_[k].y());
				if (px < 1 && py < 1) {
					double w = (1 - px) * (1 - py);
					landmark_p3_depth[k] += d * w;
					landmark_weight[k] += w;
				}
				else if (px + py < landmark_distance[k] && landmark_weight[k] == 0) {
					landmark_p3_depth_approx[k] = d;
					landmark_distance[k] = px + py;
				}
			}
		}
	}

	for (int k = landmark_begin; k < landmark_end; k++) {
		if (landmark_weight[k] != 0) {
			landmark_p3_depth[k] /= landmark_weight[k];
		}
		else if (landmark_distance[k] < 5) {
			landmark_p3_depth[k] = landmark_p3_depth_approx[k];
		}
	}

	int count = 0;
	std::vector<Vector3d> ps1, ps2;

	for (int i = landmark_begin; i < landmark_end; i++) {
		if (landmark_p3_depth[i] < 50)
			continue;
		Vector3d p1 = ReprojectionDepth(landmark_detector_.pts_[i], landmark_p3_depth[i]);
		Vector3d p2 = expression_eg_.block(3 * face_landmark[i], 0, 3, 1);
		//if ((p3_landmark - p3_model_now).norm() > 50)
		//	continue;
		ps1.push_back(p1);
		ps2.push_back(p2);
		count++;
	}

	if (count == 0) {
		std::cout << "wrong svd\n";
		return false;
	}

	MatrixXd pts1(3, count);
	MatrixXd pts2(3, count);
	for (int i = 0; i < count; i++) {
		pts1.col(i) = ps1[i];
		pts2.col(i) = ps2[i];
	}
	Vector3d centroid1 = pts1.rowwise().mean();
	Vector3d centroid2 = pts2.rowwise().mean();
	pts1.colwise() -= centroid1;
	pts2.colwise() -= centroid2;


	JacobiSVD<MatrixXd> svd(pts2 * pts1.transpose(), ComputeThinU | ComputeThinV);
	rotation_eg_ = svd.matrixV() * svd.matrixU().transpose();
	if (rotation_eg_.determinant() < 0) {
		rotation_eg_.col(2) *= -1;
	}
	translation_eg_ = centroid1 - rotation_eg_ * centroid2;

	////LOG(INFO) << "rotation:" << rotation_eg_;
	//LOG(INFO) << "translation:" << Map<RowVectorXd>(translation_eg_.data(), 3);
	std::cout << Map<RowVectorXd>(translation_eg_.data(), 3) << "@" << count << "\n";
	////std::cout << rotation_eg_ << "\n@";

	return true;
}

void UpdateMotion(cv::Mat &dframe, std::vector<Eigen::Vector2d> pts,
	MatrixXd expression_eg,
	int xmin, int xmax, int ymin, int ymax)
{
	//LOG(INFO) << "update motion start";
	std::chrono::steady_clock::time_point tp1 = std::chrono::steady_clock::now();
	ceres::Problem problem1;
	ceres::Solver::Options options1;
	options1.linear_solver_type = ceres::DENSE_QR;
	options1.minimizer_progress_to_stdout = false;
	options1.max_num_iterations = 500;
	options1.num_threads = 1;
	ceres::LossFunctionWrapper* loss_function_wrapper1 = new ceres::LossFunctionWrapper(new ceres::HuberLoss(1.0), ceres::TAKE_OWNERSHIP);

	for (int i = 27; i <= 47; i++) {
		problem1.AddResidualBlock(
			CeresMotionLandmarkError::Create(dframe,
				pts[i],
				expression_eg.block(3 * face_landmark[i], 0, 3, 1),
				xmin, xmax, ymin, ymax),
			loss_function_wrapper1,
			motion_param_tmp, motion_param_tmp + 3
		);
	}

	ceres::Solver::Summary summary1;
	ceres::Solve(options1, &problem1, &summary1);
	std::chrono::steady_clock::time_point tp2 = std::chrono::steady_clock::now();
	track_time_ += std::chrono::duration_cast<std::chrono::milliseconds>(tp2 - tp1).count();
	//LOG(INFO) << "update motion end";

	motion_param_updated = true;
	//for (int i = 0; i < 6; i++) {
	//	std::cout << motion_param_tmp[i] << " ";
	//}
	//std::cout << "@update\n";

	//ceres::Problem problem2;
	//for (int i = 0; i < vertex_size; i += 25) {
	//	problem1.AddResidualBlock(
	//		CeresMotionDenseError::Create(dframe_,
	//			Vector2d(0, 0),
	//			expression_eg_.block(3 * i, 0, 3, 1),
	//			landmark_detector_.xmin, landmark_detector_.xmax, landmark_detector_.ymin, landmark_detector_.ymax),
	//		0,
	//		motion_param, motion_param + 3
	//	);
	//}
	//ceres::Solve(options1, &problem1, &summary1);

	//for (int i = 0; i < vertex_size; i += 25) {
	//	CeresMotionDenseError error = CeresMotionDenseError(dframe_,
	//		Vector2d(0, 0),
	//		expression_eg_.block(3 * i, 0, 3, 1),
	//		landmark_detector_.xmin, landmark_detector_.xmax, landmark_detector_.ymin, landmark_detector_.ymax);
	//	double residuals;
	//	error(motion_param, motion_param + 3, &residuals);
	//	//std::cout << setw(15) << residuals << "\n";
	//	////LOG(INFO) << setw(15) << residuals;
	//}

	//for (int i = 27; i <= 47; i++) {
	//	CeresMotionLandmarkError error = CeresMotionLandmarkError(dframe_,
	//		landmark_detector_.pts_[i],
	//		expression_eg_.block(3 * face_landmark[i], 0, 3, 1),
	//		landmark_detector_.xmin, landmark_detector_.xmax, landmark_detector_.ymin, landmark_detector_.ymax);
	//	double residuals[3];
	//	error(motion_param, motion_param + 3, residuals);
	//	//std::cout << setw(15) << residuals[0] << " " << setw(15) << residuals[1] << " " << setw(15) << residuals[2] << "\n";
	//	////LOG(INFO) << setw(15) << residuals[0] << " " << setw(15) << residuals[1] << "\n";
	//}

	//Ceres2Eigen(rotation_eg_, translation_eg_, motion_param[motion_param_ptr]);
	////LOG(INFO) << "translation: " << Map<RowVectorXd>(translation_eg_.data(), 3);
	////std::cout << "translation: " << Map<RowVectorXd>(translation_eg_.data(), 3) << "@\n";
}

bool UpdateFrame(bool force_motion)
{
	static ImageReaderKinect image_reader(Kinect_Data_Dir);
	image_reader.GetFrame(frame_count_, cframe_, dframe_);
	//return true;
	landmark_detector_.Detect(cframe_, frame_count_, false);
	//return true;

	//LOG(INFO) << "rigid motion";
	//SolvePnP();
	//WriteExpressionFace(frame_count_, expression_eg_, translation_eg_, rotation_eg_);
	//if (!SVD()) {
	//	WriteExpressionFace(frame_count_, expression_eg_, translation_eg_, rotation_eg_);
	//	return false;
	//}
	//WriteExpressionFace(frame_count_, expression_eg_, translation_eg_, rotation_eg_);

	if (!force_motion) {
		while (!motion_param_updated) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		motion_param_updated = false;
		for (int i = 0; i < 6; i++) {
			motion_param[motion_param_ptr][i] = motion_param_tmp[i];
		}
		motion_param_ptr = (frame_count_) % motion_param_size;

		UpdateMotion(dframe_,
			landmark_detector_.pts_,
			expression_eg_,
			landmark_detector_.xmin,
			landmark_detector_.xmax,
			landmark_detector_.ymin,
			landmark_detector_.ymax);

		FitMotion();
		return true;
	}
	else {
		UpdateMotion(dframe_,
			landmark_detector_.pts_,
			expression_eg_,
			landmark_detector_.xmin,
			landmark_detector_.xmax,
			landmark_detector_.ymin,
			landmark_detector_.ymax);
		motion_param_ptr = (frame_count_) % motion_param_size;
		for (int i = 0; i < 6; i++) {
			motion_param[motion_param_ptr][i] = motion_param_tmp[i];
		}
		//SVD();
	}

	return true;
}