#include "dem.h"

#include <chrono>

MatrixXd M_eg_;
MatrixXd P_eg_;
MatrixXd delta_B1_eg_;
MatrixXd delta_B2_eg_;
MatrixXd delta_B_eg_;

Vector3d translation_eg_;
Matrix<double, 3, 3> rotation_eg_;
cv::Mat translation_cv_;
cv::Mat rotation_cv_;

MatrixXd x_coeff_eg_;
MatrixXd xx_coeff_eg_, xxx_coeff_eg_;
MatrixXd y_coeff_eg_;
VectorXd y_weights_eg_;

ml::MeshDatad mesh_;
MatrixXd neutral_eg_;
MatrixXd expression_eg_;
MatrixXd normal_eg_;

// track
SparseMatrix<double> A_track_eg_;
MatrixXd C_track_eg_;

// refine
//DemRefineMiddleWare middleware_;
//DemRefine dem_refine_;

int frame_count_;
cv::Mat dframe_;
cv::Mat cframe_;
DepthCameraIntrinsic depth_camera_;
RgbCameraIntrinsic rgb_camera_;
Matrix<double, 3, 3> depth_camera_project_;
Matrix<double, 3, 3> depth_camera_reproject_;
Matrix<double, 3, 1> camera_extrinsic_translation_;
Matrix<double, 3, 3> rgb_camera_project_;
//Matrix<double, 3, 3> rgb_camera_reproject_;

DlibLandmarkDetector landmark_detector_;

void DEM()
{
	//
	depth_camera_project_ <<
		-1 * depth_camera_.fx, 0, depth_camera_.cx,
		0, -1 * depth_camera_.fy, depth_camera_.cy,
		0, 0, 1;
	depth_camera_reproject_ <<
		-1 / depth_camera_.fx, 0, depth_camera_.cx / depth_camera_.fx,
		0, -1 / depth_camera_.fy, depth_camera_.cy / depth_camera_.fy,
		0, 0, 1;
	//camera_extrinsic_translation_ << 0, 0, 0;
	camera_extrinsic_translation_ << -0.05192784012425894 * 1000, -0.0004530758522097678 * 1000, 0.0007057198534333861 * 1000;
	rgb_camera_project_ <<
		-1 * rgb_camera_.fx, 0, rgb_camera_.cx,
		0, -1 * rgb_camera_.fy, rgb_camera_.cy,
		0, 0, 1;
	//
	frame_count_ = 0;
	//
	ModelReader mr(M_eg_, P_eg_, delta_B1_eg_, delta_B2_eg_);
	//
	rotation_cv_ = cv::Mat(3, 1, CV_64FC1);
	translation_cv_ = cv::Mat(3, 1, CV_64FC1);
	rotation_eg_.setZero();
	translation_eg_ = Vector3d(0, 0, 500);
	//
	x_coeff_eg_.resize(exp_size, 1);
	x_coeff_eg_.setZero();
	xxx_coeff_eg_ = xx_coeff_eg_ = x_coeff_eg_;
	y_coeff_eg_.resize(pca_size, 1);
	y_coeff_eg_.setZero();
	y_weights_eg_.resize(pca_size);
	for (int i = 0; i < pca_size; i++) {
		y_weights_eg_(i) = 1.0 / P_eg_.col(i).norm();
	}
	//
	ml::MeshIOd::loadFromOBJ(Data_Input_Dir + "landmark.obj", mesh_);
	mesh_.m_Colors.resize(0);
	mesh_.m_Colors.resize(mesh_.m_Vertices.size(), ml::vec4d(0.5, 0.5, 0.5, 1.0));
	for (int i = 0; i < face_landmark.size(); i++) {
		mesh_.m_Colors[face_landmark[i]] = ml::vec4d(1.0, 0.0, 0.0, 1.0);
	}
	//
	neutral_eg_.resize(3 * vertex_size, 1);
	expression_eg_.resize(3 * vertex_size, 1);
	normal_eg_.resize(3, vertex_size);
	//
	A_track_eg_.resize(total_residual_size, 3 * vertex_size);
	C_track_eg_.resize(total_residual_size, 1);

	//
	DlibFaceDetector fd(landmark_detector_);
	std::thread tt1(fd);
	tt1.detach();
	//
	UpdateNeutralFaceCPU();
	UpdateDeltaBlendshapeCPU();
	UpdateExpressionFaceCPU();
	UpdateNormalCPU();
}

void SolvePnP()
{
	std::vector<cv::Point3d> pts3;
	std::vector<cv::Point2d> pts2;
	for (int i = 0; i < face_landmark_size; i++) {
		if (i < 17 || i >= 60 || (i >= 27 && i <= 30))
			continue;
		Vector3d pt3 = expression_eg_.block(3 * face_landmark[i], 0, 3, 1);
		pts3.push_back(cv::Point3d(pt3(0), pt3(1), pt3(2)));
		pts2.push_back(cv::Point2d(landmark_detector_.pts_[i](0), landmark_detector_.pts_[i](1)));
	}
	static double K[9] = {
		-1 * depth_camera_.fx, 0, depth_camera_.cx,
		0, -1 * depth_camera_.fy, depth_camera_.cy,
		0, 0, 1
	};
	static double D[5] = {
		0, 0, 0, 0, 0
	};
	static cv::Mat cam_matrix = cv::Mat(3, 3, CV_64FC1, K);
	static cv::Mat dist_coeffs = cv::Mat(5, 1, CV_64FC1, D);

	cv::Mat inlier;
	//cv::solvePnPRansac(pts3, pts2, cam_matrix, dist_coeffs, rotation_cv_, translation_cv_,
		//true, 100, 4.0, 0.95, inlier);
	cv::solvePnP(pts3, pts2, cam_matrix, dist_coeffs, rotation_cv_, translation_cv_);
	//for (int i = 0; i < inlier.rows; i++) {
	//	std::cout << inlier.at<int>(i, 0) << " ";
	//}
	//std::cout << "\n";
	LOG(INFO) << "inlier size" << inlier.size();

	cv::Mat rotation_mat = cv::Mat(3, 3, CV_64FC1);
	cv::Rodrigues(rotation_cv_, rotation_mat);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			rotation_eg_(i, j) = rotation_mat.at<double>(i, j);
		}
		translation_eg_(i) = translation_cv_.at<double>(i, 0);
	}
	//LOG(INFO) << "rotation:" << rotation_eg_;
	LOG(INFO) << "translation:" << Map<RowVectorXd>(translation_eg_.data(), 3);
	std::cout << "translation:" << Map<RowVectorXd>(translation_eg_.data(), 3) << "\n";
}

bool SVD()
{
	int count = 0;
	std::vector<Vector3d> ps1, ps2;

	for (int i = 0; i < face_landmark_size; i++) {
		if (i < 17 || i > 47)
			continue;
		Vector2d p2_landmark = landmark_detector_.pts_[i];
		Vector3d p3_landmark = ReprojectionDepth(p2_landmark, dframe_.at<unsigned short>(p2_landmark(1), p2_landmark(0)));
		int index = face_landmark[i];
		Vector3d p3_model = expression_eg_.block(3 * index, 0, 3, 1);
		Vector3d p3_model_now = rotation_eg_ * p3_model + translation_eg_;

		//std::cout << Map<RowVector3d>(p3_landmark.data()) << "\n" << Map<RowVector3d>(p3_model_now.data()) << "\n\n";

		//if ((p3_landmark - p3_model_now).norm() > 50)
		//	continue;
		ps1.push_back(p3_landmark);
		ps2.push_back(p3_model);
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

	//LOG(INFO) << "rotation:" << rotation_eg_;
	LOG(INFO) << "translation:" << Map<RowVectorXd>(translation_eg_.data(), 3);
	std::cout << Map<RowVectorXd>(translation_eg_.data(), 3) << "@" << count << "\n";
	//std::cout << rotation_eg_ << "\n@";

	return true;
}

void UpdateMotion()
{
	double param[6];
	Eigen2Ceres(rotation_eg_, translation_eg_, param);

	ceres::Problem problem1;
	ceres::Solver::Options options1;
	options1.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
	options1.minimizer_progress_to_stdout = false;
	options1.max_num_iterations = 500;
	options1.num_threads = 16;
	ceres::LossFunctionWrapper* loss_function_wrapper1 = new ceres::LossFunctionWrapper(new ceres::HuberLoss(1.0), ceres::TAKE_OWNERSHIP);
	CeresMotionLandmarkError::camera_extrinsic_translation = camera_extrinsic_translation_;

	for (int i = 17; i <= 47; i++) {
		problem1.AddResidualBlock(
			CeresMotionLandmarkError::Create(dframe_,
				landmark_detector_.pts_[i],
				expression_eg_.block(3 * face_landmark[i], 0, 3, 1),
				landmark_detector_.xmin, landmark_detector_.xmax, landmark_detector_.ymin, landmark_detector_.ymax),
			loss_function_wrapper1,
			param, param + 3
		);
	}

	ceres::Solver::Summary summary1;
	ceres::Solve(options1, &problem1, &summary1);

	ceres::Problem problem2;
	for (int i = 0; i < vertex_size; i += 25) {
		problem1.AddResidualBlock(
			CeresMotionDenseError::Create(dframe_,
				Vector2d(0, 0),
				expression_eg_.block(3 * i, 0, 3, 1),
				landmark_detector_.xmin, landmark_detector_.xmax, landmark_detector_.ymin, landmark_detector_.ymax),
			0,
			param, param + 3
		);
	}
	ceres::Solve(options1, &problem1, &summary1);
	//std::cout << summary2.FullReport() << "\n";

	for (int i = 0; i < vertex_size; i += 25) {
		CeresMotionDenseError error = CeresMotionDenseError(dframe_,
			Vector2d(0, 0),
			expression_eg_.block(3 * i, 0, 3, 1),
			landmark_detector_.xmin, landmark_detector_.xmax, landmark_detector_.ymin, landmark_detector_.ymax);
		double residuals;
		error(param, param + 3, &residuals);
		//std::cout << setw(15) << residuals << "\n";
		//LOG(INFO) << setw(15) << residuals;
	}

	for (int i = 17; i <= 47; i++) {
		CeresMotionLandmarkError error = CeresMotionLandmarkError(dframe_,
			landmark_detector_.pts_[i],
			expression_eg_.block(3 * face_landmark[i], 0, 3, 1),
			landmark_detector_.xmin, landmark_detector_.xmax, landmark_detector_.ymin, landmark_detector_.ymax);
		double residuals[2];
		error(param, param + 3, residuals);
		//std::cout << setw(15) << residuals[0] << " " << setw(15) << residuals[1] << "\n";
		//LOG(INFO) << setw(15) << residuals[0] << " " << setw(15) << residuals[1] << "\n";
	}

	Ceres2Eigen(rotation_eg_, translation_eg_, param);
	LOG(INFO) << "translation: " << Map<RowVectorXd>(translation_eg_.data(), 3);
	std::cout << "translation: " << Map<RowVectorXd>(translation_eg_.data(), 3) << "@\n";
}

bool UpdateFrame()
{
	static ImageReaderKinect image_reader(Kinect_Data_Dir);
	image_reader.GetFrame(frame_count_, cframe_, dframe_); 
	//LOG(INFO) << "gauss blur";
	//cv::GaussianBlur(dframe_, dframe_, cv::Size(3, 3), 0);
	landmark_detector_.Detect(cframe_, frame_count_, false);
	//return true;

	LOG(INFO) << "rigid motion";
	//SolvePnP();
	//WriteExpressionFace(frame_count_, expression_eg_, translation_eg_, rotation_eg_);
	//if (!SVD()) {
	//	WriteExpressionFace(frame_count_, expression_eg_, translation_eg_, rotation_eg_);
	//	return false;
	//}
	//WriteExpressionFace(frame_count_, expression_eg_, translation_eg_, rotation_eg_);
	UpdateMotion();
	//WriteExpressionFace(frame_count_, expression_eg_, translation_eg_, rotation_eg_);

  	return true;
}

Vector3d ReprojectionDepth(Vector2d p2, int depth)
{
	if (depth == 0)
		return Vector3d(0, 0, 0);

	return depth * depth_camera_reproject_ * Vector3d(p2.x(), p2.y(), 1);
}

Vector3d ProjectionDepth(Vector3d p3)
{
	if (p3.z() == 0)
		return Vector3d(0, 0);

	return 1.0 / p3.z() * depth_camera_project_ * p3;
}

Vector3d ProjectionRgb(Vector3d p3)
{
	if (p3.z() == 0)
		return Vector3d(0, 0);

	return 1.0 / p3.z() * rgb_camera_project_ * p3;
}

void Initialize()
{
	double param[6];
	Eigen2Ceres(rotation_eg_, translation_eg_, param);

	ceres::Problem problem1;
	ceres::Solver::Options options1;
	options1.linear_solver_type = ceres::DENSE_SCHUR;
	options1.minimizer_progress_to_stdout = false;
	options1.max_num_iterations = 25;
	options1.num_threads = 4;
	ceres::LossFunctionWrapper* loss_function_wrapper1 = new ceres::LossFunctionWrapper(new ceres::CauchyLoss(5.0), ceres::TAKE_OWNERSHIP);
	CeresLandmarkError::camera_extrinsic_translation = camera_extrinsic_translation_;
	for (int i = 17; i <= 67; i++) {
		problem1.AddResidualBlock(
			CeresLandmarkError::Create(face_landmark[i],
				dframe_,
				M_eg_, P_eg_,
				landmark_detector_.pts_[i]),
			loss_function_wrapper1, 
			param, param + 3, y_coeff_eg_.data()
		);
	}
	for (int i = 0; i < vertex_size; i += 75) {
		problem1.AddResidualBlock(
			CeresFaceDenseError::Create(i,
				dframe_,
				M_eg_, P_eg_,
				normal_eg_,
				1,
				false),
			loss_function_wrapper1,
			param, param + 3, y_coeff_eg_.data()
		);
	}
	problem1.AddResidualBlock(
		CeresInitializationRegulation::Create(y_weights_eg_),
		0, y_coeff_eg_.data()
	);
	ceres::Solver::Summary summary1;
	ceres::Solve(options1, &problem1, &summary1);

	Ceres2Eigen(rotation_eg_, translation_eg_, param);
	LOG(INFO) << "translation: " << Map<RowVectorXd>(translation_eg_.data(), 3);
	LOG(INFO) << "Y: " << Map<RowVectorXd>(y_coeff_eg_.data(), pca_size);
	std::cout << "translation: " << Map<RowVectorXd>(translation_eg_.data(), 3) << "\n";
	//std::cout << "Y1: " << Map<RowVectorXd>(y_coeff_eg_.data(), exp_size);

	//
	//ceres::Problem problem2;
	//ceres::Solver::Options options2 = options1;
	//ceres::LossFunctionWrapper* loss_function_wrapper2 = new ceres::LossFunctionWrapper(new ceres::CauchyLoss(1.0), ceres::TAKE_OWNERSHIP);
	//for (int i = 0; i < vertex_size; i += 20) {
	//	problem2.AddResidualBlock(
	//		CeresFaceDenseError::Create(i,
	//			dframe_,
	//			M_eg_, P_eg_),
	//		loss_function_wrapper2,
	//		param, param + 3, y_coeff_eg_.data()
	//	);
	//}
	//problem2.AddResidualBlock(
	//	CeresInitializationRegulation::Create(y_weights_eg_),
	//	0, y_coeff_eg_.data()
	//);
	//ceres::Solver::Summary summary2;
	//ceres::Solve(options2, &problem2, &summary2);

	//std::cout << "Y2: " << Map<RowVectorXd>(y_coeff_eg_.data(), exp_size);

	UpdateNeutralFaceCPU();
	UpdateDeltaBlendshapeCPU();
	UpdateExpressionFaceCPU();
	WriteExpressionFace(frame_count_, expression_eg_, translation_eg_, rotation_eg_);
	UpdateNormalCPU();
}

void GenerateIcpMatrix()
{
	MatrixXd C(total_residual_size, 1);
	C.setZero();
	std::vector<Tripletd> tris;
	double alpha1 = 2;
	for (int lm = 0; lm < eye_landmark_size + mouth_landmark_size; lm++) {
		int lm_index;
		if (lm < 10)
			lm_index = lm + 18 - 1;
		else if (lm < 22)
			lm_index = lm - 10 + 37 - 1;
		else
			lm_index = lm - 22 + 49 - 1;

		Vector2d p2_landmark = landmark_detector_.pts_[lm_index];
		//Vector3d p3_landmark = ReprojectionDepth(p2_landmark, dframe_.at<unsigned short>(p2_landmark(1), p2_landmark(0)));
		int index = face_landmark[lm_index];
		Vector3d p3_model = expression_eg_.block(3 * index, 0, 3, 1);
		Vector3d p3_model_now = rotation_eg_ * p3_model + translation_eg_ + camera_extrinsic_translation_;
		Vector3d n3_model = normal_eg_.col(index);

		// 2d landmark displacement
		{
			MatrixXd lhs = alpha1 * (rgb_camera_project_.topRows(2) * rotation_eg_ / p3_model_now(2));
			MatrixXd rhs = alpha1 * (p2_landmark - rgb_camera_project_.topRows(2) * (translation_eg_ + camera_extrinsic_translation_) / p3_model_now(2));
			if ((lhs * p3_model - rhs).squaredNorm() < 8 || true) {
				for (int i = 0; i < 2; i++) {
					for (int j = 0; j < 3; j++) {
						tris.push_back(Tripletd(2 * lm + i, 3 * index + j, lhs(i, j)));
					}
					C(2 * lm + i, 0) = rhs(i, 0);
				}
			}
			else {
				for (int i = 0; i < 2; i++) {
					for (int j = 0; j < 3; j++) {
						tris.push_back(Tripletd(2 * lm + i, 3 * index + j, 0));
					}
				}
			}
		}
	}

	A_track_eg_.setZero();
	A_track_eg_.setFromTriplets(tris.begin(), tris.end());
	C_track_eg_ = C;
}

void Track()
{
	//
	xxx_coeff_eg_ = xx_coeff_eg_;
	xx_coeff_eg_ = x_coeff_eg_;
	// generate A C
	GenerateIcpMatrix();
	//
	EyeTrack();
	//
	MouthTrack();
	//
	LOG(INFO) << "X: " << Map<RowVectorXd>(x_coeff_eg_.data(), exp_size);
	// output
	UpdateDeltaBlendshapeCPU();
	UpdateExpressionFaceCPU();
	std::thread t(WriteExpressionFace, frame_count_, expression_eg_, translation_eg_, rotation_eg_);
	t.detach();
	//
	UpdateNormalCPU();
}

void EyeTrack()
{
	const static double lambda1 = 5.0;
	const static double lambda2 = 20.0;
	// X
	MatrixXd X(2 * eye_landmark_size + eye_exp_size, eye_exp_size);
	X.setZero();
	X.topRows(2 * eye_landmark_size) = A_track_eg_.topRows(2 * eye_landmark_size) * delta_B_eg_.leftCols(eye_exp_size);
	// Y
	MatrixXd Y(2 * eye_landmark_size + eye_exp_size, 1);
	Y.setZero();
	Y.topRows(2 * eye_landmark_size) = C_track_eg_.topRows(2 * eye_landmark_size) - A_track_eg_.topRows(2 * eye_landmark_size) * neutral_eg_;

	for (int i = 0; i < eye_exp_size; i++) {
		X(2 * eye_landmark_size + i, i) = lambda1;
		Y(2 * eye_landmark_size + i, 0) = lambda1 * (2 * xx_coeff_eg_(i) - xxx_coeff_eg_(i));
	}
	// Beta
	VectorXd Beta, Beta_result;
	Beta = Beta_result = x_coeff_eg_.topRows(eye_exp_size);

	// (X_j) * (X_j)
	std::vector<double> Xs(eye_exp_size);
	for (int i = 0; i < eye_exp_size; i++) {
		Xs[i] = X.col(i).dot(X.col(i));
	}

	double cost = -1;
	for (int step = 0; step < 10; step++) {
#pragma omp parallel for
		for (int i = 0; i < eye_exp_size; i++) {
			double Si = -1 * X.col(i).dot(Y.col(0));
			for (int j = 0; j < eye_exp_size; j++) {
				if (i == j)
					continue;
				Si += X.col(i).dot(X.col(j)) * Beta(j);
			}
			if (Si > lambda2) {
				Beta_result(i) = (lambda2 - Si) / Xs[i];
			}
			else if (Si < -1 * lambda2) {
				Beta_result(i) = (-1 * lambda2 - Si) / Xs[i];
			}
			else {
				Beta_result(i) = 0;
			}
			Beta_result(i) = std::min(1.0, std::max(0.0, Beta_result(i)));
		}
		//
		MatrixXd res = X * Beta_result - Y;
		double new_cost = res.squaredNorm();
		printf("%f + %f = %f@eye\n",
			res.topRows(2 * eye_landmark_size).squaredNorm(),
			res.bottomRows(eye_exp_size).squaredNorm(),
			new_cost);
		if ((cost - new_cost) > 0.00001 * cost || cost < 0) {
			cost = new_cost;
			Beta = Beta_result;
		}
		else/* if(new_cost <= cost)*/ {
			break;
		}
	}
	//
	x_coeff_eg_.topRows(eye_exp_size) = Beta;
}

void MouthTrack()
{
	const static double lambda1 = 0.0;
	const static double lambda2 = 300.0;
	// X
	MatrixXd X(2 * mouth_landmark_size + mouth_exp_size, mouth_exp_size);
	X.setZero();
	X.topRows(2 * mouth_landmark_size) = A_track_eg_.bottomRows(2 * mouth_landmark_size) * delta_B_eg_.rightCols(mouth_exp_size);
	// Y
	MatrixXd Y(2 * mouth_landmark_size + mouth_exp_size, 1);
	Y.setZero();
	Y.topRows(2 * mouth_landmark_size) = C_track_eg_.bottomRows(2 * mouth_landmark_size) - A_track_eg_.bottomRows(2 * mouth_landmark_size) * neutral_eg_;

	for (int i = 0; i < mouth_exp_size; i++) {
		X(2 * mouth_landmark_size + i, i) = lambda1;
		Y(2 * mouth_landmark_size + i, 0) = lambda1 * (2 * xx_coeff_eg_(i + eye_exp_size) - xxx_coeff_eg_(i + eye_exp_size));
	}
	// Beta
	VectorXd Beta, Beta_result;
	Beta = Beta_result = x_coeff_eg_.bottomRows(mouth_exp_size);

	// (X_j) * (X_j)
	std::vector<double> Xs(mouth_exp_size);
	for (int i = 0; i < mouth_exp_size; i++) {
		Xs[i] = X.col(i).dot(X.col(i));
	}

	double cost = -1;
	for (int step = 0; step < 10; step++) {
#pragma omp parallel for
		for (int i = 0; i < mouth_exp_size; i++) {
			double Si = -1 * X.col(i).dot(Y.col(0));
			for (int j = 0; j < mouth_exp_size; j++) {
				if (i == j)
					continue;
				Si += X.col(i).dot(X.col(j)) * Beta(j);
			}
			if (Si > lambda2) {
				Beta_result(i) = (lambda2 - Si) / Xs[i];
			}
			else if (Si < -1 * lambda2) {
				Beta_result(i) = (-1 * lambda2 - Si) / Xs[i];
			}
			else {
				Beta_result(i) = 0;
			}
			Beta_result(i) = std::min(1.0, std::max(0.0, Beta_result(i)));
		}
		//
		MatrixXd res = X * Beta_result - Y;
		double new_cost = res.squaredNorm();
		printf("%f + %f = %f@mouth\n",
			res.topRows(2 * mouth_landmark_size).squaredNorm(),
			res.bottomRows(mouth_exp_size).squaredNorm(),
			new_cost);
		if ((cost - new_cost) > 0.00001 * cost || cost < 0) {
			cost = new_cost;
			Beta = Beta_result;
		}
		else/* if(new_cost <= cost)*/ {
			break;
		}
	}
	//
	x_coeff_eg_.bottomRows(mouth_exp_size) = Beta;
}

void UpdateNeutralFaceCPU()
{
	neutral_eg_ = P_eg_ * y_coeff_eg_ + M_eg_;
}

void UpdateDeltaBlendshapeCPU()
{
	LOG(INFO) << "delta blendshape cpu";
#pragma omp parallel for
	for (int i = 0; i < exp_size; i++) {
		delta_B_eg_.col(i) = Map<MatrixXd>(delta_B2_eg_.col(i).data(), 3 * vertex_size, pca_size) * y_coeff_eg_;
	}
	/*Map<MatrixXd>(delta_B_eg_.data(), 3 * vertex_size * exp_size, 1) =
	Map<MatrixXd>(delta_B2_eg_.data(), 3 * vertex_size * exp_size, pca_size) * y_coeff_eg_;*/
	delta_B_eg_ = delta_B1_eg_;
}

void UpdateExpressionFaceCPU()
{
	LOG(INFO) << "expression cpu";
	expression_eg_ = neutral_eg_ + delta_B_eg_ * x_coeff_eg_;
}

void UpdateNormalCPU()
{
	LOG(INFO) << "normal cpu";
	Map<MatrixXd> model_map(expression_eg_.data(), 3, vertex_size);
	normal_eg_.setZero();
#pragma omp parallel for
	for (int f = 0; f < (int)mesh_.m_FaceIndicesVertices.size(); f++) {
		ml::MeshDatad::Indices::Face &ind = mesh_.m_FaceIndicesVertices[f];
		Vector3d vec1 = model_map.col(ind[1]) - model_map.col(ind[0]);
		Vector3d vec2 = model_map.col(ind[2]) - model_map.col(ind[0]);
		Vector3d n = vec1.cross(vec2);
		if (n(2) > 0)
			n(2) *= -1;
		normal_eg_.col(ind[0]) += n;
		normal_eg_.col(ind[1]) += n;
		normal_eg_.col(ind[2]) += n;
	}
#pragma omp parallel for
	for (int v = 0; v < vertex_size; v++) {
		normal_eg_.col(v).normalize();
	}
}

void WriteNeutralFace(int count, MatrixXd tmesh)
{
	char str[100];
	sprintf(str, "%d/n.obj", frame_count_);

	LOG(INFO) << "write neutral face";
	//Map<MatrixXd> tmap(tmesh.data(), 3, vertex_size);
	//tmap = rotation_eg_ * tmap;
	//tmap.colwise() += translation_eg_;
	UpdateMeshVertex(tmesh, mesh_);

	ml::MeshIOd::saveToOBJ(Test_Output_Dir + str, mesh_);
}

void WriteExpressionFace(int count, MatrixXd tmesh, Vector3d translation_eg, Matrix<double, 3, 3> rotation_eg)
{
	char str[100];
	sprintf(str, "%d/e.obj", count);

	LOG(INFO) << "write expression face";
	Map<MatrixXd> tmap(tmesh.data(), 3, vertex_size);
	tmap = rotation_eg * tmap;
	tmap.colwise() += translation_eg;
	ml::MeshDatad mesh;
	mesh.m_Vertices.resize(vertex_size);
	mesh.m_Colors = mesh_.m_Colors;
	mesh.m_FaceIndicesVertices = mesh_.m_FaceIndicesVertices;
	UpdateMeshVertex(tmesh, mesh);

	ml::MeshIOd::saveToOBJ(Test_Output_Dir + str, mesh);
}

void WritePointCloud()
{
	ml::MeshDatad tmp;

	landmark_detector_.xmax = dframe_.cols;
	landmark_detector_.ymax = dframe_.rows;
	landmark_detector_.xmin = 0;
	landmark_detector_.ymin = 0;

	int width = landmark_detector_.xmax - landmark_detector_.xmin;
	int height = landmark_detector_.ymax - landmark_detector_.ymin;
	tmp.m_Vertices.resize(width * height, ml::vec3d(0, 0, 1000));
	tmp.m_Colors.resize(width * height);
	for (int i = landmark_detector_.ymin; i < landmark_detector_.ymax; i++) {
#pragma omp parallel for
		for (int j = landmark_detector_.xmin; j < landmark_detector_.xmax; j++) {
			int depth = dframe_.at<unsigned short>(i, j);
			if (depth > 2000)
				continue;
			Vector3d p3 = ReprojectionDepth(Vector2d(j, i), depth);
			tmp.m_Vertices[(i - landmark_detector_.ymin) * width + j - landmark_detector_.xmin] = ml::vec3d(p3.data());
		}
	}
	//for (int f = 17; f < face_landmark_size; f++) {
	//	Vector2d& p2 = landmark_detector_.pts_[f];
	//	tmp.m_Colors[(p2.y() - landmark_detector_.ymin) * width + p2.x() - landmark_detector_.xmin] = ml::vec4d((f + 130.0) / 200, 0, 0, 1);
	//}
	char str[20];
	sprintf(str, "%d/pcl.obj", frame_count_);
	ml::MeshIOd::saveToOBJ(Test_Output_Dir + str, tmp);
	//std::thread t(ml::MeshIOd::saveToOBJ, Test_Output_Dir + str, tmp);
	//t.detach();
}