#include "dem.h"

#include <chrono>

long long track_time_;
long long track_time1_;
long long track_time2_;
long long track_time3_;
long long solve_time1_;
long long solve_time2_;
long long solve_time3_;

MatrixXd M_eg_;
MatrixXd P_eg_;
MatrixXd delta_B1_eg_;
MatrixXd delta_B2_eg_;
MatrixXd delta_B_eg_;

int motion_param_ptr;
bool motion_param_updated;
double motion_param[motion_param_size][6];
double motion_param_tmp[6];
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
Matrix<double, 3, 3> rgb_camera_reproject_;
//Matrix<double, 3, 3> rgb_camera_reproject_;

DlibLandmarkDetector landmark_detector_;

void DEM()
{
	track_time_ = 0;
	track_time1_ = 0;
	track_time2_ = 0;
	track_time3_ = 0;
	solve_time1_ = 0;
	solve_time2_ = 0;
	solve_time3_ = 0;
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
	camera_extrinsic_translation_ << -52, 0, 0;
	rgb_camera_project_ <<
		-1 * rgb_camera_.fx, 0, rgb_camera_.cx,
		0, -1 * rgb_camera_.fy, rgb_camera_.cy,
		0, 0, 1;
	rgb_camera_reproject_ <<
		-1 / rgb_camera_.fx, 0, rgb_camera_.cx / rgb_camera_.fx,
		0, -1 / rgb_camera_.fy, rgb_camera_.cy / rgb_camera_.fy,
		0, 0, 1;
	//
	frame_count_ = 0;
	//
	ModelReader mr(M_eg_, P_eg_, delta_B1_eg_, delta_B2_eg_);
	delta_B_eg_.resize(3 * vertex_size, exp_size);
	//
	rotation_cv_ = cv::Mat(3, 1, CV_64FC1);
	translation_cv_ = cv::Mat(3, 1, CV_64FC1);
	rotation_eg_.setZero();
	translation_eg_ = Vector3d(0, 0, 500);
	motion_param_updated = 0;
	motion_param_ptr = 1;
	for (int i = 0; i < motion_param_size; i++) {
		Eigen2Ceres(rotation_eg_, translation_eg_, motion_param[i]);
	}
	Eigen2Ceres(rotation_eg_, translation_eg_, motion_param_tmp);
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
	std::thread t(fd);
	t.detach();
	//
	UpdateNeutralFaceCPU();
	UpdateDeltaBlendshapeCPU();
	UpdateExpressionFaceCPU();
	UpdateNormalCPU();
}

void FitMotion()
{
	//LOG(INFO) << "fit motion";

	ceres::Problem problem1;
	ceres::Solver::Options options1;
	options1.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
	options1.minimizer_progress_to_stdout = false;
	options1.max_num_iterations = 500;
	options1.num_threads = 16;
	//ceres::LossFunctionWrapper* loss_function_wrapper1 = new ceres::LossFunctionWrapper(new ceres::HuberLoss(1.0), ceres::TAKE_OWNERSHIP);

	int data_size = 5;
	double coeffs[3 * 6];
	for (int i = 0; i < data_size; i++) {
		int prev_ptr = (motion_param_ptr + motion_param_size + i - data_size) % motion_param_size;
		for (int j = 0; j < 6; j++) {
			problem1.AddResidualBlock(CeresMotionFitError::Create(i, motion_param[prev_ptr][j]),
				0,
				&(coeffs[3 * j])
			);
		}
	}
	//LOG(INFO) << "start solve";
	ceres::Solver::Summary summary1;
	ceres::Solve(options1, &problem1, &summary1);
	for (int i = 0; i < 6; i++) {
		motion_param[motion_param_ptr][i] = coeffs[3 * i] + coeffs[3 * i + 1] * data_size + coeffs[3 * i + 2] * data_size * data_size;
	}

	Ceres2Eigen(rotation_eg_, translation_eg_, motion_param[motion_param_ptr]);
	//LOG(INFO) << "translation: " << Map<RowVectorXd>(translation_eg_.data(), 3);
	////std::cout << "translation: " << Map<RowVectorXd>(translation_eg_.data(), 3) << "@fit\n";
	for (int i = 0; i < 6; i++) {
		//std::cout << motion_param[motion_param_ptr][i] << " ";
	}
	//std::cout << "@fit\n";
	for (int i = 0; i < 3; i++) {
		//std::cout << coeffs[i] << " ";
	}
	//std::cout << "@fit\n";
}

void Initialize()
{
	ceres::Problem problem1;
	ceres::Solver::Options options1;
	options1.linear_solver_type = ceres::DENSE_SCHUR;
	options1.minimizer_progress_to_stdout = false;
	options1.max_num_iterations = 25;
	options1.num_threads = 1;
	ceres::LossFunctionWrapper* loss_function_wrapper1 = new ceres::LossFunctionWrapper(new ceres::CauchyLoss(5.0), ceres::TAKE_OWNERSHIP);
	CeresLandmarkError::camera_extrinsic_translation = camera_extrinsic_translation_;
	for (int i = 17; i <= 67; i++) {
		problem1.AddResidualBlock(
			CeresLandmarkError::Create(face_landmark[i],
				dframe_,
				M_eg_, P_eg_,
				landmark_detector_.pts_[i]),
			loss_function_wrapper1, 
			motion_param[motion_param_ptr], motion_param[motion_param_ptr] + 3, y_coeff_eg_.data()
		);
	}
	for (int i = 0; i < vertex_size; i += 25) {
		problem1.AddResidualBlock(
			CeresFaceDenseError::Create(i,
				dframe_,
				M_eg_, P_eg_,
				normal_eg_,
				0.05,
				true),
			loss_function_wrapper1,
			motion_param[motion_param_ptr], motion_param[motion_param_ptr] + 3, y_coeff_eg_.data()
		);
	}
	problem1.AddResidualBlock(
		CeresInitializationRegulation::Create(y_weights_eg_),
		0, y_coeff_eg_.data()
	);
	ceres::Solver::Summary summary1;
	ceres::Solve(options1, &problem1, &summary1);

	Ceres2Eigen(rotation_eg_, translation_eg_, motion_param[motion_param_ptr]);
	//LOG(INFO) << "translation: " << Map<RowVectorXd>(translation_eg_.data(), 3);
	//LOG(INFO) << "Y: " << Map<RowVectorXd>(y_coeff_eg_.data(), pca_size);
	//std::cout << "translation: " << Map<RowVectorXd>(translation_eg_.data(), 3) << "\n";
	////std::cout << "Y1: " << Map<RowVectorXd>(y_coeff_eg_.data(), exp_size);

	//for (int i = 17; i <= 67; i++) {
	//	CeresLandmarkError error = CeresLandmarkError(face_landmark[i],
	//		dframe_,
	//		M_eg_, P_eg_,
	//		landmark_detector_.pts_[i]);
	//	double residuals[3];
	//	error(motion_param[motion_param_ptr], motion_param[motion_param_ptr] + 3, y_coeff_eg_.data(), residuals);
	//	//std::cout << setw(15) << residuals[0] << " " << setw(15) << residuals[1] << setw(15) << residuals[2] << "\n";
	//}

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

void TrackCeres()
{
	//LOG(INFO) << "track start";

	xxx_coeff_eg_ = xx_coeff_eg_;
	xx_coeff_eg_ = x_coeff_eg_;

	ceres::Problem problem1;
	ceres::Solver::Options options1;
	options1.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
	options1.minimizer_progress_to_stdout = false;
	options1.max_num_iterations = 25;
	options1.num_threads = 1;
	ceres::LossFunctionWrapper* loss_function_wrapper1 = new ceres::LossFunctionWrapper(new ceres::CauchyLoss(5.0), ceres::TAKE_OWNERSHIP);
	CeresTrackLandmarkError::camera_extrinsic_translation = camera_extrinsic_translation_;

	for (int i = 36; i <= 47; i++) {
		int index = face_landmark[i];
		problem1.AddResidualBlock(
			CeresTrackLandmarkError::Create(dframe_,
				landmark_detector_.pts_[i],
				neutral_eg_.block(3 * index, 0, 3, 1),
				delta_B_eg_.block(3 * index, 0, 3, exp_size),
				motion_param[motion_param_ptr]),
			0,
			x_coeff_eg_.data()
		);
	}
	for (int i = 48; i <= 67; i++) {
		int index = face_landmark[i];
		problem1.AddResidualBlock(
			CeresTrackLandmarkError::Create(dframe_,
				landmark_detector_.pts_[i],
				neutral_eg_.block(3 * index, 0, 3, 1),
				delta_B_eg_.block(3 * index, 0, 3, exp_size),
				motion_param[motion_param_ptr]),
			0,
			x_coeff_eg_.data()
		);
	}

	problem1.AddResidualBlock(
		CeresTrackRegulation::Create(xx_coeff_eg_, xxx_coeff_eg_),
		0,
		x_coeff_eg_.data()
	);

	//LOG(INFO) << "solve problem";
	ceres::Solver::Summary summary1;
	ceres::Solve(options1, &problem1, &summary1);

	//
	//LOG(INFO) << "X: " << Map<RowVectorXd>(x_coeff_eg_.data(), exp_size);
	// output
	UpdateExpressionFaceCPU();
	std::thread t(WriteExpressionFace, frame_count_, expression_eg_, translation_eg_, rotation_eg_);
	t.detach();
	//
	//UpdateNormalCPU();
}

void UpdateNeutralFaceCPU()
{
	neutral_eg_ = P_eg_ * y_coeff_eg_ + M_eg_;
}

void UpdateDeltaBlendshapeCPU()
{
	//LOG(INFO) << "delta blendshape cpu";
#pragma omp parallel for
	for (int i = 0; i < exp_size; i++) {
		delta_B_eg_.col(i) = Map<MatrixXd>(delta_B2_eg_.col(i).data(), 3 * vertex_size, pca_size) * y_coeff_eg_;
	}
	/*Map<MatrixXd>(delta_B_eg_.data(), 3 * vertex_size * exp_size, 1) =
	Map<MatrixXd>(delta_B2_eg_.data(), 3 * vertex_size * exp_size, pca_size) * y_coeff_eg_;*/
	delta_B_eg_ += delta_B1_eg_;
}

void UpdateExpressionFaceCPU()
{
	//LOG(INFO) << "expression cpu";
	expression_eg_ = neutral_eg_;
	expression_eg_.noalias() += delta_B_eg_ * x_coeff_eg_;
}

void UpdateNormalCPU()
{
	//LOG(INFO) << "normal cpu";
	Map<MatrixXd> model_map(expression_eg_.data(), 3, vertex_size);
	normal_eg_.setZero();
#pragma omp parallel for
	for (int f = 0; f < (int)mesh_.m_FaceIndicesVertices.size(); f++) {
		ml::MeshDatad::Indices::Face &ind = mesh_.m_FaceIndicesVertices[f];
		Vector3d vec1 = model_map.col(ind[1]) - model_map.col(ind[0]);
		Vector3d vec2 = model_map.col(ind[2]) - model_map.col(ind[0]);
		Vector3d n = vec2.cross(vec1);
		if (n(2) < 0)
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

	//LOG(INFO) << "write neutral face";
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

	//LOG(INFO) << "write expression face";
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

Vector3d ReprojectionDepth(Vector2d p2, int depth)
{
	if (depth == 0)
		return Vector3d(0, 0, 0);

	return depth * depth_camera_reproject_ * Vector3d(p2.x(), p2.y(), 1);
}

Vector3d ReprojectionRgb(Vector2d p2, int depth)
{
	if (depth == 0)
		return Vector3d(0, 0, 0);

	return depth * rgb_camera_reproject_ * Vector3d(p2.x(), p2.y(), 1);
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
	//	//std::cout << inlier.at<int>(i, 0) << " ";
	//}
	////std::cout << "\n";
	//LOG(INFO) << "inlier size" << inlier.size();

	cv::Mat rotation_mat = cv::Mat(3, 3, CV_64FC1);
	cv::Rodrigues(rotation_cv_, rotation_mat);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			rotation_eg_(i, j) = rotation_mat.at<double>(i, j);
		}
		translation_eg_(i) = translation_cv_.at<double>(i, 0);
	}
	////LOG(INFO) << "rotation:" << rotation_eg_;
	//LOG(INFO) << "translation:" << Map<RowVectorXd>(translation_eg_.data(), 3);
	//std::cout << "translation:" << Map<RowVectorXd>(translation_eg_.data(), 3) << "\n";
}

void Track()
{
	//LOG(INFO) << "track start";
	//
	xxx_coeff_eg_ = xx_coeff_eg_;
	xx_coeff_eg_ = x_coeff_eg_;
	// generate A C
	std::chrono::steady_clock::time_point tp1 = std::chrono::steady_clock::now();
	GenerateIcpMatrix();
	//
	std::chrono::steady_clock::time_point tp2 = std::chrono::steady_clock::now();
	EyeTrack();
	MouthTrack();
	//EyeMouthTrack();
	//
	//LOG(INFO) << "X: " << Map<RowVectorXd>(x_coeff_eg_.data(), exp_size);
	// output
	std::chrono::steady_clock::time_point tp3 = std::chrono::steady_clock::now();
	UpdateExpressionFaceCPU();
	std::thread t(WriteExpressionFace, frame_count_, expression_eg_, translation_eg_, rotation_eg_);
	t.detach();
	//
	//UpdateNormalCPU();
	std::chrono::steady_clock::time_point tp4 = std::chrono::steady_clock::now();
	track_time1_ += std::chrono::duration_cast<std::chrono::milliseconds>(tp2 - tp1).count();
	track_time2_ += std::chrono::duration_cast<std::chrono::milliseconds>(tp3 - tp2).count();
	track_time3_ += std::chrono::duration_cast<std::chrono::milliseconds>(tp4 - tp3).count();
}

void EyeMouthTrack()
{
	std::chrono::steady_clock::time_point tp1 = std::chrono::steady_clock::now();
	const static double e_lambda1 = 10.0;
	const static double e_lambda2 = 20.0;
	const static double m_lambda1 = 400.0;
	const static double m_lambda2 = 400.0;
	// X
	MatrixXd X(total_residual_size, exp_size);
	X.setZero();
	X.topRows(total_residual_size) = A_track_eg_ * delta_B_eg_.leftCols(exp_size);
	// Y
	MatrixXd Y(total_residual_size + exp_size, 1);
	Y.setZero();
	Y.topRows(total_residual_size) = C_track_eg_;
	Y.topRows(total_residual_size) -= A_track_eg_ * neutral_eg_;

	for (int i = 0; i < exp_size; i++) {
		//double lambda1 = (i < eye_exp_size) ? e_lambda1 : m_lambda1;
		//double lambda1 = 50;
		X(total_residual_size + i, i) = e_lambda2;
		Y(total_residual_size + i, 0) = e_lambda2 * (2 * xx_coeff_eg_(i) - xxx_coeff_eg_(i));
	}
	// Beta
	VectorXd Beta, Beta_result;
	Beta = Beta_result = x_coeff_eg_;

	// (X_j) * (X_j)
	std::vector<double> Xs(exp_size);
	for (int i = 0; i < exp_size; i++) {
		Xs[i] = X.col(i).dot(X.col(i));
	}
	std::chrono::steady_clock::time_point tp2 = std::chrono::steady_clock::now();
	double cost = -1;
	for (int step = 0; step < 10; step++) {
#pragma omp parallel for
		for (int i = 0; i < exp_size; i++) {
			double Si = -1 * X.col(i).dot(Y.col(0));
			double lambda2 = (i < eye_exp_size) ? e_lambda2 : m_lambda2;
			for (int j = 0; j < exp_size; j++) {
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
			res.topRows(total_residual_size).squaredNorm(),
			res.bottomRows(exp_size).squaredNorm(),
			new_cost);
		if ((cost - new_cost) > 1 || cost < 0) {
			cost = new_cost;
			Beta = Beta_result;
		}
		else/* if(new_cost <= cost)*/ {
			break;
		}
	}
	//
	x_coeff_eg_ = Beta;
	std::chrono::steady_clock::time_point tp3 = std::chrono::steady_clock::now();
	solve_time1_ += std::chrono::duration_cast<std::chrono::milliseconds>(tp2 - tp1).count();
	solve_time2_ += std::chrono::duration_cast<std::chrono::milliseconds>(tp3 - tp2).count();
}

void EyeTrack()
{
	std::chrono::steady_clock::time_point tp1 = std::chrono::steady_clock::now();
	const static double lambda1 = 15.0;
	const static double lambda2 = 25.0;
	// X
	MatrixXd X(2 * eye_landmark_size + eye_exp_size, eye_exp_size);
	X.setZero();
	X.topRows(2 * eye_landmark_size) = A_track_eg_.topRows(2 * eye_landmark_size) * delta_B_eg_.leftCols(eye_exp_size);
	// Y
	MatrixXd Y(2 * eye_landmark_size + eye_exp_size, 1);
	Y.setZero();
	Y.topRows(2 * eye_landmark_size) = C_track_eg_.topRows(2 * eye_landmark_size);
	Y.topRows(2 * eye_landmark_size) -= A_track_eg_.topRows(2 * eye_landmark_size) * neutral_eg_;

	for (int i = 0; i < eye_exp_size; i++) {
		X(2 * eye_landmark_size + i, i) = lambda1;
		Y(2 * eye_landmark_size + i, 0) = lambda1 * (2 * xx_coeff_eg_(i) - xxx_coeff_eg_(i));
	}
	// Beta
	VectorXd Beta, Beta_result;
	Beta = Beta_result = x_coeff_eg_.topRows(eye_exp_size);

	MatrixXd Ys = Y.transpose() * X;
	MatrixXd XXs = X.transpose() * X;
	std::chrono::steady_clock::time_point tp2 = std::chrono::steady_clock::now();
	double cost = -1;
	for (int step = 0; step < 500; step++) {
		for (int i = 0; i < eye_exp_size; i++) {
			double Si = -1 * Ys(0, i);
//#pragma omp parallel for
			for (int j = 0; j < eye_exp_size; j++) {
				if (i == j)
					continue;
				Si += XXs(i,j) * Beta(j);
			}
			if (Si > lambda2) {
				Beta_result(i) = (lambda2 - Si) / XXs(i, i);
			}
			else if (Si < -1 * lambda2) {
				Beta_result(i) = (-1 * lambda2 - Si) / XXs(i, i);
			}
			else {
				Beta_result(i) = 0;
			}
			Beta_result(i) = std::min(1.0, std::max(0.0, Beta_result(i)));
		}
		//
		//if ((Beta_result - Beta).norm() < 0.1)
		//	break;
		//else {
		//	Beta = Beta_result;
		//}
		//
		MatrixXd res = X * Beta_result - Y;
		double cost1 = res.topRows(2 * eye_landmark_size).squaredNorm();
		double cost2 = res.bottomRows(eye_exp_size).squaredNorm();
		double new_cost = cost1 + cost2;
		printf("%f + %f = %f@eye\n",
			cost1,
			cost2,
			new_cost);
		if ((cost - new_cost) > 1 || cost < 0) {
			cost = new_cost;
			Beta = Beta_result;
		}
		else {
			break;
		}
	}
	//
	x_coeff_eg_.topRows(eye_exp_size) = Beta;
	std::chrono::steady_clock::time_point tp3 = std::chrono::steady_clock::now();
	solve_time1_ += std::chrono::duration_cast<std::chrono::milliseconds>(tp2 - tp1).count();
	solve_time2_ += std::chrono::duration_cast<std::chrono::milliseconds>(tp3 - tp2).count();
}

void MouthTrack()
{
	std::chrono::steady_clock::time_point tp1 = std::chrono::steady_clock::now();
	const static double lambda1 = 400.0;
	const static double lambda2 = 500.0;
	// X
	MatrixXd X(2 * mouth_landmark_size + mouth_exp_size, mouth_exp_size);
	X.setZero();
	X.topRows(2 * mouth_landmark_size) = A_track_eg_.bottomRows(2 * mouth_landmark_size) * delta_B_eg_.rightCols(mouth_exp_size);
	// Y
	MatrixXd Y(2 * mouth_landmark_size + mouth_exp_size, 1);
	Y.setZero();
	Y.topRows(2 * mouth_landmark_size) = C_track_eg_.bottomRows(2 * mouth_landmark_size);
	Y.topRows(2 * mouth_landmark_size) -= A_track_eg_.bottomRows(2 * mouth_landmark_size) * neutral_eg_;

	for (int i = 0; i < mouth_exp_size; i++) {
		X(2 * mouth_landmark_size + i, i) = lambda1;
		Y(2 * mouth_landmark_size + i, 0) = lambda1 * (2 * xx_coeff_eg_(i + eye_exp_size) - xxx_coeff_eg_(i + eye_exp_size));
	}
	// Beta
	VectorXd Beta, Beta_result;
	Beta = Beta_result = x_coeff_eg_.bottomRows(mouth_exp_size);

	MatrixXd Ys = Y.transpose() * X;
	MatrixXd XXs = X.transpose() * X;
	std::chrono::steady_clock::time_point tp2 = std::chrono::steady_clock::now();
	double cost = -1;
	for (int step = 0; step < 500; step++) {
		for (int i = 0; i < eye_exp_size; i++) {
			double Si = -1 * Ys(0, i);
//#pragma omp parallel for
			for (int j = 0; j < eye_exp_size; j++) {
				if (i == j)
					continue;
				Si += XXs(i, j) * Beta(j);
			}
			if (Si > lambda2) {
				Beta_result(i) = (lambda2 - Si) / XXs(i, i);
			}
			else if (Si < -1 * lambda2) {
				Beta_result(i) = (-1 * lambda2 - Si) / XXs(i, i);
			}
			else {
				Beta_result(i) = 0;
			}
			Beta_result(i) = std::min(1.0, std::max(0.0, Beta_result(i)));
		}
		//
		//if ((Beta_result - Beta).norm() < 0.1)
		//	break;
		//else {
		//	Beta = Beta_result;
		//}
		//
		MatrixXd res = X * Beta_result - Y;
		double cost1 = res.topRows(2 * eye_landmark_size).squaredNorm();
		double cost2 = res.bottomRows(eye_exp_size).squaredNorm();
		double new_cost = cost1 + cost2;
		printf("%f + %f = %f@mouth\n",
			cost1,
			cost2,
			new_cost);
		if ((cost - new_cost) > 1 || cost < 0) {
			cost = new_cost;
			Beta = Beta_result;
		}
		else {
			break;
		}
	}
	//
	x_coeff_eg_.bottomRows(mouth_exp_size) = Beta;
	std::chrono::steady_clock::time_point tp3 = std::chrono::steady_clock::now();
	solve_time1_ += std::chrono::duration_cast<std::chrono::milliseconds>(tp2 - tp1).count();
	solve_time2_ += std::chrono::duration_cast<std::chrono::milliseconds>(tp3 - tp2).count();
}