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
MatrixXd X_refine_eg_;
MatrixXd Y_refine_eg_;

// refine
//DemRefineMiddleWare middleware_;
//DemRefine dem_refine_;

int frame_count_;
cv::Mat dframe_;
cv::Mat cframe_;
Camera camera_;
DlibLandmarkDetector landmark_detector_;

void DEM()
{
	frame_count_ = 0;
	//
	ModelReader mr(M_eg_, P_eg_, delta_B1_eg_, delta_B2_eg_);
	//
	rotation_cv_ = cv::Mat(3, 1, CV_64FC1);
	translation_cv_ = cv::Mat(3, 1, CV_64FC1);
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
	for (int i = 0; i < dense_residual_pair; i++) {
		mesh_.m_Colors[i * 10] = ml::vec4d(0.0, 1.0, 0.0, 1.0);
	}
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
	X_refine_eg_.resize(total_residual_size, exp_size);
	Y_refine_eg_.resize(total_residual_size, 1);

	//
	DlibFaceDetector fd(landmark_detector_);
	std::thread tt1(fd);
	tt1.detach();
}

void SolvePnP()
{
	std::vector<cv::Point3d> pts3;
	std::vector<cv::Point2d> pts2;
	for (int i = 0; i < face_landmark_size; i++) {
		if (i < 17 || i >= 60)
			continue;
		Vector3d pt3 = expression_eg_.block(3 * face_landmark[i], 0, 3, 1);
		pts3.push_back(cv::Point3d(pt3(0), pt3(1), pt3(2)));
		pts2.push_back(cv::Point2d(landmark_detector_.pts_[i](0), landmark_detector_.pts_[i](1)));
	}
	static double K[9] = {
		-1 * camera_.fx, 0, camera_.cx,
		0, -1 * camera_.fy, camera_.cy,
		0, 0, 1
	};
	static double D[5] = {
		0, 0, 0, 0, 0
	};
	static cv::Mat cam_matrix = cv::Mat(3, 3, CV_64FC1, K);
	static cv::Mat dist_coeffs = cv::Mat(5, 1, CV_64FC1, D);

	cv::Mat inlier;
	//cv::solvePnPRansac(pts3, pts2, cam_matrix, dist_coeffs, rotation_cv_, translation_cv_,
	//	true, 100, 4.0, 0.95, inlier);
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
}

bool SVD()
{
	int count = 0;
	std::vector<Vector3d> ps1, ps2;

	for (int i = 0; i < face_landmark_size; i++) {
		if (i < 17 || i >= 60)
			continue;
		Vector2d p2_landmark = landmark_detector_.pts_[i];
		Vector3d p3_landmark = Point2d_2_Point3d(p2_landmark, dframe_.at<unsigned short>(p2_landmark(1), p2_landmark(0)));
		int index = face_landmark[i];
		Vector3d p3_model = expression_eg_.block(3 * index, 0, 3, 1);
		Vector3d p3_model_now = rotation_eg_ * p3_model + translation_eg_;

		//std::cout << Map<RowVector3d>(p3_landmark.data()) << "\n" << Map<RowVector3d>(p3_model_now.data()) << "\n\n";

		if ((p3_landmark - p3_model_now).norm() > 50)
			continue;
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

bool UpdateFrame(bool init)
{
	static ImageReaderKinect image_reader(Kinect_Data_Dir);
	image_reader.GetFrame(frame_count_, cframe_, dframe_);
	LOG(INFO) << "gauss blur";
	cv::GaussianBlur(dframe_, dframe_, cv::Size(3, 3), 0);
	landmark_detector_.Detect(cframe_, frame_count_, false);

	//face_detector_();

	if (!init)
		//SolvePnP();
		if (!SVD()) {
			WriteExpressionFace();
			return false;
		}

	return true;
}

Vector3d Point2d_2_Point3d(Vector2d p2, int depth)
{
	if (depth == 0)
		return Vector3d(0, 0, 0);
	Vector3d p3;

	p3(2) = depth;
	p3(0) = -1 * (p2(0) - camera_.cx) * p3(2) / camera_.fx;
	p3(1) = -1 * (p2(1) - camera_.cy) * p3(2) / camera_.fy;

	return p3;
}

Vector2d Point3d_2_Point2d(Vector3d p3)
{
	if (p3(2) == 0)
		return Vector2d(0, 0);
	Vector2d p2;
	p2(0) = p3(0) / p3(2) * (-1) * camera_.fx + camera_.cx;
	p2(1) = p3(1) / p3(2) * (-1) * camera_.fx + camera_.cx;

	return p2;
}

void Initialize()
{
	double param[6];
	Eigen2Ceres(rotation_eg_, translation_eg_, param);

	ceres::Problem problem;
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = false;
	options.max_num_iterations = 20;
	options.num_threads = 4;
	//ceres::LossFunctionWrapper* loss_function_wrapper = new ceres::LossFunctionWrapper(new ceres::HuberLoss(1.0), ceres::TAKE_OWNERSHIP);

	for (int i = 0; i < face_landmark_size; i++) {

		Vector3d p3_landmark = Point2d_2_Point3d(landmark_detector_.pts_[i],
			dframe_.at<unsigned short>(landmark_detector_.pts_[i](1), landmark_detector_.pts_[i](0)));

		problem.AddResidualBlock(
			CeresLandmarkError::Create(face_landmark[i], i,
				dframe_,
				M_eg_, P_eg_,
				camera_.fx, camera_.fy, camera_.cx, camera_.cy,
				p3_landmark),
			0, param, param + 3, y_coeff_eg_.data()
		);
	}

	problem.AddResidualBlock(
		CeresInitializationRegulation::Create(y_weights_eg_),
		0, y_coeff_eg_.data()
	);

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//std::cout << summary.FullReport();
	//system("pause");

	Ceres2Eigen(rotation_eg_, translation_eg_, param);

	//LOG(INFO) << "rotation: " << rotation_eg_;
	LOG(INFO) << "translation: " << Map<RowVectorXd>(translation_eg_.data(), 3);
	LOG(INFO) << "Y: " << Map<RowVectorXd>(y_coeff_eg_.data(), pca_size);
}

void GenerateIcpMatrix()
{
	std::vector<Tripletd> tris;
	SparseMatrix<double> A(total_residual_size, 3 * vertex_size);
	MatrixXd C(total_residual_size, 1);
	A.setZero();
	C.setZero();

	double alpha1 = 2;
	double alpha2 = 0;
	double alpha3 = 1;
	for (int lm = 0; lm < face_landmark_size; lm++) {
		if (lm < 17 /*|| lm >= 60*/)
			continue;
		Vector2d p2_landmark = landmark_detector_.pts_[lm];
		Vector3d p3_landmark = Point2d_2_Point3d(p2_landmark, dframe_.at<unsigned short>(p2_landmark(1), p2_landmark(0)));
		int index = face_landmark[lm];
		Vector3d p3_model = expression_eg_.block(3 * index, 0, 3, 1);
		p3_model = rotation_eg_ * p3_model + translation_eg_;
		Vector3d n3_model = normal_eg_.col(index);

		double alpha4;
		if (lm >= 36 && lm <= 45)
			alpha4 = 12;
		else
			alpha4 = 1;

		if ((p3_landmark - p3_model).norm() > 50) {
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 3; j++) {
					tris.push_back(Tripletd(6 * lm + i, 3 * index + j, 0));
				}
			}
			for (int j = 0; j < 3; j++) {
				tris.push_back(Tripletd(6 * lm + 2, 3 * index + j, 0));
			}
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					tris.push_back(Tripletd(6 * lm + 3 + i, 3 * index + j, 0));
				}
			}
		}

		// 2d landmark displacement
		{
			MatrixXd projection(2, 3);
			projection.setZero();
			projection(0, 0) = -1 * camera_.fx;
			projection(0, 2) = camera_.cx;
			projection(1, 1) = -1 * camera_.fy;
			projection(1, 2) = camera_.cy;
			projection /= p3_model(2);
			MatrixXd lhs = alpha4 * alpha1 * (projection * rotation_eg_);
			MatrixXd rhs = alpha4 * alpha1 * (p2_landmark - projection * translation_eg_);
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 3; j++) {
					tris.push_back(Tripletd(6 * lm + i, 3 * index + j, lhs(i, j)));
				}
				C(6 * lm + i, 0) = rhs(i, 0);
			}
		}
		// 3d landmark displacement
		// point-to-plane
		{
			MatrixXd lhs = /*alpha4 **/ alpha2 * n3_model.transpose() * rotation_eg_;
			double rhs = /*alpha4 **/ alpha2 * n3_model.dot(p3_landmark - translation_eg_);
			for (int j = 0; j < 3; j++) {
				tris.push_back(Tripletd(6 * lm + 2, 3 * index + j, lhs(0, j)));
			}
			C(6 * lm + 2, 0) = rhs;
		}
		// point-to-point
		{
			MatrixXd lhs = /*alpha4 **/ alpha3 * rotation_eg_;
			MatrixXd rhs = /*alpha4 **/ alpha3 * (p3_landmark - translation_eg_);
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					tris.push_back(Tripletd(6 * lm + 3 + i, 3 * index + j, lhs(i, j)));
				}
				C(6 * lm + 3 + i) = rhs(i, 0);
			}
		}
	}

	for (int r = 0; r < 0; r++) {
		int index = 5 * r;
		Vector3d p3_model = expression_eg_.block(3 * index, 0, 3, 1);
		p3_model = rotation_eg_ * p3_model + translation_eg_;
		Vector2d p2_model = Point3d_2_Point2d(p3_model);
		double x = p2_model(0);
		double y = p2_model(1);
		int px = (int)x;
		int py = (int)y;
		double wx = x - px;
		double wy = y - py;
		int rx = px + 1, ry = py + 1;
		if (!(px > 0 && py > 0 && rx < dframe_.cols - 1 && ry < dframe_.rows - 1)) {
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					tris.push_back(Tripletd(6 * face_landmark_size + 3 * r + i, 3 * index + j, 0));
				}
			}
			continue;
		}
		double d = dframe_.at<unsigned short>(py, px) * (1 - wy) * (1 - wx)
			+ dframe_.at<unsigned short>(py, rx) * (1 - wy) * wx
			+ dframe_.at<unsigned short>(ry, px) * wy * (1 - wx)
			+ dframe_.at<unsigned short>(ry, rx) * wy * wx;
		if (std::abs(d - p3_model(2)) > 10) {
			//std::cout << d << " " << p3_model(2) << "@big\n";
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					tris.push_back(Tripletd(6 * face_landmark_size + 3 * r + i, 3 * index + j, 0));
				}
			}
			continue;
		}
		else {
			//std::cout << d << " " << p3_model(2) << "@small\n";
			Vector3d closest_point = d / p3_model(2) * p3_model;
			MatrixXd lhs = alpha3 * rotation_eg_;
			MatrixXd rhs = alpha3 * (closest_point - translation_eg_);
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					tris.push_back(Tripletd(6 * face_landmark_size + 3 * r + i, 3 * index + j, lhs(i, j)));
				}
				C(6 * face_landmark_size + 3 * r + i) = rhs(i, 0);
			}
		}
	}

	A_track_eg_.setZero();
	A_track_eg_.setFromTriplets(tris.begin(), tris.end());
	C_track_eg_ = C;
}

void Track()
{
	const static double lambda1 = 200.0;
	const static double lambda2 = 400.0;
	//
	xxx_coeff_eg_ = xx_coeff_eg_;
	xx_coeff_eg_ = x_coeff_eg_;
	// generate A C
	GenerateIcpMatrix();
	// X
	MatrixXd X(total_residual_size + exp_size, exp_size);
	X.setZero();
	X_refine_eg_ = A_track_eg_ * delta_B_eg_;
	X.block(0, 0, total_residual_size, exp_size) = X_refine_eg_;
	// Y
	MatrixXd Y(total_residual_size + exp_size, 1);
	Y.setZero();
	Y_refine_eg_ = C_track_eg_ - A_track_eg_ * neutral_eg_;
	Y.block(0, 0, total_residual_size, 1) = Y_refine_eg_;

	for (int i = 0; i < exp_size; i++) {
		X(total_residual_size + i, i) = lambda1;
		Y(total_residual_size + i, 0) = lambda1 * (2 * xx_coeff_eg_(i) - xxx_coeff_eg_(i));
	}
	// Beta
	VectorXd Beta, Beta_result;
	Beta = Beta_result = x_coeff_eg_;
	// (X_j) * (X_j)
	std::vector<double> Xs(exp_size);
	for (int i = 0; i < exp_size; i++) {
		Xs[i] = X.col(i).dot(X.col(i));
	}

	double cost = -1;
	for (int step = 0; step < 10; step++) {
#pragma omp parallel for
		for (int i = 0; i < exp_size; i++) {
			double Si = -1 * X.col(i).dot(Y.col(0));
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
		double new_cost = res.norm();
		printf("%f + %f = %f\n",
			res.block(0, 0, total_residual_size, 1).norm(),
			res.block(total_residual_size, 0, exp_size, 1).norm(),
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
	x_coeff_eg_ = Beta;
	LOG(INFO) << "X: " << Map<RowVectorXd>(x_coeff_eg_.data(), exp_size);

	// output
	UpdateDeltaBlendshapeCPU();
	UpdateExpressionFaceCPU();
	WriteExpressionFace();
	WritePointCloud();
	//
	UpdateNormalCPU();
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
		normal_eg_.col(ind[0]) += n;
		normal_eg_.col(ind[1]) += n;
		normal_eg_.col(ind[2]) += n;
	}
#pragma omp parallel for
	for (int v = 0; v < vertex_size; v++) {
		normal_eg_.col(v).normalize();
	}
}

void WriteNeutralFace()
{
	char str[100];
	sprintf(str, "%d/n.obj", frame_count_);
	MatrixXd tmesh = neutral_eg_;

	LOG(INFO) << "write neutral face";
	//Map<MatrixXd> tmap(tmesh.data(), 3, vertex_size);
	//tmap = rotation_eg_ * tmap;
	//tmap.colwise() += translation_eg_;
	UpdateMeshVertex(tmesh, mesh_);

	ml::MeshIOd::saveToOBJ(Test_Output_Dir + str, mesh_);
}

void WriteExpressionFace()
{
	char str[100];
	sprintf(str, "%d/e.obj", frame_count_);
	MatrixXd tmesh = expression_eg_;

	LOG(INFO) << "write expression face";
	Map<MatrixXd> tmap(tmesh.data(), 3, vertex_size);
	tmap = rotation_eg_ * tmap;
	tmap.colwise() += translation_eg_;
	UpdateMeshVertex(tmesh, mesh_);

	ml::MeshIOd::saveToOBJ(Test_Output_Dir + str, mesh_);
}

void WritePointCloud()
{
	ml::MeshDatad tmp;
	int width = landmark_detector_.xmax - landmark_detector_.xmin;
	int height = landmark_detector_.ymax - landmark_detector_.ymin;
	tmp.m_Vertices.resize(width * height);
	tmp.m_Colors.resize(width * height);
	for (int i = landmark_detector_.ymin; i < landmark_detector_.ymax; i++) {
	for (int j = landmark_detector_.xmin; j < landmark_detector_.xmax; j++) {
	int depth = dframe_.at<unsigned short>(i, j);
	if (depth > 2000)
	continue;
	Vector3d p3 = Point2d_2_Point3d(Vector2d(j, i), depth);
	tmp.m_Vertices[(i - landmark_detector_.ymin) * width + j - landmark_detector_.xmin] = ml::vec3d(p3.data());
	}
	}
	for (int f = 0; f < face_landmark_size; f++) {
	if (f < 17)
	continue;
	Vector2d& p2 = landmark_detector_.pts_[f];
	tmp.m_Colors[(p2.y() - landmark_detector_.ymin) * width + p2.x() - landmark_detector_.xmin] = ml::vec4d(1.0, 0, 0, 1);
	}
	char str[20];
	sprintf(str, "%d/pcl.obj", frame_count_);
	ml::MeshIOd::saveToOBJ(Test_Output_Dir + str, tmp);
}