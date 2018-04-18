#include "dem.h"

cublasHandle_t handle;
cusparseHandle_t shandle;
std::vector<cudaStream_t> streams_cu_;
CuDenseMatrix inter_result1_;
CuDenseMatrix inter_result2_;
CuDenseMatrix inter_result3_;
CuDenseMatrix inter_result4_;

MatrixXd M_eg_;
MatrixXd P_eg_;
//MatrixXd delta_B1_eg_;
//MatrixXd delta_B2_eg_;
CuDenseMatrix M_cu_;
CuDenseMatrix P_cu_;
CuDenseMatrix delta_B_cu_;
CuDenseMatrix delta_B1_cu_;
CuDenseMatrix delta_B2_cu_;

Vector3d translation_eg_;
Matrix<double, 3, 3> rotation_eg_;
cv::Mat translation_cv_;
cv::Mat rotation_cv_;

MatrixXd x_coeff_eg_;
MatrixXd xx_coeff_eg_, xxx_coeff_eg_;
MatrixXd y_coeff_eg_;
VectorXd y_weights_eg_;
CuDenseMatrix x_coeff_cu_;
CuDenseMatrix y_coeff_cu_;

ml::MeshDatad mesh_;
//MatrixXd neutral_eg_;
//MatrixXd expression_eg_;
double *p_neutral_eg_;
double *p_expression_eg_;
Map<MatrixXd> neutral_eg_(NULL, 0, 0);
Map<MatrixXd> expression_eg_(NULL, 0, 0);
MatrixXd normal_eg_;
CuDenseMatrix neutral_cu_;
CuDenseMatrix expression_cu_;

// Track
CuSparseMatrix A_track_cu_;
CuDenseMatrix C_track_cu_;
CuDenseMatrix X_track_cu_;
CuDenseMatrix Y_track_cu_;
double *p_X_refine_eg_;
double *p_Y_refine_eg_;
Map<MatrixXd> X_refine_eg_(NULL, 0, 0);
Map<MatrixXd> Y_refine_eg_(NULL, 0, 0);

// refine
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
	cublasCreate(&handle);
	cusparseCreate(&shandle);
	streams_cu_.resize(stream_size);
	for (int i = 0; i < stream_size; i++) {
		cudaStreamCreate(&(streams_cu_[i]));
	}
	cublasSetStream(handle, streams_cu_[default_stream]);
	inter_result1_.SetSize(vertex_size * 3, 1);
	inter_result2_.SetSize(vertex_size * 3, 1);
	inter_result3_.SetSize(vertex_size * 3, exp_size);
	inter_result4_.SetSize(vertex_size * 3, pca_size);
	//
	ModelReader mr(M_cu_, M_eg_,
		P_cu_, P_eg_,
		delta_B1_cu_, /*delta_B1_eg_,*/
		delta_B2_cu_/*, delta_B2_eg_*/);
	delta_B_cu_.SetSize(3 * vertex_size, exp_size);
	//
	rotation_cv_ = cv::Mat(3, 1, CV_64FC1);
	translation_cv_ = cv::Mat(3, 1, CV_64FC1);
	//
	x_coeff_eg_.resize(exp_size, 1);
	x_coeff_eg_.setZero();
	xxx_coeff_eg_ = xx_coeff_eg_ = x_coeff_eg_;
	y_coeff_eg_.resize(pca_size, 1);
	y_coeff_eg_.setZero();
	x_coeff_cu_.SetSize(exp_size, 1);
	y_coeff_cu_.SetSize(pca_size, 1);
	y_weights_eg_.resize(pca_size);
	for (int i = 0; i < pca_size; i++) {
		y_weights_eg_(i) = 1.0 / P_eg_.col(i).norm();
	}
	//
	ml::MeshIOd::loadFromOBJ(Data_Input_Dir + "landmark.obj", mesh_);
	//neutral_eg_.resize(3 * vertex_size, 1);
	//expression_eg_.resize(3 * vertex_size, 1);
	cudaMallocHost(&p_neutral_eg_, 3 * vertex_size * sizeof(double));
	cudaMallocHost(&p_expression_eg_, 3 * vertex_size * sizeof(double));
	new (&neutral_eg_) Map<MatrixXd>(p_neutral_eg_, 3 * vertex_size, 1);
	new (&expression_eg_) Map<MatrixXd>(p_expression_eg_, 3 * vertex_size, 1);
	normal_eg_.resize(3, vertex_size);
	neutral_cu_.SetSize(3 * vertex_size, 1);
	expression_cu_.SetSize(3 * vertex_size, 1);
	//
	cudaMallocHost(&p_X_refine_eg_, 6 * face_landmark.size() * exp_size * sizeof(double));
	cudaMallocHost(&p_Y_refine_eg_, 6 * face_landmark.size() * sizeof(double));
	new (&X_refine_eg_) Map<MatrixXd>(p_X_refine_eg_, 6 * face_landmark.size(), exp_size);
	new (&Y_refine_eg_) Map<MatrixXd>(p_Y_refine_eg_, 6 * face_landmark.size(), 1);
	C_track_cu_.SetSize(6 * face_landmark.size(), 1);
	X_track_cu_.SetSize(6 * face_landmark.size(), exp_size);
	Y_track_cu_.SetSize(6 * face_landmark.size(), 1);

	//
	//int eindex = 2;
	//int nindex = 1;
	//MatrixXd b2(3 * vertex_size, exp_size);
	//for (int i = 0; i < exp_size; i++) {
	//	b2.col(i) = 1 * delta_B2_eg_.block(nindex * vertex_size * 3, i, 3 * vertex_size, 1);
	//}
	//MatrixXd delta_B_eg = delta_B1_eg_ + b2;
	//expression_eg_ = M_eg_ + P_eg_.col(nindex) + delta_B_eg.col(eindex);
	//rotation_eg_.setIdentity();
	//translation_eg_.setZero();
	//WriteExpressionFace();
	//system("pause");
	// create thread for face detect and refine
	// then detach
	DlibFaceDetector fd(landmark_detector_);
	std::thread tt(fd);
	tt.detach();
}

void SolvePnP()
{
	std::vector<cv::Point3d> pts3;
	std::vector<cv::Point2d> pts2;
	for (int i = 0; i < face_landmark.size(); i++) {
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

void SVD()
{
	int count = 0;
	std::vector<Vector3d> ps1, ps2;

	for (int i = 0; i < face_landmark.size(); i++) {
		if (i < 17 || i >= 60)
			continue;
		Vector2d p2_landmark = landmark_detector_.pts_[i];
		Vector3d p3_landmark = Point2d_2_Point3d(p2_landmark, dframe_.at<unsigned short>(p2_landmark(1), p2_landmark(0)));
		int index = face_landmark[i];
		Vector3d p3_model = expression_eg_.block(3 * index, 0, 3, 1);
		Vector3d p3_model_now = rotation_eg_ * p3_model + translation_eg_;
		if ((p3_landmark - p3_model_now).norm() > 20)
			continue;
		ps1.push_back(p3_landmark);
		ps2.push_back(p3_model);
		count++;
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
}

void GetFrame(bool init)
{
	static ImageReaderKinect image_reader(Kinect_Data_Dir);
	image_reader.GetFrame(frame_count_, cframe_, dframe_);
	LOG(INFO) << "gauss blur";	
	cv::GaussianBlur(dframe_, dframe_, cv::Size(3, 3), 0);
	landmark_detector_.Detect(cframe_, frame_count_);
	if (!init)
		//SolvePnP();
		SVD();

	/*ml::MeshDatad tmp;
	tmp.m_Vertices.resize(face_detector_.width * face_detector_.height);
	tmp.m_Colors.resize(face_detector_.width * face_detector_.height);
	for (int i = face_detector_.ymin; i < face_detector_.ymax; i++) {
		for (int j = face_detector_.xmin; j < face_detector_.xmax; j++) {
			int depth = dframe_.at<unsigned short>(i, j);
			if (depth > 2000)
				continue;
			Vector3d p3 = Point2d_2_Point3d(Vector2d(j, i), depth);
			tmp.m_Vertices[(i - face_detector_.ymin) * face_detector_.width + j - face_detector_.xmin] = ml::vec3d(p3.data());
		}
	}
	for (int f = 0; f < face_landmark.size(); f++) {
		if (f < 17)
			continue;
		Vector2d& p2 = face_detector_.pts_[f];
		tmp.m_Colors[(p2.y() - face_detector_.ymin) * face_detector_.width + p2.x() - face_detector_.xmin] = ml::vec4d(1.0, 0, 0, 1);
	}
	char str[20];
	sprintf(str, "%d/pcl.obj", frame_count_);
	ml::MeshIOd::saveToOBJ(Test_Output_Dir + str, tmp);*/
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

	for (int i = 0; i < face_landmark.size(); i++) {

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
	Eigen::SparseMatrix<double, Eigen::RowMajor> A(6 * face_landmark.size(), 3 * vertex_size);
	MatrixXd C(6 * face_landmark.size(), 1);
	A.setZero();
	C.setZero();

	double alpha1 = 4;
	double alpha2 = 0;
	double alpha3 = 1;
	for (int lm = 0; lm < face_landmark.size(); lm++) {
		if (lm < 17 /*|| lm >= 60*/)
			continue;
		Vector2d p2_landmark = landmark_detector_.pts_[lm];
		Vector3d p3_landmark = Point2d_2_Point3d(p2_landmark, dframe_.at<unsigned short>(p2_landmark(1), p2_landmark(0)));
		int index = face_landmark[lm];
		Vector3d p3_model = expression_eg_.block(3 * index, 0, 3, 1);
		p3_model = rotation_eg_ * p3_model + translation_eg_;
		Vector3d n3_model = normal_eg_.col(index);

		if ((p3_landmark - p3_model).norm() > 50){
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
			MatrixXd lhs = alpha1 * (projection * rotation_eg_);
			MatrixXd rhs = alpha1 * (p2_landmark - projection * translation_eg_);
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
			MatrixXd lhs = alpha2 * n3_model.transpose() * rotation_eg_;
			double rhs = alpha2 * n3_model.dot(p3_landmark - translation_eg_);
			for (int j = 0; j < 3; j++) {
				tris.push_back(Tripletd(6 * lm + 2, 3 * index + j, lhs(0, j)));
			}
			C(6 * lm + 2, 0) = rhs;
		}
		// point-to-point
		{
			MatrixXd lhs = alpha3 * rotation_eg_;
			MatrixXd rhs = alpha3 * (p3_landmark - translation_eg_);
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					tris.push_back(Tripletd(6 * lm + 3 + i, 3 * index + j, lhs(i, j)));
				}
				C(6 * lm + 3 + i) = rhs(i, 0);
			}
		}
	}

	A.setFromTriplets(tris.begin(), tris.end());

	//MatrixXd result = A * expression_eg_ - C;
	//for (int i = 0; i < face_landmark.size(); i++) {
	//	for (int j = 0; j < 6; j++) {
	//		printf("%10.5f  ", result(i * 6 + j));
	//	}
	//	std::cout << "\n";
	//}
	//std::cout << "\n";

	if (A_track_cu_.entries)
		A_track_cu_.SetData(A);
	else
		A_track_cu_.SetMatrix(A);
	C_track_cu_.SetData(C.data());
}

void Track()
{
	const static double lambda1 = 150.0;
	const static double lambda2 = 80.0;
	//
	xxx_coeff_eg_ = xx_coeff_eg_;
	xx_coeff_eg_ = x_coeff_eg_;
	// generate A C
	GenerateIcpMatrix();
	// X
	MatrixXd X(6 * face_landmark.size() + exp_size, exp_size);
	X.setZero();
	SMmulDM(shandle, A_track_cu_, delta_B_cu_, X_track_cu_);
	X_track_cu_.GetMatrix(6 * face_landmark.size(), exp_size, X_refine_eg_.data(), streams_cu_[default_stream]);
	X.block(0, 0, 6 * face_landmark.size(), exp_size) = X_refine_eg_;
	// Y
	MatrixXd Y(6 * face_landmark.size() + exp_size, 1);
	Y.setZero();
	DM2DM(handle, C_track_cu_, Y_track_cu_);
	SMmulDV(shandle, A_track_cu_, neutral_cu_, Y_track_cu_, -1, 1);
	Y_track_cu_.GetMatrix(6 * face_landmark.size(), 1, Y_refine_eg_.data(), streams_cu_[default_stream]);
	Y.block(0, 0, 6 * face_landmark.size(), 1) = Y_refine_eg_;

	for (int i = 0; i < exp_size; i++) {
		X(6 * face_landmark.size() + i, i) = lambda1;
		Y(6 * face_landmark.size() + i, 0) = lambda1 * (2 * xx_coeff_eg_(i) - xxx_coeff_eg_(i));
	}
	// Beta
	VectorXd Beta, Beta_result;
	Beta = Beta_result = x_coeff_eg_;
	// (X_j) * (X_j)
	std::vector<double> Xs(exp_size);
	for (int i = 0; i < exp_size; i++) {
		Xs[i] = X.col(i).dot(X.col(i));
	}

	double cost = DBL_MAX;
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
			res.block(0, 0, 6 * face_landmark.size(), 1).norm(),
			res.block(6 * face_landmark.size(), 0, exp_size, 1).norm(),
			new_cost);
		if ((cost - new_cost) > 0.00001 * cost) {
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
	// update
	x_coeff_cu_.SetData(x_coeff_eg_.data());
	UpdateDeltaBlendshapeGPU();
	UpdateExpressionFaceGPU();
	UpdateExpressionFaceCPU();
	// output
	WriteExpressionFace();
	//
	UpdateNormalCPU();
}

// y updated
void UpdateNeutralFaceGPU()
{
	LOG(INFO) << "neutral face gpu";
	DMmulDV(handle, P_cu_, y_coeff_cu_, neutral_cu_); // update y_coeff_cu_ ?
	DVaddDV(handle, M_cu_, neutral_cu_);
}

void UpdateNeutralFaceCPU()
{
	LOG(INFO) << "neutral face cpu";
	neutral_cu_.GetMatrix(3 * vertex_size, 1, neutral_eg_.data(), streams_cu_[default_stream]);
}

// y updated
void UpdateDeltaBlendshapeGPU()
{
	LOG(INFO) << "delta blendshape gpu";
	double al = 1.0;
	double bet = 0.0;

	//cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
	//cublasDgemmStridedBatched(handle,
	//	CUBLAS_OP_N,
	//	CUBLAS_OP_N,
	//	3 * vertex_size, 1, pca_size,
	//	&al,
	//	delta_B2_cu_.d_Val, 3 * vertex_size, 3 * vertex_size * pca_size,
	//	y_coeff_cu_.d_Val, pca_size, 0,
	//	&bet,
	//	inter_result3_.d_Val, 3 * vertex_size, 3 * vertex_size,
	//	exp_size);
	//cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

	// read y_coeff_cu_ first

	cudaStreamSynchronize(streams_cu_[default_stream]);
	for (int i = 0; i < exp_size; i++) {
		cublasSetStream(handle, streams_cu_[track_stream_begin + i]);
		cublasDgemv(handle,
			CUBLAS_OP_N,
			3 * vertex_size, pca_size,
			&al,
			delta_B2_cu_.d_Val + i * 3 * vertex_size * pca_size, 3 * vertex_size,
			y_coeff_cu_.d_Val, 1,
			&bet,
			inter_result3_.d_Val + i * 3 * vertex_size, 1);
	}
	for (int i = 0; i < exp_size; i++) {
		cudaStreamSynchronize(streams_cu_[track_stream_begin + i]);
	}
	cublasSetStream(handle, streams_cu_[default_stream]);

	DMaddDM(handle, delta_B1_cu_, inter_result3_, delta_B_cu_);
	cudaStreamSynchronize(streams_cu_[default_stream]); // for test
}
// x & y updated
void UpdateExpressionFaceGPU()
{
	LOG(INFO) << "expression gpu";
	DMmulDV(handle, delta_B_cu_, x_coeff_cu_, expression_cu_);
	DVaddDV(handle, neutral_cu_, expression_cu_);
}

void UpdateExpressionFaceCPU()
{
	LOG(INFO) << "expression cpu";
	expression_cu_.GetMatrix(3 * vertex_size, 1, expression_eg_.data(), streams_cu_[default_stream]);
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
	Map<MatrixXd> tmap(tmesh.data(), 3, vertex_size);
	tmap = rotation_eg_ * tmap;
	tmap.colwise() += translation_eg_;
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

DemRefine::DemRefine()
	:S_re_(0), S_total_re_(0),
	X_eg_(NULL, 0, 0), Y_eg_(NULL, 0, 0),
	updated(false)
{

	cudaMallocHost(&p_X_eg_, 6 * face_landmark.size() * pca_size * sizeof(double));
	cudaMallocHost(&p_Y_eg_, 6 * face_landmark.size() * sizeof(double));
	new (&X_eg_) Map<MatrixXd>(p_X_eg_, 6 * face_landmark.size(), pca_size);
	new (&Y_eg_) Map<MatrixXd>(p_Y_eg_, 6 * face_landmark.size(), 1);
	X_eg_.setZero();
	Y_eg_.setZero();
	X_re_.SetMatrix(6 * face_landmark.size(), pca_size, X_eg_.data());
	Y_re_.SetMatrix(6 * face_landmark.size(), 1, Y_eg_.data());
	// in
	x_in_.resize(exp_size, 1);
	C_in_.SetSize(6 * face_landmark.size(), 1);
	// out
	y_coeff_re_.resize(pca_size, 1);
	// run time variable
	x_coeff_re_.resize(exp_size, 1);
	C_re_.SetSize(6 * face_landmark.size(), 1);
	A_hat_cu_.SetSize(6 * face_landmark.size(), pca_size);
	C_hat_cu_.SetSize(6 * face_landmark.size(), 1);
}

void DemRefine::operator()()
{
	while (true) {
		while (!updated) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			continue;
		}
		//
		cublasSetStream(handle, streams_cu_[refine_default_stream]);
		GetX(streams_cu_[refine_default_stream]);
		// update A_hat C_hat
		CuDenseMatrix tA(3 * vertex_size, pca_size);
		CuDenseMatrix tC(3 * vertex_size, 1);
		tA.SetZero(handle);
		tC.SetZero(handle);
		for (int i = 0; i < exp_size; i++) {

		}
		// update X_total Y_total
		double gamma = 0.9;
		double S_total_re_ = gamma * S_re_ + 1;
		double al = 1.0 / S_total_re_;
		double bet = gamma * S_re_ / S_total_re_;
		cublasDgemm(handle,
			CUBLAS_OP_T, CUBLAS_OP_N,
			pca_size, pca_size, 6 * face_landmark.size(),
			&al,
			A_hat_cu_.d_Val, A_hat_cu_.rows,
			A_hat_cu_.d_Val, A_hat_cu_.rows,
			&bet,
			X_re_.d_Val, X_re_.rows);
		cublasDgemv(handle,
			CUBLAS_OP_T,
			6 * face_landmark.size(), pca_size,
			&al,
			A_hat_cu_.d_Val, A_hat_cu_.rows,
			C_hat_cu_.d_Val, 1,
			&bet,
			Y_re_.d_Val, 1);
		S_re_ = S_total_re_;
		// read [X Y] --> CPU
		X_re_.GetMatrix(pca_size, pca_size, X_eg_.data(), streams_cu_[refine_default_stream]);
		Y_re_.GetMatrix(pca_size, 1, Y_eg_.data(), streams_cu_[refine_default_stream]);

		// solve it
		MatrixXd result1, result2;
		result1 = result2 = y_coeff_eg_;

		MatrixXd tmp = X_eg_;
		MatrixXd D = tmp.diagonal().asDiagonal().toDenseMatrix();
		MatrixXd L = tmp.triangularView<Eigen::StrictlyLower>().toDenseMatrix();
		MatrixXd U = tmp.triangularView<Eigen::StrictlyUpper>().toDenseMatrix();

		LLT<MatrixXd> llt;
		llt.compute(D + U);
		double cost = DBL_MAX;
		for (int i = 0; i < 10; i++) {
			result1 = llt.solve(Y_eg_ - L * result2);
			double new_cost = (tmp * result1 - Y_eg_).norm();
			if ((cost - new_cost) > 0.00001 * cost) {
				result2 = result1;
				cost = new_cost;
			}
			else
				break;
		}
		std::cout << Map<RowVectorXd>(result2.data(), pca_size);
		UpdateY(result2);
	}
}

// call outside
void DemRefine::GetY(CuDenseMatrix &dm, cudaStream_t &stream)
{
	y_mtx_.lock();
	dm.SetData(y_coeff_re_.data());
	cudaStreamSynchronize(stream);
	y_mtx_.unlock();
}

// call inside
void DemRefine::UpdateY(MatrixXd &result)
{
	y_mtx_.lock();
	y_coeff_re_ = result;
	y_mtx_.unlock();
}

// call inside
void DemRefine::GetX(cudaStream_t &stream)
{
	x_mtx_.lock();
	SM2SM(handle, A_in_, A_re_);
	DM2DM(handle, C_in_, C_re_);
	x_coeff_re_ = x_in_;
	cudaStreamSynchronize(stream);
	x_mtx_.unlock();
}

// call outside
void DemRefine::UpdateX(MatrixXd &x, CuSparseMatrix A_in, CuDenseMatrix C_in, cudaStream_t &stream)
{
	x_mtx_.lock();
	SM2SM(handle, A_in, A_in_);
	DM2DM(handle, C_in, C_in_);
	x_in_ = x;
	cudaStreamSynchronize(stream);
	x_mtx_.unlock();
}
