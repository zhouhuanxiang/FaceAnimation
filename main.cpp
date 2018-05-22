#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>

#include "dem.h"

#include "direct.h"

int main(int argc, char** argv)
{
	FLAGS_logtostderr = false;
	google::SetLogDestination(google::GLOG_INFO, "C:/Users/zhx/Desktop/log/log");
	google::SetLogDestination(google::GLOG_WARNING, "C:/Users/zhx/Desktop/log/warning");
	google::InitGoogleLogging(argv[0]);

	//for (int i = 0; i < 1000; i++) {
	//	char str[200];
	//	sprintf(str, "%d", i);
	//	mkdir((Test_Output_Dir + str).c_str());
	//}
	//system("pause");

	//ModelReader mr;
	//mr.ConcatMatrix();
	//system("pause");

	//ceres::Problem problem1, problem2;
	//ceres::Solver::Options options1;
	//options1.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
	//options1.minimizer_progress_to_stdout = false;
	//options1.max_num_iterations = 500;
	//options1.num_threads = 16;
	//ceres::LossFunctionWrapper* loss_function_wrapper1 = new ceres::LossFunctionWrapper(new ceres::HuberLoss(0.2), ceres::TAKE_OWNERSHIP);
	//double a = 11.37;
	//double b = -0.25;
	//double c = 1.255;
	//double coeffs[3];
	//for (int i = 0; i < 3; i++) {
	//	double error = (i % 2 == 0)? 1 : -1;
	//	error *= 0.001;
	//	double x = i * 0.2;
	//	problem1.AddResidualBlock(CeresMotionFitError::Create(x, a + b * x + c * x * x + error),
	//		loss_function_wrapper1,
	//		coeffs
	//	);
	//}
	//ceres::Solver::Summary summary1, summary2;
	//ceres::Solve(options1, &problem1, &summary1);
	//double x = 3 * 0.2;
	//std::cout << coeffs[0] << " " << coeffs[1] << " " << coeffs[2] << "\n";
	//std::cout << coeffs[0] + coeffs[1] * x + coeffs[2] * x * x - a - b * x - c * x * x << "\n";

	DEM();

	//for (frame_count_ = 25; frame_count_ < 370; frame_count_++) {
	//	UpdateFrame();
	//}

	//for (frame_count_ = 80; frame_count_ < 500; frame_count_++) {
	//	std::cout << frame_count_ << "\n";
	//	UpdateFrame();
	//	WritePointCloud();
	//}
	//return 0;

	int base = 25;
	for (frame_count_ = base; frame_count_ <= base + 5; frame_count_++) {
		LOG(INFO) << "\n\nframe No." << frame_count_;
		std::cout << "# " << frame_count_ << "\n";
		UpdateFrame(true);
		Initialize();
	}

	for (frame_count_ = base + 6; frame_count_ < frame_count_end;) {
		LOG(INFO) << "\n\nframe No." << frame_count_;
		std::cout << "# " << frame_count_ << "\n";
		UpdateFrame(false);
		Track();
		//TrackCeres();
		//Refine();
		frame_count_ += 1;
	}

	return 0;
}