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

	//for (int i = 0; i < 500; i++) {
	//	char str[200];
	//	sprintf(str, "%d", i);
	//	mkdir((Test_Output_Dir + str).c_str());
	//}
	//system("pause");

	//ModelReader mr;
	//mr.ConcatMatrix();
	//system("pause");

	DEM();
	for (frame_count_ = 40; frame_count_ <= 43; frame_count_++) {
		LOG(INFO) << "\n\n";
		LOG(INFO) << "frame No." << frame_count_;
		GetFrame(true);
		Initialize();
	}

	{
		y_coeff_cu_.SetData(y_coeff_eg_.data());
		x_coeff_cu_.SetData(x_coeff_eg_.data());
		UpdateNeutralFaceGPU();
		UpdateDeltaBlendshapeGPU();
		UpdateExpressionFaceGPU();
		// output
		UpdateExpressionFaceCPU();
		WriteExpressionFace();
	}

	for (frame_count_ = 44; frame_count_ <= 380; frame_count_++) {
		LOG(INFO) << "\n\n";
		LOG(INFO) << "frame No." << frame_count_;
		GetFrame(false);
		Track();
		//Refine();
	}

	system("pause");
	return 0;
}