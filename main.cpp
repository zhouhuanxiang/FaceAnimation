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

	DEM();

	for (int i = 0; i < 500; i++) {
		UpdateFrame();
		WritePointCloud();
	}
	return 0;

	for (frame_count_ = 10; frame_count_ <= 13; frame_count_++) {
		LOG(INFO) << "\n\n";
		LOG(INFO) << "frame No." << frame_count_;
		std::cout << "# " << frame_count_ << "\n";
		UpdateFrame();
		Initialize();
	}

	for (frame_count_ = 14; frame_count_ <= 10000;) {
		LOG(INFO) << "\n\n";
		LOG(INFO) << "frame No." << frame_count_;
		std::cout << "# " << frame_count_ << "\n";
		UpdateFrame();
		Track();
		//Refine();

		frame_count_ += 1;
	}

	system("pause");
	return 0;
}