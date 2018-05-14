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
 

	//cv::Mat t; 
	////t = cv::imread("C:/Users/zhx/Desktop/1.PNG");
	////t = cv::imread("C:/Users/zhx/Desktop/1.jpeg");
	//t = cv::imread("F:/Kinect2/test11/c193.png");
	//cv::Mat tt;
	//cv::pyrUp(t, tt, cv::Size(t.cols * 2, t.rows * 2));
	//cv::pyrUp(tt, t, cv::Size(tt.cols * 2, tt.rows * 2));
	//landmark_detector_.test(t);
	//cv::imwrite("C:/Users/zhx/Desktop/2.png", t);
	//return 0;

	DEM();
	for (frame_count_ = 10; frame_count_ <= 13; frame_count_++) {
		LOG(INFO) << "\n\n";
		LOG(INFO) << "frame No." << frame_count_;
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