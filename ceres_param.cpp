#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>

#include "ceres_param.h"

void Ceres2Eigen(Eigen::Matrix3d& rotate, Eigen::Vector3d& tranlate, double* param)
{
	double rotation_R[9];
	ceres::AngleAxisToRotationMatrix(param, rotation_R);
	//Eigen::Matrix4d t_extrinsic;
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			//Note: ceres-rotationMatrix is the transpose of opencv Matrix, or ml Matrix
			rotate(i, j) = rotation_R[j * 3 + i];
		}
		tranlate(i) = param[3 + i];
	}
}

void Eigen2Ceres(Eigen::Matrix3d& rotate, Eigen::Vector3d& tranlate, double* param)
{
	double rotation_R[9];
	ceres::AngleAxisToRotationMatrix(param, rotation_R);
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			//Note: ceres-rotationMatrix is the transpose of opencv Matrix, or ml Matrix
			rotation_R[j * 3 + i] = rotate(i, j);
		}
		param[3 + i] = tranlate(i);
	}
	ceres::RotationMatrixToAngleAxis(rotation_R, param);
}