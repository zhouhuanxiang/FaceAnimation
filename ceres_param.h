#ifndef CERES_PARAM_
#define CERES_PARAM_

#include <Eigen/Eigen>
#include <Eigen/Sparse>

void Ceres2Eigen(Eigen::Matrix3d& rotate, Eigen::Vector3d& tranlate, double* param);

void Eigen2Ceres(Eigen::Matrix3d& rotate, Eigen::Vector3d& tranlate, double* param);

#endif