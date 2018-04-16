#ifndef DEM_REFINE_H_
#define DEM_REFINE_H_

#include "dem.h"

#include <Eigen/Core>
using namespace Eigen;

class DemRefine
{
public:
	DemRefine()
		:X_eg_(NULL, 0, 0), Y_eg_(NULL, 0, 0)
	{
		cudaMallocHost(&p_X_eg_, 6 * face_landmark.size() * pca_size * sizeof(double));
		cudaMallocHost(&p_Y_eg_, 6 * face_landmark.size() * sizeof(double));
		new (&X_eg_) Map<MatrixXd>(p_X_eg_, 6 * face_landmark.size(), pca_size);
		new (&Y_eg_) Map<MatrixXd>(p_Y_eg_, 6 * face_landmark.size(), 1);

		y_coeff_re_.SetSize(pca_size, 1);
		C_coeff_re_.SetSize(6 * face_landmark.size(), 1);
		X_re_.SetSize(6 * face_landmark.size(), pca_size);
		Y_re_.SetSize(6 * face_landmark.size(), 1);
	}

	void Refine()
	{

	}

public:
	CuDenseMatrix y_coeff_re_;
	CuSparseMatrix A_coeff_re_;
	CuDenseMatrix C_coeff_re_;
	CuDenseMatrix X_re_;
	CuDenseMatrix Y_re_;
	double *p_X_eg_;
	double *p_Y_eg_;
	Map<MatrixXd> X_eg_;
	Map<MatrixXd> Y_eg_;
};

#endif // !DEM_REFINE_H_
