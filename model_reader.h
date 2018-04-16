#ifndef MODEL_READER_
#define MODEL_READER_

#include "parameters.h"
#include "eigen_binary_io.h"
#include "cuda/cuda_helper_func.cuh"

#include <Eigen/Core>
#include <Eigen/Sparse>
using namespace Eigen;

#include <vector>
#include <string>

class ModelReader
{
public:
	ModelReader(){}

	ModelReader(CuDenseMatrix &M_cu_, MatrixXd &M_eg_,
		CuDenseMatrix &P_cu_, MatrixXd &P_eg_,
		CuDenseMatrix &delta_B1_cu_, MatrixXd &delta_B1_eg_,
		CuDenseMatrix &delta_B2_cu_, MatrixXd &delta_B2_eg_)
	{
		MatrixXd pca;
		read_binary(Data_Input_Dir + "pca50", pca);

		M_eg_ = pca.col(0);
		P_eg_ = pca.block(0, 1, 3 * vertex_size, pca_size);
		for (int i = 0; i < pca_size; i++) {
			P_eg_.col(i) -= M_eg_;
		}
		M_cu_.SetMatrix(M_eg_.rows(), M_eg_.cols(), M_eg_.data());
		P_cu_.SetMatrix(P_eg_.rows(), P_eg_.cols(), P_eg_.data());

		//MatrixXd delta_B1_eg_, delta_B2_eg_;
		read_binary(Data_Input_Dir + "delta_B1_min", delta_B1_eg_);
		read_binary(Data_Input_Dir + "delta_B2_min", delta_B2_eg_);
		delta_B1_cu_.SetMatrix(delta_B1_eg_.rows(), delta_B1_eg_.cols(), delta_B1_eg_.data());
		delta_B2_cu_.SetMatrix(delta_B2_eg_.rows(), delta_B2_eg_.cols(), delta_B2_eg_.data());
	}

	void ConcatMatrix()
	{
		MatrixXd delta_B1(3 * vertex_size, exp_size);
		MatrixXd delta_B2(3 * vertex_size * pca_size, exp_size);

		MatrixXd pca;
		read_binary(Data_Input_Dir + "pca50", pca);
		for (int i = 0; i < pca_size; i++) {
			pca.col(i + 1) -= pca.col(0);
		}

		int count = 0;
		for (int i = 0; i < exp_size + useless_expression.size(); i++) {
			if (useless_expression.find(i) != useless_expression.end())
				continue;
			MatrixXd b1_i, b2_i;
			char str[20];
			sprintf(str, "expression/m%d", i);
			read_binary(Data_Input_Dir + str, b1_i);
			b1_i = b1_i - pca.col(0);
			sprintf(str, "expression/Py%d", i);
			read_binary(Data_Input_Dir + str, b2_i);
			b2_i = b2_i - pca.block(0, 1, 3 * vertex_size, pca_size);

			if (b1_i.rows() != 3 * vertex_size ||
				b2_i.rows() != 3 * vertex_size ||
				b1_i.cols() != 1 ||
				b2_i.cols() != pca_size)
				printf("wrong size!\n");
			/*Map<MatrixXd> map_b1_i(b1_i.data(), 3 * vertex_size * 1, 1);
			Map<MatrixXd> map_b2_i(b2_i.data(), 3 * vertex_size * pca_size, 1);
			delta_B1.col(i) = map_b1_i;
			delta_B2.col(i) = map_b2_i;*/
			delta_B1.col(count) = b1_i;
			for (int j = 0; j < pca_size; j++) {
				delta_B2.block(j * 3 * vertex_size, count, 3 * vertex_size, 1) = b2_i.col(j);
			}
			count++;
		}

		write_binary(Data_Input_Dir + "delta_B1_min", delta_B1);
		write_binary(Data_Input_Dir + "delta_B2_min", delta_B2);
	}
};


#endif