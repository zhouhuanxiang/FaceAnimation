#ifndef MODEL_READER_
#define MODEL_READER_

#include "parameters.h"
#include "eigen_binary_io.h"

#include <Eigen/Core>
#include <Eigen/Sparse>
using namespace Eigen;

#include <vector>
#include <string>

class ModelReader
{
public:
	ModelReader(){}

	ModelReader(MatrixXd &M_eg_, MatrixXd &P_eg_, MatrixXd &delta_B1_eg_, MatrixXd &delta_B2_eg_)
	{
		MatrixXd pca;
		read_binary(Data_Input_Dir + "pca50", pca);

		M_eg_ = pca.col(0);
		P_eg_ = pca.block(0, 1, 3 * vertex_size, pca_size);
#pragma omp parallel for
		for (int i = 0; i < pca_size; i++) {
			P_eg_.col(i) -= M_eg_;
		}

		read_binary(Data_Input_Dir + "delta_B1_min", delta_B1_eg_);
		read_binary(Data_Input_Dir + "delta_B2_min", delta_B2_eg_);

		////std::cout << delta_B2_eg_.rows() << " " << delta_B1_eg_.cols() << "\n";
		////std::cout << "\n";
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
		for (int i = 0; i < exp_size; i++) {
			int index;
			if (i < eye_exp_size)
				index = eye_expression[i];
			else
				index = mouth_expression[i - eye_exp_size];
			//std::cout << index << "\n";

			MatrixXd b1_i, b2_i;
			char str[20];
			sprintf(str, "e/m%d", index);
			read_binary(Data_Input_Dir + str, b1_i);
			b1_i = b1_i - pca.col(0);
			sprintf(str, "e/Py%d", index);
			read_binary(Data_Input_Dir + str, b2_i);
			b2_i = b2_i - pca.block(0, 1, 3 * vertex_size, pca_size);

			if (b1_i.rows() != 3 * vertex_size ||
				b2_i.rows() != 3 * vertex_size ||
				b1_i.cols() != 1 ||
				b2_i.cols() != pca_size)
				printf("wrong size!\n");
			delta_B1.col(count) = b1_i;
			for (int j = 0; j < pca_size; j++) {
				delta_B2.block(j * 3 * vertex_size, count, 3 * vertex_size, 1) = b2_i.col(j);
			}
			count++;
		}

		//std::cout << "\n";
		
		write_binary(Data_Input_Dir + "delta_B1_min", delta_B1);
		write_binary(Data_Input_Dir + "delta_B2_min", delta_B2);
	}
};


#endif