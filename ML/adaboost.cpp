/*
* Copyright (C) 2017 China.ShangHai, zhiyeyin@gmail.com
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
* 
* http://www.apache.org/licenses/LICENSE-2.0
* 
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "stdafx.h"
#include "MachineLearning.h"
#include "Utils.h"
#include "xFile.h"



void stumpClassify(Mat& data, int dimen, double threshVal, int  ineq, Mat& ret)
{
	ret = Mat::ones(data.rows, 1, data.type());
	for(int j = 0; j < data.rows; j++)
	{
		if(ineq == 0) {
			if(data.at<double>(j, dimen) <= threshVal)
				ret.at<double>(j, 0) = -1;
		} else if (ineq == 1) {
			if(data.at<double>(j, dimen) > threshVal)
				ret.at<double>(j, 0) = -1;
		}
	}
}

typedef struct BestStump {
	int dim;
	double thresh;
	int ineq;
	
} BestStump;
void buildStump(Mat& data, Mat& label, Mat& D, Mat& bestEst, BestStump& bs)
{
	int m = data.rows;
	int n = data.cols;
	label = label.t();
	bestEst = Mat::zeros(m, 1, data.type());
	double numSteps = 10.0;
	double minError = DBL_MAX;
	for(int j = 0; j < n; j++)
	{
		double minVal, maxVal;
		minMaxLoc(data.col(j), &minVal, &maxVal);
		double stepSize = (maxVal-minVal)/numSteps;
		for(int i = -1; i < numSteps+1; i++)
		{
			for(int ineq = 0; ineq < 2; ineq++) {
				double threshVal = minVal + i*stepSize;
				Mat predict;
				stumpClassify(data, j, threshVal,ineq, predict);
				Mat errorArr = Mat::ones(m, 1, data.type());
				for(int ii = 0; ii < m; ii++) {
					if(label.at<double>(ii, 0) == predict.at<double>(ii, 0))
						errorArr.at<double>(ii, 0) = 0;
				}
				Mat we = D.t()*errorArr;
				double weightedError = we.at<double>(0, 0);

				if(weightedError < minError)
				{
					minError = weightedError;
					bestEst = predict.clone();
					bs.dim = j;
					bs.ineq = ineq;
					bs.thresh = threshVal;
				}
			}
		}
	}

}

void testSimpleData()
{

	double b[5] = {1, 1, -1, -1, 1};
	CvMat labels = cvMat(1, 5, CV_64F, b);
	double c[10] = {1.0, 2.1, 2.0, 1.1, 1.3, 1, 1, 1, 2, 1};
	CvMat data = cvMat(5, 2, CV_64F, c);
	Mat D = Mat::ones(5, 1, CV_64F);
	D /= 5.0;
	BestStump bs;
	Mat est;
	buildStump(Mat(&data), Mat(&labels), D, est, bs);
	cout<<"############### testSimpleData ################"<<endl;
	cout<< D<<endl;
	cout<<est<<endl;

	printf("Best Stump: %d %d, %.6f\n", bs.dim, bs.ineq, bs.thresh);

	cout<<"##############################################"<<endl;
}