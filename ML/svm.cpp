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
#include <string>
#include "Utils.h"
#include "xFile.h"

int smoSimple(Mat& src, Mat& labels, float C, float toler, int maxIter, float& b, Mat& ret)
{
    Mat labelMat;
	transpose(labels, labelMat);
    int m = src.rows;
	int n = src.cols;
	Mat alphas(m, 1, CV_32F);
	alphas = 0;
	int iter = 0;
	while(iter < maxIter)
	{
		int alphaChanged = 0;
		for( int i = 0; i < m; i++)
		{
			Mat al = alphas*labelMat;
			Mat dd = src*(src.row(i));
			Mat ad = al.t()*dd.t();
			float fxi = ad.at<float>(0, 0) + b;
			cout<<fxi<<endl;
		}
	}

	return 0;
}

bool loadDataSet(const char* fileName, Mat& data, Mat& labels)
{
	int lineNumber = 0;
	xFile file(fileName, "r");
	while(file.ReadString())
	{
		lineNumber++;
	}
	if(lineNumber < 1)
		return false;
	file.SeekToBegin();
	data.create(lineNumber, 2, CV_32F);
	labels.create(lineNumber, 1, CV_32F);
	Utils util;
	char *pBuffer;
	string strPattern = "\t";
	int i = 0;
	while(pBuffer=file.ReadString())
	{
		string str = pBuffer;
		vector<string> vec = util.split(str, strPattern);
		data.at<float>(i, 0) = atof(vec[0].c_str());
		data.at<float>(i, 1) = atof(vec[1].c_str());
		labels.at<float>(i, 0) = atof(vec[2].c_str());
		i++;
	}
	return true;
}

void testSVM()
{
	Mat data, labels;
	loadDataSet("testSet.txt", data, labels);
	cout<<data<<endl;
	cout<<labels;
}