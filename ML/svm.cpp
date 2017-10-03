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

data,label=loadDataSet("testSet.txt")
smoSimple(data, label, 0.6, 0.001, 40)

*/

#include "stdafx.h"
#include "MachineLearning.h"
#include <string>
#include <stdlib.h>
#include "Utils.h"
#include "xFile.h"


int selectJrand(int i, int m)
{
	int j = i;
	while(j == i)
	{
		j = rand()%(m);
	}

	return j;
}
float clipAlpha(float data, float L, float H)
{
	return min(H, max(L, data));
}
int smoSimple(Mat& src, Mat& labels, float C, float toler, int maxIter, float& b, Mat& ret)
{
    Mat labelMat;
	transpose(labels, labelMat);
	srand(time(0));

    int m = src.rows;
	int n = src.cols;
	Mat alphas = Mat::zeros(m, 1, CV_32F);

	int iter = 0;
	while(iter < maxIter)
	{
		int alphaChanged = 0;
		for( int i = 0; i < m; i++)
		{
			Mat al = alphas.mul(labels);
			Mat dd = src*(src.row(i).t());
			
			Mat ad = al.t()*dd;
			float fxi = ad.at<float>(0, 0) + b;
	
			float fli = labels.at<float>(i, 0);
			float Ei = fxi - fli;
			float iOld = alphas.at<float>(i, 0);
			float L, H;
			if((fli*Ei < -toler && alphas.at<float>(i, 0) < C) || 
				(fli*Ei > toler && alphas.at<float>(i, 0) > 0))
			{
				int j = selectJrand(i, m);
				
				Mat ddj = src*(src.row(j).t());
				Mat adj = al.t()*ddj; 
				float fxj = adj.at<float>(0, 0) + b;
				float flj = labels.at<float>(j, 0);
				float Ej = fxj - flj;
				float jOld = alphas.at<float>(j, 0);
				if(fli != flj){
					L = max(0, jOld-iOld);
					H = min(C, C+jOld-iOld);
				} else {
					L = max(0, jOld +iOld - C);
					H = min(C, jOld+iOld);
				}
				if(L== H)
				{
					cout<< "L==H"<< endl;
					continue;
				}
				Mat diff = 2.0 * src.row(i)*src.row(j).t() - src.row(i)*src.row(i).t() - src.row(j)*src.row(j).t();
				float eta = diff.at<float>(0 ,0);
				if(eta > 0)
				{
					cout<<"eta > 0"<<endl;
					continue;
				}
				float changes = flj*(Ei-Ej)/eta;
				alphas.at<float>(j, 0) -= changes;

		
				alphas.at<float>(j, 0)= clipAlpha(alphas.at<float>(j, 0), L,H);

				if (abs(alphas.at<float>(j, 0) - jOld) < 0.00001) {
					cout<<"j not moving enough"<<endl;
					continue;
				}
				alphas.at<float>(i, 0) += flj*fli*(jOld-alphas.at<float>(j, 0));
				diff =  fli*(alphas.at<float>(i, 0)-iOld)*src.row(i)*src.row(i).t() - flj*(alphas.at<float>(i, 0) - jOld)*src.row(i)*src.row(j).t();
				float b1 = b - Ei - diff.at<float>(0 ,0);
				diff = fli*(alphas.at<float>(i, 0)-iOld)*src.row(i)*src.row(j).t() - flj*(alphas.at<float>(i, 0) - jOld)*src.row(j)*src.row(j).t();
				float b2 = b - Ej - diff.at<float>(0 ,0);
				if(alphas.at<float>(i, 0) > 0 && C > alphas.at<float>(i, 0)) b = b1;
				else if(alphas.at<float>(j, 0) > 0 && C > alphas.at<float>(j, 0)) b = b2;
				else b = (b1 + b2) / 2.0f;
				alphaChanged += 1;

			}
			
		}
		if(!alphaChanged) iter++;
		else iter = 0;
		cout<<"iter: "<<iter <<endl;

	}
	ret = alphas.clone();
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
	float b = 0; Mat ret;
	smoSimple(data, labels, 0.6, 0.001, 40, b, ret);

	cout<<"##################################################################"<<endl;
	cout<<b<<endl;
	cout<<ret<<endl;

}