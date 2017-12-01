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
	double minError;
	double alpha;
	
} BestStump;
void buildStump(Mat& data, Mat& label, Mat& D, Mat& bestEst, BestStump& bs)
{
	int m = data.rows;
	int n = data.cols;
	
	bestEst = Mat::zeros(m, 1, data.type());
	double numSteps = 10.0;
	bs.minError = DBL_MAX;
	for(int j = 0; j < n; j++)
	{
		double minVal, maxVal;
		minMaxLoc(data.col(j), &minVal, &maxVal);
		double stepSize = (maxVal-minVal)/numSteps;
		for(int i = 0; i < numSteps+1; i++)
		{
			for(int ineq = 0; ineq < 2; ineq++) {
				double threshVal = minVal + i*stepSize;
				Mat predict;
				stumpClassify(data, j, threshVal,ineq, predict);
				Mat errorArr;
				absdiff(label , predict, errorArr);
				
				Mat we = D.t()*errorArr/2;
				double weightedError = we.at<double>(0, 0);

				if(weightedError < bs.minError)
				{
					bs.minError = weightedError;
					bestEst = predict.clone();
					bs.dim = j;
					bs.ineq = ineq;
					bs.thresh = threshVal;
				}
			}
		}
	}

}

void sign(Mat& m)
{
	for(int j = 0; j < m.rows; j++)
	{
		if(m.at<double>(j, 0) > 0)
			m.at<double>(j, 0) = 1;
		else if(m.at<double>(j, 0) < 0)
			m.at<double>(j, 0) = -1;
	}

}
class WeakClassifier
{
public:
	WeakClassifier(BestStump& bs)
	{
		m_bs = bs;
	}

	void solve(Mat& data, Mat& ret){
		Mat est;
		stumpClassify(data, m_bs.dim, m_bs.thresh, m_bs.ineq, ret);
		ret *= m_bs.alpha;
	}

	BestStump m_bs;

};


void adaBoostTrainDS(Mat& data, Mat& label, int numIter, vector<WeakClassifier>& weakClassifierList)
{
	int m = data.rows;
	Mat D = Mat::ones(m, 1, data.type())/m;
	Mat agg = Mat::zeros(m, 1, data.type());
	for(int i = 0; i < numIter; i++)
	{

		BestStump bs;
		Mat est;
		buildStump(data, label, D, est, bs);

		double alpha = 0.5*log((1.0-bs.minError)/max(bs.minError,1e-16));
		bs.alpha = alpha;
		weakClassifierList.push_back(bs);
		Mat tmp = -1*alpha*label;
		Mat expon = tmp.mul(est);
		for(int j = 0; j < expon.rows; j++)
		{
			expon.at<double>(j, 0) = exp(expon.at<double>(j, 0));
		}
		D = D.mul(expon);
		double sum = 0;
		for(int j = 0; j < m; j++)
		{
			sum += D.at<double>(j, 0);
		}
		D /= sum;
		Mat tt = alpha*est;
		agg += alpha*est;
		Mat aggTmp = agg.clone();
		sign(aggTmp);
		absdiff(aggTmp, label, aggTmp);
		aggTmp/=2;
		double errorRate = mean(aggTmp)[0];
		cout<<errorRate<<endl;
		if(errorRate == 0)
			break;
		
	}
}

void adaClassify(Mat& data, vector<WeakClassifier>& weakClassifierList, Mat& agg)
{
	int m = data.rows;
	agg = Mat::zeros(m, 1, data.type());
	for(int i = 0; i < weakClassifierList.size(); i++)
	{
		Mat ret;
		weakClassifierList[i].solve(data, ret);
		agg += ret;
	}
	sign(agg);
}
void testSimpleData()
{

	double b[5] = {1, 1, -1, -1, 1};
	CvMat labels = cvMat(5, 1, CV_64F, b);
	double c[10] = {1.0, 2.1, 2.0, 1.1, 1.3, 1, 1, 1, 2, 1};
	CvMat data = cvMat(5, 2, CV_64F, c);
	Mat D = Mat::ones(5, 1, CV_64F);
	D /= 5.0;
	BestStump bs;
	Mat est;
	Mat mData(&data);
	Mat mLabel(&labels);
	buildStump(mData, mLabel, D, est, bs);
	cout<<"############### testSimpleData ################"<<endl;
	cout<< D<<endl;
	cout<<est<<endl;

	printf("Best Stump: %d %d, %.6f\n", bs.dim, bs.ineq, bs.thresh);

	cout<<"##############################################"<<endl;
	
	vector<WeakClassifier> weakClassifierList;
	cout<<weakClassifierList.size()<<endl;
	adaBoostTrainDS(mData, mLabel, 9, weakClassifierList);
	
	mData = Mat::zeros(1, 2, CV_64F);
	mData.at<double>(0, 0) = 0;
	mData.at<double>(0, 1) = -11;
	Mat agg;
	adaClassify(mData, weakClassifierList, agg);
	

	cout<<agg<<endl;
}

bool loadDataSetBoost(const char* fileName, Mat& data, Mat& labels)
{
	int lineNumber = 0;
	xFile file(fileName, "r");
	vector<string> vec;
	Utils util;
	char *pBuffer;
	string strPattern = "\t";
	while(pBuffer=file.ReadString())
	{
		if(vec.empty()) {
			string str = pBuffer;
			vec = util.split(str, strPattern);
		}
		lineNumber++;
	}
	if(lineNumber < 1)
		return false;
	file.SeekToBegin();
	data.create(lineNumber, vec.size()-1, CV_64F);
	labels.create(lineNumber, 1, CV_64F);
	int i = 0;
	while(pBuffer=file.ReadString())
	{
		string str = pBuffer;
		vector<string> vec = util.split(str, strPattern);
		for(int j = 0; j < vec.size() - 1; j++){
			data.at<double>(i, j) = atof(vec[j].c_str());
		
		}
		labels.at<double>(i, 0) = atof(vec[vec.size()-1].c_str());
		i++;
	}
	return true;
}

void testAdaBoost()
{
	Mat data, label;
	loadDataSetBoost("horseColicTraining2.txt", data, label);
	vector<WeakClassifier> weakClassifierList;
	
	adaBoostTrainDS(data, label, 100, weakClassifierList);
	cout<<weakClassifierList.size()<<endl;
	loadDataSetBoost("horseColicTest2.txt", data, label);
	Mat agg;
	adaClassify(data, weakClassifierList, agg);

	absdiff(agg, label, agg);

	agg/=2;
	cout<<mean(agg)[0]*data.rows<<endl;
}