#include "stdafx.h"
#include "MachineLearning.h"
#include <string>
#include <stdlib.h>
#include "Utils.h"
#include "xFile.h"
#include <time.h>


bool loadDataSet(const char* fileName, Mat& data)
{
	int lineNumber = 0;
	xFile file(fileName, "r");
	Utils util;
	char *pBuffer;
	string strPattern = "\t";
	vector<string> vec;
	while(file.ReadString())
	{
		lineNumber++;
		if(vec.empty()) {
			string str = pBuffer;
			vec = util.split(str, strPattern);
		}
	}
	if(lineNumber < 1)
		return false;
	file.SeekToBegin();
	data.create(lineNumber, vec.size(), CV_64F);
	
	int i = 0;
	while(pBuffer=file.ReadString())
	{
		string str = pBuffer;
		vector<string> vec = util.split(str, strPattern);
		for(int j = 0; j < vec.size() - 1; j++){
			data.at<double>(i, j) = atof(vec[j].c_str());
		}
		i++;
	}
	return true;
}


void binSplit(Mat& data, int j, double val, Mat& m1, Mat& m2)
{
	int row1 = 0;
	int row2 = 0;
	for(int i = 0; i < data.rows; i++)
	{
		double d = data.at<double>(i, j);
		if(d > val) row1++;
		else row2++;
	}
	m1.create(row1, data.cols, data.type());
	m2.create(row2, data.cols, data.type());
	int k1 = 0;
	int k2 = 0;
	for(int i = 0; i < data.rows; i++)
	{
		double d = data.at<double>(i, j);
		if(d > val) data.row(i).copyTo(m1.row(k1++));
		else data.row(i).copyTo(m2.row(k2++));
	}
}

void testCARTree()
{
	Mat data = Mat::eye(4, 4, CV_64F);
	Mat m1, m2;
	binSplit(data, 1, 0.5, m1, m2);

	cout<<m1<<endl;
	cout<<m2<<endl;
}