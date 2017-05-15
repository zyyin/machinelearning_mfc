#include "stdafx.h"
#include "MachineLearning.h"

float calcShannonEnt(Mat& data)
{
	map<float, int> labelCounts;
	int n = data.rows;
	int m = data.cols;
	for(int i = 0; i < n; i++)
	{
		float label = data.at<float>(i, m-1);
		map<float, int>::iterator it = labelCounts.begin();
		for(; it != labelCounts.end(); it++)
		{
			if(label == it->first)
			{
				it->second+=1;
				break;
			}
		}
		if(it == labelCounts.end())
		{
			labelCounts[label] = 1;
		}

	}
	float shannonEnt = 0.0;
	map<float, int>::iterator it = labelCounts.begin();
	for(; it != labelCounts.end(); it++)
	{
		float prob = it->second;
		prob/=n;
		printf("%.2f\n", prob);
		shannonEnt -= prob * log(prob)/log(2.0f);
	}
	return shannonEnt;
}

int splitDataSet(Mat& data, int axis, float val, Mat& dst)
{
	int n = data.rows;
	int m = data.cols;
	Mat reducedFeatVec(1, m-1, CV_32F);
	int rows = 0;
	int row = 0;
	for(int i = 0; i < n; i++)
	{
		if(data.at<float>(i, axis) == val)
		{
			rows++;
		}
	}
	dst.create(rows, m-1, CV_32F);
	for(int i = 0; i < n; i++)
	{
		if(data.at<float>(i, axis) == val)
		{
			int k = 0;
			for(int j = 0; j < axis; j++)
			{
				reducedFeatVec.at<float>(0, k++) = data.at<float>(i, j);
			}
			for(int j = axis + 1; j < m; j++)
			{
				reducedFeatVec.at<float>(0, k++) = data.at<float>(i, j);
			}
			for(int j = 0; j < m-1; j++)
			{
				dst.at<float>(row, j) = reducedFeatVec.at<float>(0, j);
			}
			row++;
		}
		
	}
	cout<<dst<<endl;
	return row;
}

void testTree()
{
	float c[15] = {1, 1, 101, 1, 1, 101, 1, 0, 100, 0, 1, 100, 0, 1, 100};
	CvMat data = cvMat(5, 3, CV_32F, c);
	float shannonEnt = calcShannonEnt(Mat(&data));
	printf("shannonEnt = %.15f\n", shannonEnt);
	Mat dst;
	int ret = splitDataSet(Mat(&data), 0, 0, dst);
	cout<<ret <<endl;


	
}