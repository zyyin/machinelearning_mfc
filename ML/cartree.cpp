#include "stdafx.h"
#include "MachineLearning.h"
#include <string>
#include <stdlib.h>
#include "Utils.h"
#include "xFile.h"
#include <time.h>
#include "MLTree.h"

bool loadDataSetCART(const char* fileName, Mat& data)
{
	int lineNumber = 0;
	xFile file(fileName, "r");
	Utils util;
	char *pBuffer;
	string strPattern = "\t";
	vector<string> vec;
	while(pBuffer = file.ReadString())
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

double regLeaf(Mat& data)
{
	return mean(data.col(data.cols-1))[0];
}
double regErr(Mat& data)
{
	Scalar mean, dev;
	meanStdDev(data, mean, dev);
	return dev[0];
}

void chooseBestSplit(Mat& data, int ts, int tn, int& bestIndex, double& bestVal)
{
	Mat t = data.col(data.cols-1);
	
	for(int i = 1; i < t.rows; i++)
	{
		if(t.at<double>(i, 0) == t.at<double>(0, 0) )
		{
			bestIndex = -1;
			bestVal = regLeaf(data);
			return;
		}
	}
	int m = data.rows;
	int n = data.cols;
	double S = regErr(data);
	double bestS = DBL_MAX;
	bestIndex = 0;
	bestVal = 0;
	for(int i = 0; i < n-1; i++) {
		set<double> setData;
		for(int j = 0; j < m; j++) {
			setData.insert(data.at<double>(j, i));
		}
		for(set<double>::iterator iter = setData.begin(); iter != setData.end(); iter++)
		{
			Mat m1, m2;
			binSplit(data, i, *iter, m1, m2);
			if(m1.rows < tn || m2.rows < tn) continue;
			double newS = regErr(m1) + regErr(m2);
			if(newS < bestS) {
				bestIndex = i;
				bestVal = *iter;
				bestS = newS;
			}
		}
	}

	if(S - bestS < ts) {
		bestIndex = -1;
		bestVal = regLeaf(data);
		return;
	}
	Mat m1, m2;
	binSplit(data, bestIndex, bestVal, m1, m2);
	if(m1.rows < tn || m2.rows < tn) {
		bestIndex = -1;
		bestVal = regLeaf(data);
	}
}

void createCARTree(Mat& data, int ts, int tn, MLTree& tree, TreeNode* node){
	int index;
	double bestVal;
	chooseBestSplit(data, ts, tn, index, bestVal);
	if(index == -1)
		return;
	TreeNode* pNewNode = new TreeNode;
	pNewNode->val = bestVal;
	pNewNode->element = index;
	tree.addNode(node, pNewNode);
	Mat m1, m2;
	binSplit(data, index, bestVal, m1, m2);
	createCARTree(m1, ts, tn, tree, pNewNode);
	createCARTree(m2, ts, tn, tree, pNewNode);
}

void testCARTree()
{
#if 0
	Mat data = Mat::eye(4, 4, CV_64F);
	Mat m1, m2;
	binSplit(data, 1, 0.5, m1, m2);

	cout<<m1<<endl;
	cout<<m2<<endl;
#endif
	Mat data;
	loadDataSetCART("ex00.txt", data);
	MLTree tree;
	createCARTree(data, 1, 4, tree, NULL);
	tree.print();
}