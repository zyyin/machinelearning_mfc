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
#include "MLTree.h"

double calcShannonEnt(Mat& data)
{
	map<double, int> labelCounts;
	int n = data.rows;
	int m = data.cols;
	for(int i = 0; i < n; i++)
	{
		double label = data.at<double>(i, m-1);
		map<double, int>::iterator it = labelCounts.begin();
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
	double shannonEnt = 0.0;
	map<double, int>::iterator it = labelCounts.begin();
	for(; it != labelCounts.end(); it++)
	{
		double prob = it->second;
		prob/=n;
		shannonEnt -= prob * log(prob)/log(2.0f);
	}
	return shannonEnt;
}

int splitDataSet(Mat& data, int axis, double val, Mat& dst)
{
	int n = data.rows;
	int m = data.cols;
	Mat reducedFeatVec(1, m-1, CV_64F);
	int rows = 0;
	int row = 0;
	for(int i = 0; i < n; i++)
	{
		if(data.at<double>(i, axis) == val)
		{
			rows++;
		}
	}
	dst.create(rows, m-1, CV_64F);
	for(int i = 0; i < n; i++)
	{
		if(data.at<double>(i, axis) == val)
		{
			int k = 0;
			for(int j = 0; j < axis; j++)
			{
				reducedFeatVec.at<double>(0, k++) = data.at<double>(i, j);
			}
			for(int j = axis + 1; j < m; j++)
			{
				reducedFeatVec.at<double>(0, k++) = data.at<double>(i, j);
			}
			for(int j = 0; j < m-1; j++)
			{
				dst.at<double>(row, j) = reducedFeatVec.at<double>(0, j);
			}
			row++;
		}
		
	}
	return row;
}

int chooseBestFeatureToSplit(Mat& data)
{
	int numFeatures = data.cols -1;
	double baseEntropy = calcShannonEnt(data);
	double bestInfoGain = 0.0;
	double bestFeature = -1;
	
	for(int i = 0; i< numFeatures; i++)
	{
		Mat featList(1, data.rows, CV_64F);
		set<double> s;
		for(int j = 0; j < data.rows; j++)
		{
			featList.at<double>(0, j) = data.at<double>(j, i);
			s.insert(data.at<double>(j, i));
		}

		double newEntropy = 0.0;
		for(set<double>::iterator j = s.begin(); j != s.end(); j++)
		{
			//cout<<*j<<"  "<<s.count(*j)<<endl;
			Mat dst;
			splitDataSet(data, i, *j, dst);
			
			double prob = dst.rows;
			prob /= double(data.rows);
			newEntropy += prob * calcShannonEnt(dst) ;    
		}
		double infoGain = baseEntropy - newEntropy;
		if (infoGain > bestInfoGain)
		{
			bestInfoGain = infoGain;
			bestFeature = i;
		}
		
	}
	return bestFeature;
}


double majorityCnt(Mat& classList)
{
	map<double, int> classCount;
	for(int i = 0; i < classList.cols; i++)
	{
		map<double, int>::iterator it = classCount.begin();
		double vote = classList.at<double>(0, i);
		for(; it != classCount.end(); it++)
		{
			if(vote == it->first) {
				it->second++;
				break;
			}
		}
		if(it == classCount.end())
		{
			classCount[vote] = 1;
		}
	}
	double maxKey = 0;
	int maxVal = 0;
	map<double, int>::iterator it = classCount.begin();
	for(; it != classCount.end(); it++)
	{
		if(maxVal < it->second) {
			maxVal = it->second;
			maxKey = it->first;
		}
	}
	return maxKey;
}

void DeleteOneColOfMat(Mat& object,int num)  
{  
    if (num<0 || num>=object.cols)  
    {  
    }  
    else  
    {  
        if (num == object.cols-1)  
        {  
            object = object.t();  
            object.pop_back();  
            object = object.t();  
        }  
        else  
        {  
            for (int i=num+1;i<object.cols;i++)  
            {  
                object.col(i-1) = object.col(i) + Scalar(0);  
            }  
            object = object.t();  
            object.pop_back();  
            object = object.t();  
        }  
    }  
}
double createTree(Mat& data, Mat& label, MLTree& tree, TreeNode* parent, int iVal)
{
	Mat classList(1, data.rows, CV_64F);
	for(int i = 0; i < data.rows; i++)
	{
		classList.at<double>(0, i) = data.at<double>(i, data.cols -1);
	}
	
	// check classList
	bool allEqual = true;
	for(int i = 0; i < classList.cols; i++)
	{
		if(classList.at<double>(0, i) != classList.at<double>(0, 0))
		{
			allEqual = false;
			break;
		}
	}
	if(allEqual) {
		TreeNode* node = new TreeNode();
		node->element = classList.at<double>(0, 0);
		node->val = iVal;
        tree.addNode(parent, node);
		return 0;
	}
	if(data.cols == 1)
	{
		TreeNode* node = new TreeNode();
		node->element = majorityCnt(classList);
		node->val = iVal;
        tree.addNode(parent, node);
		return 0;
	}
	int best = chooseBestFeatureToSplit(data);
	double bestLabel = label.at<double>(0, best);
	TreeNode* node = new TreeNode();
	node->element = bestLabel;
	node->val = iVal;
        tree.addNode(parent, node);
	DeleteOneColOfMat(label, best);
	Mat featValues = data.col(best);
	cout<<featValues<<endl;
        set<double> s;
	for(int i = 0; i < featValues.rows; i++)
	{
		s.insert(featValues.at<double>(i, 0));
	}
	set<double>::iterator j = s.begin();
	for(; j != s.end(); j++)
	{
		cout<<*j<<"  "<<s.count(*j)<<endl;
		Mat subLabel = label.clone();
		Mat dst;
		splitDataSet(data, best, *j, dst);
		
		createTree(dst, subLabel, tree, node, *j);
		tree.print();
		
	}
	return 0;
}

TreeNode* findRoot(TreeNode* node,double key)
{
	cout<<"findRoot "<<key<<endl;
	if(node == NULL)
		return NULL;
	if(node->val == key)
		return node;
	TreeNode* p = findRoot(node->firstChild, key);
	if(p != NULL) return p;
	return findRoot(node->nextSibling, key);	
}

double classifyTree(TreeNode* node, Mat& label, Mat& testData)
{
	double first = node->element;
	TreeNode* second = node->firstChild;
	int featIndex = 0; 
	for(; featIndex < label.cols; featIndex++)
	{
		if(first == label.at<double>(0, featIndex))
			break;
	}
	double key = testData.at<double>(0, featIndex);
	TreeNode* valueOfFeat = findRoot(second, key);
	if(valueOfFeat) {
		double classLabel = 0;
		if(valueOfFeat->firstChild)
		{
			classLabel = classifyTree(valueOfFeat, label, testData);
		}
		else classLabel = valueOfFeat->element;
	
		return classLabel;
	}
	return 0;

}
void testTree()
{
	double c[15] = {1, 1, 101, 1, 1, 101, 1, 0, 100, 0, 1, 100, 0, 1, 100};
	CvMat data = cvMat(5, 3, CV_64F, c);
	double d[2] = {201, 202};
    double e[2] = {1, 1};
	CvMat label = cvMat(1, 2, CV_64F, d);
	CvMat testData = cvMat(1, 2, CV_64F, e);
	Mat mData(&data);
	Mat mLabel(&label);
	Mat mTestData(&testData);
	int n = chooseBestFeatureToSplit(mData);
	cout<< n <<endl;
	MLTree tree;
	createTree(mData, mLabel, tree, NULL, -1);
	cout<<"################"<<endl;
	tree.print();
	double ret = classifyTree(tree.root, mLabel, mTestData);
	cout<<ret<<endl;
	
	/*
	TreeNode* node = new TreeNode();
	node->element = 5;
        Tree tree;
        tree.addNode(NULL, node);
        
	TreeNode* node1 = new TreeNode();
	TreeNode* node2 = new TreeNode();
	TreeNode* node3 = new TreeNode();
        node1->element = 6;node2->element = 7;node3->element = 8;
	tree.addNode(node, node2);
	tree.addNode(node, node1);
	tree.addNode(node2, node3);
	tree.print();
	*/
}