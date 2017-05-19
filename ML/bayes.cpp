#include "stdafx.h"
#include "MachineLearning.h"
#include <string>
using namespace std;

void loadDataSet(vector<vector<string> >& dataList, Mat& listClass)
{
	const string s[] = {
		"my", "dog", "has", "flea", "problems", "help", "please",
		"maybe", "not", "take", "him", "to", "dog", "park", "stupid",
		"my", "dalmation", "is", "so", "cute", "I", "love", "him",
		"stop", "posting", "stupid", "worthless", "garbage",
		"mr", "licks", "ate", "my", "steak", "how", "to", "stop", "him",
		"quit", "buying", "worthless", "dog", "food", "stupid"
	};

	dataList.push_back(vector<string>(s, s+7));
	dataList.push_back(vector<string>(s+7, s+15));
	dataList.push_back(vector<string>(s+15, s+23));
	dataList.push_back(vector<string>(s+23, s+28));
	dataList.push_back(vector<string>(s+28, s+37));
	dataList.push_back(vector<string>(s+37, s+43));
	
	for(int i = 0; i < dataList.size(); i++) {
		for(int j = 0; j < dataList[i].size(); j++) {
			cout<<dataList[i][j]<<"   ";
		}
		cout<<endl;
	}
	static float c[6] = {0,1,0,1,0,1};
	listClass = Mat(1, 6, CV_32F, c);
	cout<<listClass<<endl;
	//cout<<dataList<<endl;
}

void createVocabList(vector<vector<string> >& dataList, set<string>& vocabSet)
{
	vocabSet.clear();
	for(int i = 0; i < dataList.size(); i++) {
		for(int j = 0; j < dataList[i].size(); j++) {
			vocabSet.insert(dataList[i][j]);
		}
	}
	set<string>::iterator it = vocabSet.begin();
	for(; it != vocabSet.end(); it++)
		cout<<*it<< "  ";
}

void setOfWords2Vec(set<string>& vocabSet, vector<string>& inputSet, Mat& ret)
{
	//ret.create(1, vocabSet.size(), CV_8U);
	ret = Scalar(0);
	for(int i = 0; i < inputSet.size(); i++)
	{
		string s = inputSet[i];
		int index = 0;
		set<string>::iterator it = vocabSet.begin();
		for(; it != vocabSet.end(); it++) {
		    if(s == *it) {
		    	ret.at<float>(0, index) = 1;
			break;
		    }
		    index++;
		}
	}
	
	cout<<ret<<endl;
}

float sumMat(Mat& m){
	float sum = 0;
	for(int i = 0; i < m.rows; i++)
	{
		for(int j = 0; j < m.cols; j++)
			sum += m.at<float>(i, j);
	}
	return sum;
}
void trainNB0(Mat& trainMatrix, Mat& listClass, Mat& p0, Mat& p1, float& pAbusive)
{
	int numTrainDocs = trainMatrix.rows;
    	int numWords = trainMatrix.cols;
	pAbusive = sumMat(listClass);

	pAbusive/=numTrainDocs;
	cout<<pAbusive<<endl;
	Mat p0Num(1, numWords, CV_32F);
	p0Num = Scalar(1.0f);
	Mat p1Num = p0Num.clone();
	float p0Denom = 2;
	float p1Denom = 2;
	for(int i = 0; i < numTrainDocs; i++) {
		if(listClass.at<float>(0, i)) {
			p1Num += trainMatrix.row(i);
			Mat tmp = trainMatrix.row(i);
			p1Denom += sumMat(tmp);
		} else {
			p0Num += trainMatrix.row(i);
			Mat tmp = trainMatrix.row(i);
			p0Denom += sumMat(tmp);
		}
	}
	for(int i = 0; i < p0Num.cols; i++) {
		p0Num.at<float>(0, i) = log(p0Num.at<float>(0, i)/p0Denom)/log(2);
	}
	for(int i = 0; i < p1Num.cols; i++) {
		p1Num.at<float>(0, i) = log(p1Num.at<float>(0, i)/p1Denom)/log(2);
	}
	
	cout<<p0Num<<endl;
	cout<<p1Num<<endl;
	p0 = p0Num.clone();
	p1 = p1Num.clone();
	//pAbusive = sum(trainCategory)/float(numTrainDocs)
}

void testBayes()
{
	vector<vector<string> > dataList;
	Mat listClass;
	loadDataSet(dataList, listClass);
	set<string> vocabSet;
	createVocabList(dataList, vocabSet);

	Mat trainMatrix(dataList.size(), vocabSet.size(), CV_32F);
	for(int i = 0; i < dataList.size(); i++)
	{
		Mat m = trainMatrix.row(i);
		setOfWords2Vec(vocabSet, dataList[i], m);
	}
	cout<<"T  "  <<trainMatrix<<endl;
	Mat p0, p1;
	float abusive;
	trainNB0(trainMatrix, listClass, p0, p1, abusive);
}


