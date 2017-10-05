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
#include <stdlib.h>
#include "Utils.h"
#include "xFile.h"
#include <time.h>

int selectJrand(int i, int m)
{
	int j = i;
	while(j == i)
	{
		j = rand()%(m);
	}
	cout<<"rand: " << j<<endl;
	return j;
}
double clipAlpha(double data, double L, double H)
{
	return min(H, max(L, data));
}
int smoSimple(Mat& src, Mat& labels, double C, double toler, int maxIter, double& b, Mat& ret)
{
    Mat labelMat;
	transpose(labels, labelMat);
	srand(time(NULL));

    int m = src.rows;
	int n = src.cols;
	Mat alphas = Mat::zeros(m, 1, CV_64F);

	int iter = 0;
	while(iter < maxIter)
	{
		int alphaChanged = 0;
		for( int i = 0; i < m; i++)
		{
			Mat al = alphas.mul(labels);
			Mat dd = src*(src.row(i).t());
			
			Mat ad = al.t()*dd;
			double fxi = ad.at<double>(0, 0) + b;
	
			double fli = labels.at<double>(i, 0);
			double Ei = fxi - fli;
			double iOld = alphas.at<double>(i, 0);
			double L, H;
			if((fli*Ei < -toler && iOld < C) || 
				(fli*Ei > toler && iOld > 0))
			{
				int j = selectJrand(i, m);
				Mat ddj = src*(src.row(j).t());
				Mat adj = al.t()*ddj; 
				double fxj = adj.at<double>(0, 0) + b;
				double flj = labels.at<double>(j, 0);
				double Ej = fxj - flj;
				double jOld = alphas.at<double>(j, 0);
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
				double eta = diff.at<double>(0 ,0);
				if(eta > 0)
				{
					cout<<"eta > 0"<<endl;
					continue;
				}
				double changes = flj*(Ei-Ej)/eta;
				alphas.at<double>(j, 0) -= changes;

		
				alphas.at<double>(j, 0)= clipAlpha(alphas.at<double>(j, 0), L,H);

				if (abs(alphas.at<double>(j, 0) - jOld) < 0.00001) {
					cout<<"j not moving enough"<<endl;
					continue;
				}
				alphas.at<double>(i, 0) += flj*fli*(jOld-alphas.at<double>(j, 0));
				diff =  fli*(alphas.at<double>(i, 0)-iOld)*src.row(i)*src.row(i).t() + flj*(alphas.at<double>(j, 0) - jOld)*src.row(i)*src.row(j).t();
				double b1 = b - Ei - diff.at<double>(0 ,0);
				diff = fli*(alphas.at<double>(i, 0)-iOld)*src.row(i)*src.row(j).t() + flj*(alphas.at<double>(j, 0) - jOld)*src.row(j)*src.row(j).t();
				double b2 = b - Ej - diff.at<double>(0 ,0);
				if(alphas.at<double>(i, 0) > 0 && C > alphas.at<double>(i, 0)) b = b1;
				else if(alphas.at<double>(j, 0) > 0 && C > alphas.at<double>(j, 0)) b = b2;
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
	data.create(lineNumber, 2, CV_64F);
	labels.create(lineNumber, 1, CV_64F);
	Utils util;
	char *pBuffer;
	string strPattern = "\t";
	int i = 0;
	while(pBuffer=file.ReadString())
	{
		string str = pBuffer;
		vector<string> vec = util.split(str, strPattern);
		data.at<double>(i, 0) = atof(vec[0].c_str());
		data.at<double>(i, 1) = atof(vec[1].c_str());
		labels.at<double>(i, 0) = atof(vec[2].c_str());
		i++;
	}
	return true;
}
typedef enum {
	KERNEL_TYPE_LINEAR,
	KERNEL_TYPE_RBF,
} KERNEL_TYPE;

bool kernelTrans(Mat& X, Mat& A, Mat& K, int type, double size) 
{
	int m = X.rows;
	int n = X.cols;
	if(type == KERNEL_TYPE_LINEAR) {
		K = X*A.t();
		return true;
	} else if(type == KERNEL_TYPE_RBF) {
		K.create(m, 1, CV_64F);
		for(int i = 0; i < m; i++) {
			Mat d = X.row(i) - A;
			Mat theta = (d*d.t());
			K.at<double>(i, 0)  = exp(theta.at<double>(0, 0)/(-1*size*size));
		}
	}

	return false;
}
class SVMStruct
{
public:
	SVMStruct(Mat& data, Mat& classLabels, double C, double toler, int kernelType = KERNEL_TYPE_LINEAR, double kernelSize = 0);

	double calcError(int k);
	int selectJ(int i, double ei, double& ej); 
	void updateError(int k);
	int slove(int i);
	Mat& getAlphas() { return alphas; }
	
	Mat alphas;
	double m_b;
private:
	Mat X, labels, eCache, K;
	double m_c, m_toler;
	int m_m;
	
};


SVMStruct::SVMStruct(Mat& data, Mat& classLabels, double C, double toler, int kernelType /* = KERNEL_TYPE_LINEAR */, double kernelSize /* = 0 */)
{
	X = data.clone();
	labels = classLabels.clone();
	m_c = C;
	m_toler = toler;
	m_b = 0;
	m_m = X.rows;
	alphas = Mat::zeros(m_m, 1, CV_64F);
	eCache = Mat::zeros(m_m, 1, CV_64FC2);
	K = Mat::zeros(m_m, m_m, CV_64F);
	for(int i = 0; i < m_m; i++) {
		Mat t;
		kernelTrans(X, X.row(i), t, kernelType, kernelSize);
		t.copyTo(K.col(i));
	}
}

double SVMStruct::calcError(int k)
{
	Mat al = alphas.mul(labels);
	
	Mat kk = K.col(k);
	Mat ad = al.t()*kk;
    double fxk = ad.at<double>(0, 0) + m_b;
	return fxk -  labels.at<double>(k, 0);
}

int SVMStruct::selectJ(int i, double ei, double& ej)
{
	int maxK = -1;
	double maxE = 0;
	ej = 0;
	eCache.at<Vec2d>(i, 0) = Vec2d(1, ei);
	vector<int> errorIndexList;
	for(int ii = 0; ii < m_m; ii++) {
		if(abs(eCache.at<Vec2d>(ii, 0)[0] - 1.0f) < 0.0001) {
			errorIndexList.push_back(ii);
		}
	}

	if(errorIndexList.size() > 1)
	{
		for(int k = 0; k < errorIndexList.size(); k++)
		{
			int kk = errorIndexList[k];
			if(kk == i)  continue;
			double ek = calcError(kk);
			double deltaE = abs(ei - ek);
			if(deltaE > maxE)	
			{
				maxK = kk;
				maxE = deltaE;
				ej = ek;
			}
		}
		return maxK;
	}
	maxK = selectJrand(i, m_m);
	ej = calcError(maxK);
	return maxK;
}
void SVMStruct::updateError(int k)
{
	double e = calcError(k);
	eCache.at<Vec2d>(k, 0) = Vec2d(1, e);
	
}

int SVMStruct::slove(int i)
{
	double ei = calcError(i);
	double iOld = alphas.at<double>(i, 0);
	if( ( (labels.at<double>(i, 0)*ei < -m_toler) && (iOld < m_c)) ||
		 ( (labels.at<double>(i, 0)*ei > m_toler) && (iOld > 0)) ) {
		double ej;
		int j = selectJ(i, ei, ej);
		if(j < 0) j = X.rows - 1;

		double jOld = alphas.at<double>(j, 0);
		double fli = labels.at<double>(i, 0);
		double flj = labels.at<double>(j, 0);
		double L, H;
		if(fli != flj)
		{
			L = max(0, jOld-iOld);
			H = min(m_c, m_c + jOld - iOld);
		} else {
			L = max(0, jOld +iOld - m_c);
			H = min(m_c, jOld+iOld);
		}
		if( L == H)
		{
			cout<< "L==H"<< endl;
			return 0;
		}
		double eta = 2.0 * K.at<double>(i, j) - K.at<double>(i, i) - K.at<double>(j, j);
		if(eta >= 0) {
			cout<<"eta >=0"<<endl;
			return 0;
		}
		double changes = flj*(ei-ej)/eta;
		alphas.at<double>(j, 0) -= changes;
		alphas.at<double>(j, 0)= clipAlpha(alphas.at<double>(j, 0), L,H);
		updateError(j);
		if (abs(alphas.at<double>(j, 0) - jOld) < 0.00001) {
			cout<<"j not moving enough"<<endl;
			return 0;
		}
		alphas.at<double>(i, 0) += flj*fli*(jOld-alphas.at<double>(j, 0));
		updateError(i);
		double diff =  fli*(alphas.at<double>(i, 0)-iOld)*K.at<double>(i,i) + flj*(alphas.at<double>(j, 0) - jOld)*K.at<double>(i, j);

		double b1 = m_b - ei - diff;
		diff = fli*(alphas.at<double>(i, 0)-iOld)*K.at<double>(i,j) + flj*(alphas.at<double>(j, 0) - jOld)*K.at<double>(j,j);
		double b2 = m_b - ej - diff;
		
		if(alphas.at<double>(i, 0) > 0 && m_c > alphas.at<double>(i, 0)) 
			m_b = b1;
		else if(alphas.at<double>(j, 0) > 0 && m_c > alphas.at<double>(j, 0)) 
			m_b = b2;
		else m_b = (b1 + b2) / 2.0f;
		
		return 1;
	} else {
		//cout<<"Skip slove" <<endl;
	}
	return 0;
}

int FullSmo(Mat& src, Mat& labels, double C, double toler, 
			  int maxIter, double& b, Mat& ret, int type = KERNEL_TYPE_LINEAR, double size = 0) 
{
	
	SVMStruct svm(src, labels, C, toler, type, size);
	int iter = 0;
	bool entireSet = true;
	int alphaChanged = 0;
	srand(time(0));
	while(iter < maxIter && (alphaChanged > 0 || entireSet))
	{
		
		alphaChanged = 0;
		if(entireSet)
		{
			for(int i = 0; i < src.rows; i++)
			{
				alphaChanged += svm.slove(i);
				cout<<"fullSet, iter  "<<i << "   "<< alphaChanged<<endl;
			}
			iter++;
		}
		else
		{
			Mat& alphas = svm.getAlphas();
			for(int i = 0; i < alphas.rows; i++){
				double c = alphas.at<double>(i ,0);
				if(c > 0 && c < C) {
					alphaChanged += svm.slove(i);
					cout<<"non-bound, iter  "<<i << "   "<< alphaChanged<<endl;
				}
				
			}
			iter++;
		}
		if(entireSet) entireSet = false;
		else if(alphaChanged == 0) entireSet = true;
	}
	ret  = svm.alphas.clone();
	b = svm.m_b;
	return 0;
}

void calcWs(Mat& alphas, Mat& data, Mat& labels, Mat& ws)
{
	int m = data.rows;
	int n = data.cols;
	ws = Mat::zeros(n, 1, CV_64F);
	for(int i = 0; i < m; i++)
	{
		Mat t = alphas.row(i) * labels.row(i);
		ws += t.at<double>(0, 0)*(data.row(i).t());
	}
}

void testSVM()
{
	Mat data, labels;
	loadDataSet("testSet.txt", data, labels);
	double b = 0; Mat ret;

	//smoSimple(data, labels, 0.6, 0.001, 40, b, ret);
	FullSmo(data, labels, 0.6, 0.001, 40, b, ret);

	cout<<"##################################################################"<<endl;

	cout<<b<<endl;
	cout<<"##################################################################"<<endl;
	Mat ws(2, 1, CV_64F);
	calcWs(ret, data, labels, ws);
	Mat comp = labels.clone();
	
	for(int i = 0; i < data.rows; i++){
		Mat t = data.row(i)*(ws);
		comp.at<double>(i, 0) = t.at<double>(0, 0) + b > 0 ? 1 : -1;
	}
	cv::absdiff(comp, labels, ret);
	cout<<ret<<endl;
	

}

void testRBF()
{
	Mat data, labels;
	loadDataSet("testSetRBF.txt", data, labels);
	double b = 0; Mat ret;

	//smoSimple(data, labels, 0.6, 0.001, 40, b, ret);
	FullSmo(data, labels, 20, 0.0001, 10000, b, ret, KERNEL_TYPE_RBF, 1.3);

	cout<<"##################################################################"<<endl;

	cout<<b<<endl;
	cout<<"##################################################################"<<endl;
	Mat ws(2, 1, CV_64F);
	calcWs(ret, data, labels, ws);
	Mat comp = labels.clone();

	for(int i = 0; i < data.rows; i++){
		Mat t = data.row(i)*(ws);
		comp.at<double>(i, 0) = t.at<double>(0, 0) + b > 0 ? 1 : -1;
	}
	cv::absdiff(comp, labels, ret);
	cout<<ret<<endl;
}

void loadImages(const char* path,  Mat& data, Mat& label)
{
	vector<string> fileList;
	vector<string> nameList;
	Utils util;
	util.BrowseFolder(path, fileList, nameList);
    data.create(nameList.size(), 32*32, CV_64F);
	label.create( nameList.size(),1, CV_64F);
	for(int i = 0; i < fileList.size(); i++)
	{
		xFile file(fileList[i].c_str(), "r");
		int line = 0;
		char* pBuffer;
		while(pBuffer=file.ReadString()){
			for(int j = 0; j < 32; j++)
			{
				data.at<double>(i, j+line*32) = pBuffer[j] - '0';
			}
			line++;
		}
		file.Close();
		label.at<double>(i, 0) = atoi(nameList[i].c_str()) == 9 ? -1 : 1;
	}
}

void testSVMHandWriting()
{

	Mat data, label;
	loadImages("trainingDigits", data, label);

	double b = 0; Mat ret;

	FullSmo(data, label, 200, 0.0001, 10000, b, ret, KERNEL_TYPE_RBF, 10);

	cout<<"##################################################################"<<endl;

	int m = data.rows;
	int n = data.cols;
	int svNum = 0;
	for(int i = 0; i < ret.rows; i++)
	{
		double a = ret.at<double>(i, 0);
		if(a != 0)
		{
			svNum ++;
		}
	}
	cout<<"SVNumber: "<<svNum<<endl;

	Mat svData(svNum, n, CV_64F);
	Mat svLabel(svNum, 1, CV_64F);
	Mat svAlpha(svNum, 1, CV_64F);

	int j = 0;
	for(int i = 0; i < ret.rows; i++)
	{
		double a = ret.at<double>(i, 0);
		if(a != 0)
		{
			data.row(i).copyTo(svData.row(j));
			label.row(i).copyTo(svLabel.row(j));
			ret.row(i).copyTo(svAlpha.row(j));
			j++;
		}
	}

	Mat svLA = svLabel.mul(svAlpha);
	svNum = 0;
	for(int i = 0; i < data.rows; i++)
	{
		Mat K;
		kernelTrans(svData, data.row(i), K, KERNEL_TYPE_RBF, 10);
		Mat t = K.t()*svLA;
		double predict = t.at<double>(0, 0) + b;
		if(predict*label.at<double>(i, 0) < 0)
			svNum++;
	}
	
	printf("Training Errors: %d / %d\n", svNum, data.rows);

	loadImages("testDigits", data, label);
	svNum = 0;
	for(int i = 0; i < data.rows; i++)
	{
		Mat K;
		kernelTrans(svData, data.row(i), K, KERNEL_TYPE_RBF, 10);
		Mat t = K.t()*svLA;
		double predict = t.at<double>(0, 0) + b;
		if(predict*label.at<double>(i, 0) < 0)
			svNum++;
	}

	printf("Test Errors: %d / %d\n", svNum, data.rows);
}
