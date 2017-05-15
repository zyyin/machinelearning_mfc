#include "stdafx.h"
#include "MachineLearning.h"


class sortType {
public:
	int inx;
	float val;
	sortType(int _inx, float _val)
	{
		inx = _inx;
		val = _val;
	}
	bool operator < (const sortType& r)
	{
		return val < r.val;
	}
};

class countType {
public:
	int count;
	float val;
	countType(float _val)
	{
		count = 0;
		val = _val;
	}
	bool operator < (const countType& r)
	{
		return count < r.count;
	}
};

/*
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
dataSet: size m data set of known vectors (NxM)
labels: data set labels (1xM vector)
k: number of neighbors to use for comparison (should be an odd number)

Output:     the most popular class label
*/
float kNN(Mat& inx, Mat& dataSet, Mat& labels,  int k)
{
	int m = dataSet.rows;
	int n = inx.cols;
	if(n != dataSet.cols)
		return 0;
	vector<sortType> vecDistance;
	for(int i = 0; i < m; i++)
	{
		float distance = 0;
		float* pData = (float*)dataSet.data +i*n;
		float* iData = (float*)inx.data;
		for(int j = 0; j < n; j++)
		{
			float d = pData[j] - iData[j];
			distance +=  d*d;
		}
		vecDistance.push_back(sortType(i, sqrt(distance)));
	}

	sort(vecDistance.begin(), vecDistance.end());

	vector<countType> vecCount;
	float* data = (float*)labels.data;
	for(int i = 0; i < k; i++)
	{
		float vLable = data[ vecDistance[i].inx];
		int j = 0;
		for(; j < vecCount.size(); j++)
		{
			if(vLable == vecCount[j].val)
			{
				vecCount[j].count++;
				break;
			}
		}
		if( j == vecCount.size())	
		{
			vecCount.push_back(countType(vLable));
			vecCount.back().count++;
		}

	}
	sort(vecCount.begin(), vecCount.end());
	return vecCount.back().val;
}

// kNN test
void TestKNN()
{
	float a[2] = {0.5, 0.6};
	CvMat inx = cvMat(1, 2, CV_32F, a);
	float b[4] = {101, 101, 102, 102};
	CvMat labels = cvMat(1, 4, CV_32F, b);
	float c[8] = {1.0, 1.1, 1.0, 1.0, 0, 0, 0, 0.1};
	CvMat data = cvMat(4, 2, CV_32F, c);
	float ret = kNN(Mat(&inx), Mat(&data), Mat(&labels), 3);
	Mat m(&inx);
	cv::normalize(m, m, 1, 0, NORM_MINMAX); 

	cout<<m<<endl;
	printf("ret = %.2f \n", ret);
}


void BrowseFolder(const char* folder, vector<CString>& fileList, vector<CString>& nameList)
{
	CFileFind finder;
	CString strPath;
	CString str = folder;
	str += "\\*.*";
	BOOL bWorking = finder.FindFile(str);
	while (bWorking)
	{
		bWorking = finder.FindNextFile();
		if(!finder.IsDirectory() && !finder.IsDots()) {

			strPath=finder.GetFilePath();
			fileList.push_back(strPath);
			nameList.push_back(finder.GetFileName());
		}
	}
}

void handwritingClassTest()
{
	vector<CString> fileList;
	vector<CString> nameList;
	BrowseFolder("trainingDigits", fileList, nameList);
	Mat data(nameList.size(), 32*32, CV_32F);
	Mat lable(1, nameList.size(), CV_32F);
	int printed = 0;
	for(int i = 0; i < fileList.size(); i++)
	{
		char* c = (char*)(LPCTSTR)nameList[i];
		printf("%s, %d\n", (LPCTSTR)fileList[i], atoi(c));
		CStdioFile file(fileList[i], CFile::modeRead);
		CString str;
		int line = 0;
		while(file.ReadString(str)){
			char* p = (char*)(LPCTSTR)str;
			for(int j = 0; j < 32; j++)
			{
				data.at<float>(i, j+line*32) = p[j] - '0';
			}
			line++;
		}
		file.Close();
		lable.at<float>(0, i) = atoi(c);
	}

	fileList.clear();
	nameList.clear();
	BrowseFolder("testDigits", fileList, nameList);
	float errorNumber = 0;
	for(int i = 0; i< fileList.size(); i++)
	{
		char* c = (char*)(LPCTSTR)nameList[i];
		Mat inx(1, 32*32, CV_32F);
		CStdioFile file(fileList[i], CFile::modeRead);
		CString str;
		int line = 0;
		float* iData = (float*)inx.data;
		while(file.ReadString(str)){
			char* p = (char*)(LPCTSTR)str;
			for(int j = 0; j < 32; j++)
			{
				iData[j+line*32] = p[j] - '0';
			}
			line++;
		}
		file.Close();
		float ret = kNN(inx, data, lable, 3);
		printf("%s, %.2f\n", (LPCTSTR)fileList[i], ret);
		if(ret != atoi(c))
			errorNumber += 1;
	}

	printf("\n\n\n############ %.1f,  %.2f", errorNumber, errorNumber/fileList.size());

}