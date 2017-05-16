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

int chooseBestFeatureToSplit(Mat& data)
{
	int numFeatures = data.cols -1;
	float baseEntropy = calcShannonEnt(data);
	printf("baseEntropy = %.5f\n", baseEntropy);
	float bestInfoGain = 0.0;
	float bestFeature = -1;
	
	for(int i = 0; i< numFeatures; i++)
	{
		Mat featList(1, data.rows, CV_32F);
		set<float> s;
		for(int j = 0; j < data.rows; j++)
		{
			featList.at<float>(0, j) = data.at<float>(j, i);
			s.insert(data.at<float>(j, i));
		}

		float newEntropy = 0.0;
		for(set<float>::iterator j = s.begin(); j != s.end(); j++)
		{
			//cout<<*j<<"  "<<s.count(*j)<<endl;
			Mat dst;
			splitDataSet(data, i, *j, dst);
			
			float prob = dst.rows;
			prob /= float(data.rows);
			cout<<"prob  " << dst.rows<<endl;
			newEntropy += prob * calcShannonEnt(dst) ;    
			cout<<"newent   "<< newEntropy<<endl;
		}
		float infoGain = baseEntropy - newEntropy;
		if (infoGain > bestInfoGain)
		{
			bestInfoGain = infoGain;
			bestFeature = i;
		}
		
	}
	return bestFeature;
}

#define TYPE float

//树的节点
struct TreeNode{
	TYPE element;//该节点的元素
	TreeNode *firstChild;//指向该节点的第一个孩子
	TreeNode *nextSibling;//指向该节点的兄弟节点

};

class Tree{
public:
	Tree();
	~Tree();

	void addNode(int i, int j); 
	void preOrder();//前序遍历
	void print();//打印
private:
	void print(TreeNode* node, int num);
	void addBrotherNode(TreeNode* bro, TreeNode* node);
	void preOrder(TreeNode* parent);//前序遍历
private:
	TreeNode * root;//该树的根
};

//打印树的形状
void Tree::print()
{
	print(root, 0);
}

void printSpace(int num)
{
	int i = 0;
	for(i = 0; i < num-3; i++)
		cout << " ";

	for(; i < num-2; ++i)
		cout << "|";
	for(; i < num; ++i)
		cout << "_";
}


void Tree::print(TreeNode* node, int num)
{
	if(node != NULL){
		printSpace(num); 
		cout << node->element << endl;  
		print(node->firstChild, num+4);
		print(node->nextSibling, num);
	}
}

//前序遍历
void Tree::preOrder()
{
	cout << "前序遍历: ";
	preOrder(root);
	cout << endl;
}

void Tree::preOrder(TreeNode* parent)
{
	if(parent != NULL){
		cout << parent->element << " ";
		preOrder(parent->firstChild);
		preOrder(parent->nextSibling);
	}
}

//分配并初始化所有的树结点
Tree::Tree()
{
	root = NULL;
}

//释放所有节点的内存空间
Tree::~Tree()
{
	if(root != NULL)
		delete [] root;
}

//addNode将父子结点组对
//如果父节点的firstChild==NULL, 则firstChild = node;
//如果父节点的firstChild != NULL, 则
void Tree::addNode(int i, int j)
{
	TreeNode* parent = &root[i];
	TreeNode* node = &root[j];

	if(parent->firstChild == NULL)
		parent->firstChild = node;
	else
		addBrotherNode(parent->firstChild, node);
}

//将节点插入到兄弟节点
void Tree::addBrotherNode(TreeNode* bro, TreeNode* node)
{
	if(bro->nextSibling == NULL)
		bro->nextSibling = node;
	else
		addBrotherNode(bro->nextSibling, node);
}

float majorityCnt(Mat& classList)
{
	map<float, int> classCount;
	for(int i = 0; i < classList.cols; i++)
	{
		map<float, int>::iterator it = classCount.begin();
		float vote = classList.at<float>(0, i);
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
	float maxKey = 0;
	int maxVal = 0;
	map<float, int>::iterator it = classCount.begin();
	for(; it != classCount.end(); it++)
	{
		if(maxVal < it->second) {
			maxVal = it->second;
			maxKey = it->first;
		}
	}
	return maxKey;
}
float createTree(Mat& data, Mat& label, Tree& tree)
{
	Mat classList(1, data.rows, CV_32F);
	for(int i = 0; i < data.rows; i++)
	{
		classList.at<float>(0, i) = data.at<float>(i, data.cols -1);
	}
	cout<<classList<<endl;
	// check classList
	bool allEqual = true;
	for(int i = 0; i < classList.cols; i++)
	{
		if(classList.at<float>(0, i) != classList.at<float>(0, 0))
		{
			allEqual = false;
			break;
		}
	}
	if(allEqual)
		return classList.at<float>(0, 0);
	if(data.cols == 1)
	{
		return majorityCnt(classList);
	}
	int best = chooseBestFeatureToSplit(data);
	float bestLabel = label.at<float>(0, best);
	
	// TODO

}
void testTree()
{
	float c[15] = {1, 1, 101, 1, 1, 101, 1, 0, 100, 0, 1, 100, 0, 1, 100};
	CvMat data = cvMat(5, 3, CV_32F, c);
	float d[2] = {201, 202};
	CvMat label = cvMat(1, 2, CV_32F, d);
	int n = chooseBestFeatureToSplit(Mat(&data));
	cout<< n <<endl;
}