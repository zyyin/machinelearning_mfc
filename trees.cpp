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
	return row;
}

int chooseBestFeatureToSplit(Mat& data)
{
	int numFeatures = data.cols -1;
	float baseEntropy = calcShannonEnt(data);
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
			newEntropy += prob * calcShannonEnt(dst) ;    
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

struct TreeNode{
	TYPE element;
	
	TreeNode *firstChild;
	TreeNode *preSibling;
	TreeNode *nextSibling;
	float val;
	TreeNode(){val = 0;firstChild = preSibling = nextSibling = NULL;}

};

class Tree{
public:
	Tree();
	~Tree();

	void addNode(TreeNode* parent, TreeNode* node);
	float getFirstBranchValue(TreeNode* node);
	void deleteNode(TreeNode* node);
	void preOrder();
	void print();
private:
	void print(TreeNode* node, int num);
	void addBrotherNode(TreeNode* bro, TreeNode* node);
	void preOrder(TreeNode* parent);
public:
	TreeNode * root;
};

//´òÓ¡Ê÷µÄÐÎ×´
void Tree::print()
{
	print(root, 0);
}

float Tree::getFirstBranchValue(TreeNode* node)
{
	if(node == NULL)
	{
		return 0;
	}
	
	if(node->firstChild == NULL && node->nextSibling == NULL && node->preSibling == NULL) // leaf
	{
		return 0;
	} else if(node->firstChild != NULL && ( node->preSibling != NULL || node->nextSibling != NULL )) {
		return node->element;
	}
	else {
		float ret = getFirstBranchValue(node->firstChild);
		if(ret!= 0) return ret;
		return getFirstBranchValue(node->nextSibling);
	}
	
}
void Tree::deleteNode(TreeNode* node)
{
    if(node == NULL) return;
    deleteNode(node->firstChild);
    deleteNode(node->nextSibling);
    delete node;
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
		cout << node->element << " " << node->val <<endl;  
		print(node->firstChild, num+4);
		print(node->nextSibling, num);
	}
}

void Tree::preOrder()
{
	cout << "preOrder: ";
	preOrder(root);
	cout << endl;
}

void Tree::preOrder(TreeNode* parent)
{
	if(parent != NULL){
		cout << parent->element << " " << parent->val <<" ";
		preOrder(parent->firstChild);
		preOrder(parent->nextSibling);
	}
}

Tree::Tree()
{
	root = NULL;
}

Tree::~Tree()
{
	deleteNode(root);
}

void Tree::addNode(TreeNode* parent, TreeNode* node)
{
	if(parent == NULL)
	{
	     root = node;
	     return;
	}
	if(parent->firstChild == NULL)
		parent->firstChild = node;
	else
		addBrotherNode(parent->firstChild, node);
}

void Tree::addBrotherNode(TreeNode* bro, TreeNode* node)
{
	if(bro->nextSibling == NULL){
		bro->nextSibling = node;
		node->preSibling = bro;
	}
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
float createTree(Mat& data, Mat& label, Tree& tree, TreeNode* parent, int iVal)
{
	Mat classList(1, data.rows, CV_32F);
	for(int i = 0; i < data.rows; i++)
	{
		classList.at<float>(0, i) = data.at<float>(i, data.cols -1);
	}
	
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
	if(allEqual) {
		cout<<"1111"<<endl;
		TreeNode* node = new TreeNode();
		node->element = classList.at<float>(0, 0);
		node->val = iVal;
               	tree.addNode(parent, node);
		return 0;
	}
	if(data.cols == 1)
	{
		cout<<"222"<<endl;
		TreeNode* node = new TreeNode();
		node->element = majorityCnt(classList);
		node->val = iVal;
                tree.addNode(parent, node);
		return 0;
	}
	int best = chooseBestFeatureToSplit(data);
	float bestLabel = label.at<float>(0, best);
	TreeNode* node = new TreeNode();
	node->element = bestLabel;
	node->val = iVal;
        tree.addNode(parent, node);
	DeleteOneColOfMat(label, best);
	Mat featValues = data.col(best);
	cout<<featValues<<endl;
        set<float> s;
	for(int i = 0; i < featValues.rows; i++)
	{
		s.insert(featValues.at<float>(i, 0));
	}
	set<float>::iterator j = s.begin();
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

TreeNode* findRoot(TreeNode* node,float key)
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

float classifyTree(TreeNode* node, Mat& label, Mat& testData)
{
	float first = node->element;
	TreeNode* second = node->firstChild;
	int featIndex = 0; 
	for(; featIndex < label.cols; featIndex++)
	{
		if(first == label.at<float>(0, featIndex))
			break;
	}
	float key = testData.at<float>(0, featIndex);
	TreeNode* valueOfFeat = findRoot(second, key);
	if(valueOfFeat) {
		float classLabel = 0;
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
	float c[15] = {1, 1, 101, 1, 1, 101, 1, 0, 100, 0, 1, 100, 0, 1, 100};
	CvMat data = cvMat(5, 3, CV_32F, c);
	float d[2] = {201, 202};
        float e[2] = {1, 0};
	CvMat label = cvMat(1, 2, CV_32F, d);
	CvMat testData = cvMat(1, 2, CV_32F, e);
	Mat mData(&data);
	Mat mLabel(&label);
	Mat mTestData(&testData);
	int n = chooseBestFeatureToSplit(mData);
	cout<< n <<endl;
	Tree tree;
	createTree(mData, mLabel, tree, NULL, -1);
	cout<<"################"<<endl;
	tree.print();
	float ret = classifyTree(tree.root, mLabel, mTestData);
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
