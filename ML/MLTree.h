#pragma once

#define TREE_TYPE double

struct TreeNode{
	TREE_TYPE element;

	TreeNode *firstChild;
	TreeNode *preSibling;
	TreeNode *nextSibling;
	TREE_TYPE val;
	TreeNode(){val = 0;firstChild = preSibling = nextSibling = NULL;}

};

class MLTree{
public:
	MLTree();
	~MLTree();

	void addNode(TreeNode* parent, TreeNode* node);
	TREE_TYPE getFirstBranchValue(TreeNode* node);
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
