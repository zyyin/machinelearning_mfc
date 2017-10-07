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

#include "StdAfx.h"
#include "MLTree.h"

void MLTree::print()
{
	print(root, 0);
}

TREE_TYPE MLTree::getFirstBranchValue(TreeNode* node)
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
		TREE_TYPE ret = getFirstBranchValue(node->firstChild);
		if(ret!= 0) return ret;
		return getFirstBranchValue(node->nextSibling);
	}

}
void MLTree::deleteNode(TreeNode* node)
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


void MLTree::print(TreeNode* node, int num)
{
	if(node != NULL){
		printSpace(num); 
		cout << node->element << " " << node->val <<endl;  
		print(node->firstChild, num+4);
		print(node->nextSibling, num);
	}
}

void MLTree::preOrder()
{
	cout << "preOrder: ";
	preOrder(root);
	cout << endl;
}

void MLTree::preOrder(TreeNode* parent)
{
	if(parent != NULL){
		cout << parent->element << " " << parent->val <<" ";
		preOrder(parent->firstChild);
		preOrder(parent->nextSibling);
	}
}

MLTree::MLTree()
{
	root = NULL;
}

MLTree::~MLTree()
{
	deleteNode(root);
}

void MLTree::addNode(TreeNode* parent, TreeNode* node)
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

void MLTree::addBrotherNode(TreeNode* bro, TreeNode* node)
{
	if(bro->nextSibling == NULL){
		bro->nextSibling = node;
		node->preSibling = bro;
	}
	else
		addBrotherNode(bro->nextSibling, node);
}