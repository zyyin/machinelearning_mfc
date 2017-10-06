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

#ifndef MACHINE_LEARNING
#define MACHINE_LEARNING


#define LN_E 2.718281828459045235360287
// kNN
void TestKNN();
void handwritingClassTest();

// trees
float calcShannonEnt(Mat& data);
void testTree();

// bayes
void testBayes();


// SVM
void testSVM();
void testRBF();
void testSVMHandWriting();


// adaboost
void testSimpleData();
void testAdaBoost();

#endif